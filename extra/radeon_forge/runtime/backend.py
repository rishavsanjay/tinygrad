from __future__ import annotations

import json, queue, subprocess, threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class BackendCapabilities:
  streaming: bool = True
  structured_tools: bool = False
  prefix_cache: bool = False
  persistent_kv: bool = False
  kernel_metrics: bool = False
  cancellation: bool = False
  batched_prefill: bool = False
  stage_hooks: bool = False
  local_only: bool = True


@dataclass(frozen=True)
class GenerationRequest:
  session_id: str
  messages: Sequence[Mapping[str, Any]]
  tools: Sequence[Mapping[str, Any]] = ()
  max_tokens: int = 256
  temperature: float = 0.0
  stop: Sequence[str] = ()
  metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationEvent:
  kind: str
  text: str = ""
  tool_call: Mapping[str, Any] | None = None
  finish_reason: str | None = None
  metrics: Mapping[str, Any] = field(default_factory=dict)


class InferenceBackend(Protocol):
  @property
  def name(self) -> str: ...
  @property
  def capabilities(self) -> BackendCapabilities: ...
  @property
  def runtime_metadata(self) -> Mapping[str, Any]: ...
  def stream(self, request: GenerationRequest) -> Iterator[GenerationEvent]: ...
  def close(self) -> None: ...


class JsonlProcessBackend:
  """Long-lived local model subprocess using a strict JSON-lines protocol."""
  def __init__(self, command: Sequence[str], env: Mapping[str, str] | None = None):
    import os
    child_env = os.environ.copy()
    child_env.update(env or {})
    self._proc = subprocess.Popen(list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, bufsize=1, env=child_env)
    if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None: raise RuntimeError("failed to open model process pipes")
    self._stdin, self._stdout, self._stderr = self._proc.stdin, self._proc.stdout, self._proc.stderr
    self._lock = threading.RLock()
    self._stderr_lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=self._drain_stderr, daemon=True).start()
    ready = self._stdout.readline()
    if not ready: raise RuntimeError(f"model process exited during startup: {self.stderr_tail()}")
    payload = json.loads(ready)
    if payload.get("kind") != "ready": raise RuntimeError(f"invalid backend handshake: {payload}")
    self._name = str(payload.get("name", "jsonl-local-model"))
    self._capabilities = BackendCapabilities(**payload.get("capabilities", {}))
    self._runtime_metadata = {str(key): value for key, value in payload.items() if key not in {"kind", "name", "capabilities"}}

  @property
  def name(self) -> str: return self._name
  @property
  def capabilities(self) -> BackendCapabilities: return self._capabilities
  @property
  def runtime_metadata(self) -> Mapping[str, Any]: return dict(self._runtime_metadata)

  def _drain_stderr(self) -> None:
    for line in self._stderr:
      self._stderr_lines.put(line.rstrip())
      while self._stderr_lines.qsize() > 200:
        try: self._stderr_lines.get_nowait()
        except queue.Empty: break

  def stderr_tail(self, n: int = 20) -> str:
    lines = list(self._stderr_lines.queue)
    return "\n".join(lines[-n:])

  def stream(self, request: GenerationRequest) -> Iterator[GenerationEvent]:
    with self._lock:
      self._stdin.write(json.dumps({"op": "generate", "request": {
        "session_id": request.session_id, "messages": list(request.messages), "tools": list(request.tools),
        "max_tokens": request.max_tokens, "temperature": request.temperature, "stop": list(request.stop),
        "metadata": dict(request.metadata)}}) + "\n")
      self._stdin.flush()
      while True:
        line = self._stdout.readline()
        if not line: raise RuntimeError(f"model process terminated: {self.stderr_tail()}")
        payload = json.loads(line)
        if payload.get("kind") == "error": raise RuntimeError(str(payload.get("error", "backend error")))
        ev = GenerationEvent(kind=str(payload["kind"]), text=str(payload.get("text", "")),
                             tool_call=payload.get("tool_call"), finish_reason=payload.get("finish_reason"),
                             metrics=payload.get("metrics", {}))
        yield ev
        if ev.kind == "done": break

  def close(self) -> None:
    if self._proc.poll() is not None: return
    try:
      self._stdin.write(json.dumps({"op": "shutdown"}) + "\n")
      self._stdin.flush()
      self._proc.wait(timeout=5)
    except Exception: self._proc.kill()
