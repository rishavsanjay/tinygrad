from __future__ import annotations

import json, threading, time, uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from .backend import GenerationRequest, InferenceBackend
from .events import TraceRecorder
from .tools import ToolCall, ToolRegistry, parse_tool_call


class SessionState(str, Enum):
  IDLE = "idle"
  GENERATING = "generating"
  AWAITING_TOOL_APPROVAL = "awaiting_tool_approval"
  RUNNING_TOOL = "running_tool"
  COMPLETED = "completed"
  CANCELLED = "cancelled"
  FAILED = "failed"


@dataclass(frozen=True)
class SessionEvent:
  sequence: int
  kind: str
  data: Mapping[str, Any]
  timestamp_s: float = field(default_factory=time.time)


class AgentSession:
  """Persistent private-agent session with explicit tool approval and full tracing."""
  def __init__(self, backend: InferenceBackend, tools: ToolRegistry, system_prompt: str, session_id: str | None = None,
               max_agent_steps: int = 8, trace: TraceRecorder | None = None,
               metadata_provider: Callable[[AgentSession, int], Mapping[str, Any]] | None = None):
    self.session_id = session_id or uuid.uuid4().hex
    self.backend, self.tools, self.system_prompt = backend, tools, system_prompt
    self.max_agent_steps, self.metadata_provider = max_agent_steps, metadata_provider
    self.trace = trace or TraceRecorder()
    self.messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
    self.events: list[SessionEvent] = []
    self.pending_tool_call: ToolCall | None = None
    self.pending_assistant_content = ""
    self.partial_output = ""
    self.last_finish_reason: str | None = None
    self.state = SessionState.IDLE
    self._sequence = 0
    self._lock = threading.RLock()

  def _system_prompt(self) -> str:
    schemas = self.tools.schemas()
    tool_text = "\n".join(f"- {x['function']['name']}: {x['function']['description']} schema={x['function']['parameters']}" for x in schemas)
    return self.system_prompt.strip() + ("\n\nAvailable local tools:\n" + tool_text if schemas else "") + """

Tool protocol: when a tool is required, output exactly one object wrapped as
<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>
and no surrounding prose. Tool execution always requires user approval. Never
claim a tool result before it is returned. All inference and tools are local.
"""

  @staticmethod
  def _tool_content(call: ToolCall) -> str:
    payload = {"id": call.call_id, "name": call.name, "arguments": dict(call.arguments)}
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"

  @staticmethod
  def _looks_like_partial_tool_protocol(output: str) -> bool:
    stripped = output.lstrip()
    marker = "<tool_call>"
    return bool(stripped) and (marker.startswith(stripped) or stripped.startswith(marker))

  def _emit(self, kind: str, **data: Any) -> SessionEvent:
    self._sequence += 1
    event = SessionEvent(self._sequence, kind, data)
    self.events.append(event)
    return event

  def send(self, content: str, max_tokens: int = 512, temperature: float = 0.0) -> tuple[SessionEvent, ...]:
    with self._lock:
      if self.state in {SessionState.GENERATING, SessionState.RUNNING_TOOL}: raise RuntimeError("session is busy")
      if self.pending_tool_call is not None: raise RuntimeError("approve or reject the pending tool call first")
      text = content.strip()
      if not text: raise ValueError("message must not be empty")
      self.messages.append({"role": "user", "content": text})
      self._emit("message", role="user", content=text)
      return self._generate(max_tokens, temperature)

  def _record_backend_event(self, event, parent_id: str) -> None:
    metrics = dict(event.metrics)
    self._emit(event.kind, **metrics)
    if event.kind == "kernel":
      name = str(metrics.pop("name", metrics.pop("kernel_name", "kernel")))
      duration_ms = float(metrics.pop("duration_ms", metrics.pop("duration_us", 0.0) / 1000.0))
      self.trace.duration("kernel", name, duration_ms, parent_id, **metrics)
    elif event.kind in {"prefill", "decode"} and float(metrics.get("wall_ms", 0.0)) > 0:
      self.trace.duration("inference", event.kind, float(metrics["wall_ms"]), parent_id, **metrics)
    else:
      if "name" in metrics: metrics[f"{event.kind}_name"] = metrics.pop("name")
      if "kind" in metrics: metrics["backend_kind"] = metrics.pop("kind")
      if "parent_id" in metrics: metrics["backend_parent_id"] = metrics.pop("parent_id")
      self.trace.point("inference", event.kind, parent_id, **metrics)

  def _request_metadata(self, step: int) -> dict[str, Any]:
    metadata = {"trace_id": self.trace.trace_id, "step": step,
                "tool_round": sum(1 for event in self.events if event.kind == "tool_result"),
                "resume_after_tool": bool(self.messages and self.messages[-1].get("role") == "tool")}
    if self.metadata_provider is not None: metadata.update(dict(self.metadata_provider(self, step)))
    return metadata

  def _generate(self, max_tokens: int, temperature: float) -> tuple[SessionEvent, ...]:
    self.state = SessionState.GENERATING
    self.partial_output = ""
    self.pending_assistant_content = ""
    self.last_finish_reason = None
    started_at = len(self.events)
    pieces: list[str] = []
    done_metrics: dict[str, Any] = {}
    step = sum(1 for event in self.events if event.kind == "generation_started") + 1
    if step > self.max_agent_steps: raise RuntimeError("agent step limit exceeded")
    with self.trace.span("agent", "model_turn", session_id=self.session_id, step=step) as parent:
      self._emit("generation_started", backend=self.backend.name, step=step, capabilities=asdict(self.backend.capabilities))
      request = GenerationRequest(self.session_id, tuple(self.messages), tuple(self.tools.schemas()), max_tokens, temperature,
                                  metadata=self._request_metadata(step))
      try:
        for event in self.backend.stream(request):
          if event.kind == "token":
            pieces.append(event.text)
            self.partial_output += event.text
            self._emit("token", text=event.text, metrics=dict(event.metrics))
            token_metrics = dict(event.metrics)
            if "name" in token_metrics: token_metrics["token_name"] = token_metrics.pop("name")
            self.trace.duration("inference", "token", float(token_metrics.get("wall_ms", 0.0)), parent.event_id,
                                text=event.text, **token_metrics)
          elif event.kind in {"prefill", "decode", "kernel", "metric", "hook"}: self._record_backend_event(event, parent.event_id)
          elif event.kind == "tool_call" and event.tool_call is not None:
            arguments = event.tool_call.get("arguments", {})
            if not isinstance(arguments, Mapping): raise ValueError("backend tool-call arguments must be an object")
            self.pending_tool_call = ToolCall(str(event.tool_call.get("id") or uuid.uuid4().hex),
                                              str(event.tool_call["name"]), dict(arguments))
            raw = event.tool_call.get("raw")
            if isinstance(raw, str): self.pending_assistant_content = raw
            self._emit("structured_tool_call", call=asdict(self.pending_tool_call), native=True)
          elif event.kind == "done":
            self.last_finish_reason = event.finish_reason
            done_metrics = dict(event.metrics)
            self._emit("generation_done", finish_reason=event.finish_reason, metrics=done_metrics)
      except Exception as exc:
        self.state = SessionState.FAILED
        self._emit("error", error=str(exc), error_type=type(exc).__name__)
        raise
    output = "".join(pieces)

    if self.last_finish_reason == "cancelled":
      discarded_tool_prefix = self._looks_like_partial_tool_protocol(output)
      if output and not discarded_tool_prefix: self.messages.append({"role": "assistant", "content": output})
      self.pending_tool_call = None
      self.pending_assistant_content = ""
      self.partial_output = ""
      self.state = SessionState.CANCELLED
      self._emit("generation_cancelled", preserved_output=bool(output and not discarded_tool_prefix),
                 discarded_incomplete_tool_protocol=discarded_tool_prefix, generated_characters=len(output),
                 materialized_kv_tokens=done_metrics.get("materialized_kv_tokens"), cancel_stage=done_metrics.get("cancel_stage"))
      return tuple(self.events[started_at:])

    if self.pending_tool_call is None:
      try:
        self.pending_tool_call = parse_tool_call(output)
        if self.pending_tool_call is not None: self.pending_assistant_content = output
      except Exception as exc: self._emit("tool_parse_error", error=str(exc), raw=output)
    if self.pending_tool_call is not None:
      if not self.pending_assistant_content: self.pending_assistant_content = self._tool_content(self.pending_tool_call)
      self.partial_output = ""
      self.state = SessionState.AWAITING_TOOL_APPROVAL
      self._emit("tool_approval_required", call=asdict(self.pending_tool_call), action=self.tools.spec(self.pending_tool_call.name).action.value)
    else:
      self.messages.append({"role": "assistant", "content": output})
      self.partial_output = ""
      self.state = SessionState.COMPLETED
      self._emit("message", role="assistant", content=output)
    return tuple(self.events[started_at:])

  def approve_tool(self, permission_token: str, max_tokens: int = 512) -> tuple[SessionEvent, ...]:
    with self._lock:
      if self.state is not SessionState.AWAITING_TOOL_APPROVAL or self.pending_tool_call is None: raise RuntimeError("no tool call awaits approval")
      call = self.pending_tool_call
      assistant_content = self.pending_assistant_content or self._tool_content(call)
      started_at = len(self.events)
      self.state = SessionState.RUNNING_TOOL
      try:
        with self.trace.span("tool", call.name, call_id=call.call_id) as span:
          self._emit("tool_started", call=asdict(call))
          result = self.tools.execute(call, permission_token)
          self.trace.point("tool", "tool_result", span.event_id, ok=result.ok, elapsed_ms=result.elapsed_ms)
      except Exception as exc:
        self.state = SessionState.AWAITING_TOOL_APPROVAL
        self._emit("tool_authorization_failed", call_id=call.call_id, error=str(exc), error_type=type(exc).__name__)
        raise
      self._emit("tool_result", result=asdict(result))
      self.messages.append({"role": "assistant", "content": assistant_content})
      self.messages.append({"role": "tool", "name": call.name, "tool_call_id": call.call_id, "content": str(result.output)})
      self.pending_tool_call = None
      self.pending_assistant_content = ""
      if not result.ok:
        self.state = SessionState.FAILED
        return tuple(self.events[started_at:])
      self.state = SessionState.IDLE
      self._generate(max_tokens, 0.0)
      return tuple(self.events[started_at:])

  def reject_tool(self, reason: str) -> SessionEvent:
    with self._lock:
      if self.pending_tool_call is None: raise RuntimeError("no pending tool call")
      call = self.pending_tool_call
      assistant_content = self.pending_assistant_content or self._tool_content(call)
      self.messages.append({"role": "assistant", "content": assistant_content})
      self.messages.append({"role": "tool", "name": call.name, "tool_call_id": call.call_id, "content": f"User rejected tool call: {reason}"})
      self.pending_tool_call = None
      self.pending_assistant_content = ""
      self.state = SessionState.IDLE
      return self._emit("tool_rejected", call_id=call.call_id, reason=reason)

  def events_after(self, sequence: int) -> list[dict[str, Any]]:
    return [asdict(event) for event in self.events if event.sequence > sequence]

  def snapshot(self) -> dict[str, Any]:
    return {"session_id": self.session_id, "state": self.state.value, "messages": list(self.messages),
            "events": [asdict(x) for x in self.events], "pending_tool_call": asdict(self.pending_tool_call) if self.pending_tool_call else None,
            "pending_assistant_content": self.pending_assistant_content, "partial_output": self.partial_output,
            "last_finish_reason": self.last_finish_reason, "trace_id": self.trace.trace_id}
