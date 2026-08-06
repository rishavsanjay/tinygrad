from __future__ import annotations

import json, os, re, shlex, subprocess, time, uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..permissions import Action, PermissionController


@dataclass(frozen=True)
class ToolSpec:
  name: str
  description: str
  parameters: Mapping[str, Any]
  action: Action = Action.INSPECT

  def openai_schema(self) -> dict[str, Any]:
    return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": dict(self.parameters)}}


@dataclass(frozen=True)
class ToolCall:
  call_id: str
  name: str
  arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
  call_id: str
  name: str
  ok: bool
  output: Any
  elapsed_ms: float


@dataclass
class _RegisteredTool:
  spec: ToolSpec
  fn: Callable[[Mapping[str, Any]], Any]


class ToolRegistry:
  def __init__(self, permissions: PermissionController):
    self.permissions = permissions
    self._tools: dict[str, _RegisteredTool] = {}

  def register(self, spec: ToolSpec, fn: Callable[[Mapping[str, Any]], Any]) -> None:
    if spec.name in self._tools: raise ValueError(f"duplicate tool {spec.name}")
    self._tools[spec.name] = _RegisteredTool(spec, fn)

  def schemas(self) -> list[dict[str, Any]]: return [tool.spec.openai_schema() for tool in self._tools.values()]
  def spec(self, name: str) -> ToolSpec: return self._tools[name].spec

  def execute(self, call: ToolCall, permission_token: str | None) -> ToolResult:
    if call.name not in self._tools: return ToolResult(call.call_id, call.name, False, {"error": "unknown tool"}, 0.0)
    tool = self._tools[call.name]
    self.permissions.authorize(permission_token, tool.spec.action)
    started = time.perf_counter_ns()
    try: output, ok = tool.fn(call.arguments), True
    except Exception as exc: output, ok = {"error": str(exc), "type": type(exc).__name__}, False
    return ToolResult(call.call_id, call.name, ok, output, (time.perf_counter_ns() - started) / 1e6)


class WorkspaceTools:
  """Safe-by-construction tools rooted in one private repository workspace."""
  def __init__(self, root: str | Path, command_allowlist: Sequence[str] = ("python", "python3", "pytest", "git", "rg")):
    self.root = Path(root).resolve()
    self.command_allowlist = frozenset(command_allowlist)

  def _path(self, value: str) -> Path:
    path = (self.root / value).resolve()
    if path != self.root and self.root not in path.parents: raise ValueError("path escapes workspace")
    return path

  def list_files(self, args: Mapping[str, Any]) -> Any:
    base = self._path(str(args.get("path", ".")))
    limit = max(1, min(int(args.get("limit", 200)), 2000))
    files = []
    for path in base.rglob("*"):
      if path.is_file() and ".git" not in path.parts:
        files.append(str(path.relative_to(self.root)))
        if len(files) >= limit: break
    return {"files": files, "truncated": len(files) >= limit}

  def read_file(self, args: Mapping[str, Any]) -> Any:
    path = self._path(str(args["path"]))
    start, end = max(1, int(args.get("start_line", 1))), int(args.get("end_line", 400))
    if end < start or end - start > 2000: raise ValueError("invalid line range")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"path": str(path.relative_to(self.root)), "start_line": start, "end_line": min(end, len(lines)),
            "content": "\n".join(f"{i}: {lines[i-1]}" for i in range(start, min(end, len(lines)) + 1))}

  def search_text(self, args: Mapping[str, Any]) -> Any:
    pattern = re.compile(str(args["query"]))
    base = self._path(str(args.get("path", ".")))
    limit = max(1, min(int(args.get("limit", 50)), 500))
    hits = []
    for path in base.rglob("*"):
      if not path.is_file() or ".git" in path.parts or path.stat().st_size > 2_000_000: continue
      try: lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
      except OSError: continue
      for number, line in enumerate(lines, 1):
        if pattern.search(line):
          hits.append({"path": str(path.relative_to(self.root)), "line": number, "text": line[:500]})
          if len(hits) >= limit: return {"hits": hits, "truncated": True}
    return {"hits": hits, "truncated": False}

  def run_command(self, args: Mapping[str, Any]) -> Any:
    command = args.get("command")
    argv = shlex.split(command) if isinstance(command, str) else [str(x) for x in command]
    if not argv or Path(argv[0]).name not in self.command_allowlist: raise ValueError("command is not allowlisted")
    timeout = max(1, min(int(args.get("timeout_seconds", 60)), 600))
    proc = subprocess.run(argv, cwd=self.root, text=True, capture_output=True, timeout=timeout,
                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return {"argv": argv, "returncode": proc.returncode, "stdout": proc.stdout[-20000:], "stderr": proc.stderr[-20000:]}

  def install(self, registry: ToolRegistry) -> None:
    registry.register(ToolSpec("list_files", "List files inside the private workspace", {"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}}}), self.list_files)
    registry.register(ToolSpec("read_file", "Read a line range from a workspace file", {"type":"object","required":["path"],"properties":{"path":{"type":"string"},"start_line":{"type":"integer"},"end_line":{"type":"integer"}}}), self.read_file)
    registry.register(ToolSpec("search_text", "Regex-search private workspace text", {"type":"object","required":["query"],"properties":{"query":{"type":"string"},"path":{"type":"string"},"limit":{"type":"integer"}}}), self.search_text)
    registry.register(ToolSpec("run_command", "Run an allowlisted local development command", {"type":"object","required":["command"],"properties":{"command":{"type":["string","array"]},"timeout_seconds":{"type":"integer"}}}, Action.BENCHMARK), self.run_command)


def parse_tool_call(text: str) -> ToolCall | None:
  match = re.fullmatch(r"\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*", text, flags=re.S)
  if not match: return None
  payload = json.loads(match.group(1))
  name = payload.get("name")
  arguments = payload.get("arguments", {})
  if not isinstance(name, str) or not isinstance(arguments, dict): raise ValueError("invalid tool call payload")
  return ToolCall(str(payload.get("id") or uuid.uuid4().hex), name, arguments)
