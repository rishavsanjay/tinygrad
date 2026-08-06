from __future__ import annotations

import json, re, uuid
from dataclasses import dataclass
from typing import Iterable, Mapping, Any


_TOOL_CALL = re.compile(r"\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*", re.S)


class ToolProtocolError(ValueError): pass


@dataclass(frozen=True)
class ParsedToolCall:
  call_id: str
  name: str
  arguments: Mapping[str, Any]
  raw: str

  def to_event(self) -> dict[str, Any]:
    return {"id": self.call_id, "name": self.name, "arguments": dict(self.arguments), "raw": self.raw}


class ToolStreamParser:
  """Incrementally detects the exact local tool-call protocol.

  It is not a grammar decoder yet; it is a bounded early-stop and validation
  layer. Malformed or unknown tool calls fail closed rather than being executed
  as best-effort text.
  """
  def __init__(self, allowed_names: Iterable[str], max_bytes: int = 65536):
    self.allowed_names = frozenset(str(name) for name in allowed_names)
    self.max_bytes = max_bytes
    self.buffer = ""
    self.completed = False

  def feed(self, text: str) -> ParsedToolCall | None:
    if self.completed: raise ToolProtocolError("tool stream is already complete")
    self.buffer += text
    if len(self.buffer.encode("utf-8")) > self.max_bytes: raise ToolProtocolError("tool-call stream exceeds size limit")
    if "</tool_call>" not in self.buffer: return None
    match = _TOOL_CALL.fullmatch(self.buffer)
    if match is None: raise ToolProtocolError("malformed tool call: output must contain exactly one wrapped JSON object")
    try: payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc: raise ToolProtocolError(f"tool call contains invalid JSON: {exc}") from exc
    if not isinstance(payload, dict): raise ToolProtocolError("tool-call payload must be an object")
    name, arguments = payload.get("name"), payload.get("arguments", {})
    if not isinstance(name, str) or not name: raise ToolProtocolError("tool call requires a non-empty string name")
    if name not in self.allowed_names: raise ToolProtocolError(f"unknown or unavailable local tool {name!r}")
    if not isinstance(arguments, dict): raise ToolProtocolError("tool call arguments must be an object")
    call_id = payload.get("id") or uuid.uuid4().hex
    if not isinstance(call_id, str): raise ToolProtocolError("tool call id must be a string")
    self.completed = True
    return ParsedToolCall(call_id, name, arguments, self.buffer)

  def finalize(self) -> None:
    stripped = self.buffer.lstrip()
    if stripped.startswith("<tool_call>") and not self.completed:
      raise ToolProtocolError("generation ended with an incomplete tool call")
