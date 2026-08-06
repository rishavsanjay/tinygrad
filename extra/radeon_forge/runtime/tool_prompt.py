from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


TOOL_PROTOCOL_MARKER = "Radeon Forge local tool protocol"


def render_tool_instruction(tools: Sequence[Mapping[str, Any]]) -> str:
  rows = []
  for item in tools:
    function = item.get("function", {}) if isinstance(item, Mapping) else {}
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str): continue
    rows.append({"name": function["name"], "description": str(function.get("description", "")),
                 "parameters": function.get("parameters", {"type":"object","properties":{}})})
  if not rows: return ""
  return f"""{TOOL_PROTOCOL_MARKER}.
You may call only one of the tools listed below. When a tool is necessary, output exactly one object and no surrounding prose:
<tool_call>{{"name":"tool_name","arguments":{{...}}}}</tool_call>
Do not invent a result. Tool execution requires user approval and the result will be returned in a later message.
Available tools:
{json.dumps(rows, indent=2, sort_keys=True)}"""


def inject_tool_instruction(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
  copied = [dict(message) for message in messages]
  instruction = render_tool_instruction(tools)
  if not instruction: return copied
  if any(TOOL_PROTOCOL_MARKER in str(message.get("content", "")) for message in copied if message.get("role") == "system"):
    return copied
  index = next((i for i, message in enumerate(copied) if message.get("role") == "system"), None)
  if index is None: copied.insert(0, {"role":"system", "content":instruction})
  else: copied[index]["content"] = str(copied[index].get("content", "")).rstrip() + "\n\n" + instruction
  return copied
