from __future__ import annotations

import json, statistics, time, uuid
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from ..runtime import ForgeEngine, GenerationEvent


_TOOL_PREFIX = "<tool_call>"


class OpenAIRequestError(ValueError): pass


def _request(payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], int, float, str, tuple[str, ...]]:
  messages = payload.get("messages")
  if not isinstance(messages, list) or not messages or not all(isinstance(item, Mapping) for item in messages):
    raise OpenAIRequestError("messages must be a non-empty array of objects")
  tools = payload.get("tools", [])
  if not isinstance(tools, list) or not all(isinstance(item, Mapping) for item in tools): raise OpenAIRequestError("tools must be an array")
  max_tokens = int(payload.get("max_tokens", payload.get("max_completion_tokens", 256)))
  if max_tokens <= 0: raise OpenAIRequestError("max_tokens must be positive")
  temperature = float(payload.get("temperature", 0.0))
  session_id = str(payload.get("session_id") or payload.get("user") or uuid.uuid4().hex)
  stop = payload.get("stop", ())
  if isinstance(stop, str): stop = (stop,)
  elif isinstance(stop, list) and all(isinstance(item, str) for item in stop): stop = tuple(stop)
  elif stop in (None, ()): stop = ()
  else: raise OpenAIRequestError("stop must be a string or array of strings")
  return list(messages), list(tools), max_tokens, temperature, session_id, tuple(stop)


def _tool_call(value: Mapping[str, Any]) -> dict[str, Any]:
  return {"id": str(value.get("id") or uuid.uuid4().hex), "type": "function",
          "function": {"name": str(value["name"]), "arguments": json.dumps(value.get("arguments", {}), separators=(",", ":"))}}


def _finish_reason(value: str | None) -> str | None: return "tool_calls" if value == "tool_call" else value


def _summarize(events: Sequence[GenerationEvent], session_id: str) -> dict[str, Any]:
  prefill = next((dict(event.metrics) for event in events if event.kind == "prefill"), {})
  tokens = [dict(event.metrics) for event in events if event.kind == "token"]
  by_stage: dict[str, list[float]] = defaultdict(list)
  kernel_calls: dict[str, int] = defaultdict(int)
  hook_events = []
  for event in events:
    if event.kind == "token":
      wall = event.metrics.get("wall_ms")
      if wall is not None: by_stage[str(event.metrics.get("stage", "decode"))].append(float(wall))
    elif event.kind == "kernel": kernel_calls[str(event.metrics.get("stage", "unknown"))] += 1
    elif event.kind == "hook": hook_events.append(dict(event.metrics))
  return {"session_id": session_id, "prefill": prefill,
          "token_latency_ms": {stage: {"count": len(values), "p50": statistics.median(values), "max": max(values)}
                               for stage, values in by_stage.items() if values},
          "kernel_calls_by_stage": dict(kernel_calls), "hook_transitions": hook_events,
          "generated_tokens": len(tokens)}


def collect_chat_completion(engine: ForgeEngine, payload: Mapping[str, Any]) -> dict[str, Any]:
  messages, tools, max_tokens, temperature, session_id, stop = _request(payload)
  events = list(engine.stream_inference(messages, tools, max_tokens, temperature, session_id, stop))
  text = "".join(event.text for event in events if event.kind == "token")
  tool = next((event.tool_call for event in events if event.kind == "tool_call" and event.tool_call is not None), None)
  done = next((event for event in reversed(events) if event.kind == "done"), None)
  prefill = next((event for event in events if event.kind == "prefill"), None)
  completion_tokens = int(done.metrics.get("generated_tokens", 0)) if done is not None else sum(event.kind == "token" for event in events)
  prompt_tokens = int(prefill.metrics.get("prompt_tokens", 0)) if prefill is not None else 0
  message: dict[str, Any] = {"role": "assistant", "content": None if tool is not None else text}
  if tool is not None: message["tool_calls"] = [_tool_call(tool)]
  return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
          "model": str(payload.get("model") or engine.backend.name),
          "choices": [{"index": 0, "message": message, "finish_reason": _finish_reason(done.finish_reason if done else None)}],
          "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens},
          "system_fingerprint": engine.runtime_fingerprint().architecture or None,
          "forge": _summarize(events, session_id)}


def _content_chunk(completion_id: str, created: int, model: str, text: str, include_role: bool) -> str:
  delta = {"content": text}
  if include_role: delta = {"role": "assistant", **delta}
  chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
           "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
  return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"


def _could_be_tool_protocol(text: str) -> bool:
  stripped = text.lstrip()
  return _TOOL_PREFIX.startswith(stripped) or stripped.startswith(_TOOL_PREFIX)


def stream_chat_completion(engine: ForgeEngine, payload: Mapping[str, Any]) -> Iterable[str]:
  messages, tools, max_tokens, temperature, session_id, stop = _request(payload)
  completion_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
  model = str(payload.get("model") or engine.backend.name)
  first, buffered, buffering_tool = True, "", False
  for event in engine.stream_inference(messages, tools, max_tokens, temperature, session_id, stop):
    if event.kind == "token":
      if first or buffering_tool:
        buffered += event.text
        if _could_be_tool_protocol(buffered):
          buffering_tool = True
          continue
        if buffered:
          yield _content_chunk(completion_id, created, model, buffered, first)
          first, buffered, buffering_tool = False, "", False
      else:
        yield _content_chunk(completion_id, created, model, event.text, False)
    elif event.kind == "tool_call" and event.tool_call is not None:
      # The native worker already validated the buffered text. Standard clients
      # receive only the structured delta, never Forge's internal text protocol.
      buffered, buffering_tool = "", False
      delta = {"tool_calls": [{"index": 0, **_tool_call(event.tool_call)}]}
      if first: delta = {"role": "assistant", **delta}; first = False
      chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
      yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
    elif event.kind == "done":
      if buffered:
        yield _content_chunk(completion_id, created, model, buffered, first)
        first, buffered = False, ""
      chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
               "choices": [{"index": 0, "delta": {}, "finish_reason": _finish_reason(event.finish_reason)}],
               "forge_session_id": session_id}
      yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
  yield "data: [DONE]\n\n"
