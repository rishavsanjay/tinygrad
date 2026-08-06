from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class ToolExecutionContext:
  session_id: str
  trace_id: str
  agent_step: int
  tool_call_id: str
  tool_name: str
  attributes: Mapping[str, Any]


_CURRENT: ContextVar[ToolExecutionContext | None] = ContextVar("RADEON_FORGE_TOOL_CONTEXT", default=None)


@contextlib.contextmanager
def bind_tool_context(context: ToolExecutionContext) -> Iterator[None]:
  token = _CURRENT.set(context)
  try: yield
  finally: _CURRENT.reset(token)


def current_tool_context(required: bool = False) -> ToolExecutionContext | None:
  context = _CURRENT.get()
  if required and context is None: raise RuntimeError("tool execution provenance is unavailable")
  return context


def current_tool_context_dict() -> dict[str, Any]:
  context = current_tool_context()
  return asdict(context) if context is not None else {}
