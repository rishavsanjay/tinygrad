from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..permissions import PermissionController
from ..profiling.report import build_profile_report
from .backend import InferenceBackend
from .session import AgentSession
from .tools import ToolRegistry, WorkspaceTools


DEFAULT_SYSTEM_PROMPT = """You are Radeon Forge, a private local software and inference performance engineer.
Use evidence before making performance claims. Separate observations, inferences and unknowns. Prefer reversible changes and preserve correctness. You may inspect the private workspace and request permission to run local tools."""


class ForgeEngine:
  def __init__(self, backend: InferenceBackend, workspace: str | Path, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    self.backend, self.workspace, self.system_prompt = backend, Path(workspace).resolve(), system_prompt
    self.permissions = PermissionController()
    self.tools = ToolRegistry(self.permissions)
    WorkspaceTools(self.workspace).install(self.tools)
    self._sessions: dict[str, AgentSession] = {}
    self._lock = threading.RLock()

  def create_session(self) -> AgentSession:
    with self._lock:
      session = AgentSession(self.backend, self.tools, self.system_prompt)
      self._sessions[session.session_id] = session
      return session

  def session(self, session_id: str) -> AgentSession:
    try: return self._sessions[session_id]
    except KeyError as exc: raise KeyError(f"unknown session {session_id}") from exc

  def sessions(self) -> list[dict[str, Any]]:
    with self._lock: return [{"session_id": x.session_id, "state": x.state.value, "trace_id": x.trace.trace_id} for x in self._sessions.values()]

  def grant_for_pending_tool(self, session_id: str, reason: str, max_uses: int = 1) -> str:
    session = self.session(session_id)
    if session.pending_tool_call is None: raise RuntimeError("session has no pending tool")
    action = self.tools.spec(session.pending_tool_call.name).action
    return self.permissions.issue([action], reason, max_uses=max_uses).token

  def profile(self, session_id: str) -> dict[str, Any]: return build_profile_report(self.session(session_id).trace.events())

  def close(self) -> None: self.backend.close()
