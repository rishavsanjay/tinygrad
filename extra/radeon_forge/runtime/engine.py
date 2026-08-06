from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..permissions import PermissionController
from ..profiling.report import build_profile_report
from ..synthesis import CandidateWorkspace, HookRegistry, OptimizationTools, RuntimeFingerprint, install_default_specs
from .backend import InferenceBackend
from .session import AgentSession
from .tools import ToolRegistry, WorkspaceTools


DEFAULT_SYSTEM_PROMPT = """You are Radeon Forge, a private local software and inference performance engineer.
Use evidence before making performance claims. Separate observations, inferences and unknowns. Prefer reversible changes and preserve correctness. You may inspect the private workspace, author disposable target-specific kernel candidates, import portable optimization recipes, validate implementations through tinygrad MockGPU, and request permission for real W7900 benchmarks. Never treat MockGPU timing as performance evidence.

Optimization recipes are contracts, not programming languages. A hook has two independent coordinates: where it intercepts the stack (scheduler, model block, subgraph, kernel, KV cache, sampler) and when it applies (prefill, first token, steady decode, tool resume, KV append, sampling, or a workload predicate). Read the free-form intent, invariants, oracle, stage predicate, knowledge and failed experiments; then use your judgment to regenerate or radically restructure implementations. Cached source is merely one prior compilation and must never outrank the oracle."""


class ForgeEngine:
  def __init__(self, backend: InferenceBackend, workspace: str | Path, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    self.backend, self.workspace, self.system_prompt = backend, Path(workspace).resolve(), system_prompt
    self.permissions = PermissionController()
    self.tools = ToolRegistry(self.permissions)
    WorkspaceTools(self.workspace).install(self.tools)
    self.optimization_workspace = CandidateWorkspace(self.workspace / ".radeon_forge")
    install_default_specs(self.optimization_workspace)
    self.hooks = HookRegistry(self.optimization_workspace)
    self.optimization_tools = OptimizationTools(self.optimization_workspace, self.workspace, self.hooks, self.runtime_fingerprint)
    self.optimization_tools.install(self.tools)
    self._sessions: dict[str, AgentSession] = {}
    self._lock = threading.RLock()

  def runtime_fingerprint(self) -> RuntimeFingerprint:
    metadata = getattr(self.backend, "runtime_metadata", {})
    if callable(metadata): metadata = metadata()
    return RuntimeFingerprint.from_mapping(metadata if isinstance(metadata, Mapping) else {})

  def _session_metadata(self, session: AgentSession, step: int) -> Mapping[str, Any]:
    # Send every validated active hook to the isolated backend. The backend
    # resolves execution-state predicates at prefill/first-token/decode/tool
    # resume boundaries using live context that the outer agent cannot fake.
    return {**self.hooks.runtime_metadata(), "runtime_fingerprint": asdict(self.runtime_fingerprint())}

  def create_session(self) -> AgentSession:
    with self._lock:
      session = AgentSession(self.backend, self.tools, self.system_prompt, metadata_provider=self._session_metadata)
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

  def optimization_state(self) -> dict[str, Any]:
    return {"specs": [asdict(x) | {"spec_id": x.spec_id} for x in self.optimization_workspace.specs()],
            "candidates": [asdict(x) for x in self.optimization_workspace.candidates()],
            "recipes": [asdict(x) for x in self.optimization_tools.recipes.installed()],
            "active_hooks": [asdict(x) for x in self.hooks.active()],
            "runtime_fingerprint": asdict(self.runtime_fingerprint())}

  def close(self) -> None: self.backend.close()
