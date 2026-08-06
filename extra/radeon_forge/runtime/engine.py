from __future__ import annotations

import threading, uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..permissions import PermissionController
from ..profiling.report import build_profile_report
from ..synthesis import CandidateWorkspace, HookRegistry, OptimizationTools, RuntimeFingerprint, install_default_specs
from .backend import InferenceBackend
from .jobs import LocalJobManager
from .session import AgentSession
from .tools import ToolCall, ToolRegistry, WorkspaceTools


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
    self.jobs = LocalJobManager()
    self._sessions: dict[str, AgentSession] = {}
    self._lock = threading.RLock()

  def runtime_fingerprint(self) -> RuntimeFingerprint:
    metadata = getattr(self.backend, "runtime_metadata", {})
    if callable(metadata): metadata = metadata()
    return RuntimeFingerprint.from_mapping(metadata if isinstance(metadata, Mapping) else {})

  def _session_metadata(self, session: AgentSession, step: int) -> Mapping[str, Any]:
    metadata = self.hooks.runtime_metadata()
    for item in metadata.get("active_hooks", []):
      try:
        record = self.optimization_workspace.load_candidate(str(item["candidate_id"]))
        item["parameters"] = dict(record.evidence.get("selected_parameters", {}))
      except Exception: item["parameters"] = {}
    return {**metadata, "runtime_fingerprint": asdict(self.runtime_fingerprint())}

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

  def submit_message(self, session_id: str, content: str, max_tokens: int = 512, temperature: float = 0.0) -> dict[str, Any]:
    session = self.session(session_id)
    job = self.jobs.submit("agent_turn", session_id, lambda: session.send(content, max_tokens, temperature))
    return asdict(job)

  def submit_tool_approval(self, session_id: str, reason: str, max_tokens: int = 512) -> dict[str, Any]:
    session = self.session(session_id)
    token = self.grant_for_pending_tool(session_id, reason)
    job = self.jobs.submit("tool_and_resume", session_id, lambda: session.approve_tool(token, max_tokens))
    return asdict(job)

  def grant_for_pending_tool(self, session_id: str, reason: str, max_uses: int = 1) -> str:
    session = self.session(session_id)
    if session.pending_tool_call is None: raise RuntimeError("session has no pending tool")
    action = self.tools.spec(session.pending_tool_call.name).action
    return self.permissions.issue([action], reason, max_uses=max_uses).token

  def execute_explicit_ui_tool(self, name: str, arguments: Mapping[str, Any], reason: str) -> Any:
    """Execute one direct UI mutation through the same scoped permission boundary as the agent."""
    spec = self.tools.spec(name)
    grant = self.permissions.issue([spec.action], reason.strip() or f"Explicit local UI action: {name}", max_uses=1)
    result = self.tools.execute(ToolCall(uuid.uuid4().hex, name, dict(arguments)), grant.token)
    if not result.ok:
      if isinstance(result.output, Mapping) and result.output.get("error"): raise RuntimeError(str(result.output["error"]))
      raise RuntimeError(f"{name} failed")
    return result.output

  def profile(self, session_id: str) -> dict[str, Any]: return build_profile_report(self.session(session_id).trace.events())

  def optimization_state(self) -> dict[str, Any]:
    return {"specs": [asdict(x) | {"spec_id": x.spec_id} for x in self.optimization_workspace.specs()],
            "candidates": [asdict(x) for x in self.optimization_workspace.candidates()],
            "recipes": [asdict(x) for x in self.optimization_tools.recipes.installed()],
            "active_hooks": [asdict(x) for x in self.hooks.active()],
            "runtime_fingerprint": asdict(self.runtime_fingerprint())}

  def close(self) -> None: self.backend.close()
