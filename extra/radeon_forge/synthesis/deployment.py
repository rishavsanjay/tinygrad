from __future__ import annotations

from .hooks import (ActiveHook, HookActivationError, HookDescriptor, HookLayer, HookRegistry, RuntimeFingerprint)
from .workspace import CandidateWorkspace


STATEFUL_LAYERS = frozenset({HookLayer.MODEL, HookLayer.TRANSFORMER_BLOCK, HookLayer.KV_CACHE})
RUNTIME_ADAPTERS = frozenset({"python_transformer_block"})


class SafeHookRegistry(HookRegistry):
  """Deployment policy layered over the generic stage resolver.

  The generic registry is useful for tests and future adapters. The engine uses
  this stricter registry so metadata-only or unimplemented adapters cannot be
  presented as active optimizations, and stateful phase changes cannot deploy
  without a held-out transition oracle.
  """
  SUPPORTED_ADAPTERS = RUNTIME_ADAPTERS

  def __init__(self, workspace: CandidateWorkspace): super().__init__(workspace)

  def activate(self, candidate_id: str, fingerprint: RuntimeFingerprint, reason: str) -> ActiveHook:
    candidate = self.workspace.load_candidate(candidate_id)
    spec = self.workspace.load_spec(candidate.spec_id)
    descriptor = HookDescriptor.from_spec(spec)
    if descriptor.adapter not in self.SUPPORTED_ADAPTERS:
      raise HookActivationError(
        f"runtime adapter {descriptor.adapter!r} is not executable in this engine build; "
        "the recipe remains usable as profiling/search knowledge"
      )
    if descriptor.layer in STATEFUL_LAYERS:
      heldout = tuple(spec.metadata.get("heldout_command", ()))
      if not heldout:
        raise HookActivationError(
          f"{descriptor.layer.value} hooks require a held-out phase-transition oracle before deployment "
          "(for example prefill -> first_token -> decode and tool-resume continuity)"
        )
      if candidate.status not in {"heldout_passed", "accepted"}:
        raise HookActivationError(
          f"stateful hook candidate status {candidate.status!r} has not passed its phase-transition oracle"
        )
    return super().activate(candidate_id, fingerprint, reason)
