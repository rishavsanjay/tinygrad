from __future__ import annotations

import json
from typing import Any

from .hooks import HookDescriptor, HookLayer
from .workspace import KernelSpec


def _doc(value: Any) -> str: return json.dumps(value, indent=2, sort_keys=True, default=str)


def render_candidate_scaffold(spec: KernelSpec) -> str:
  """Render the minimum executable contract, not a performance abstraction.

  The scaffold intentionally contains no tile DSL or scheduling framework. It
  gives an agent the exact runtime boundary, invariants, stage predicate and
  parameter handoff, while leaving the implementation structure disposable.
  """
  hook = HookDescriptor.from_spec(spec)
  header = f'''"""Disposable Radeon Forge candidate.

Spec: {spec.name}
Operation: {spec.operation}
Target: {spec.target}
Objective: {spec.objective}
Placement: {hook.layer.value}:{hook.target} ({hook.mode.value})
Execution states: {hook.when.signature}

Invariants:
{chr(10).join(f"- {item}" for item in spec.invariants)}

This file is not trusted merely because it imports or compiles. It must pass
MockGPU/reference, real hardware, and any held-out phase-transition oracle.
"""

from __future__ import annotations
from typing import Any, Mapping

RADEON_FORGE_PARAMETERS: dict[str, Any] = {{}}


def configure(parameters: Mapping[str, Any]) -> None:
  """Receive the locally autotuned schedule selected for this machine/state."""
  global RADEON_FORGE_PARAMETERS
  RADEON_FORGE_PARAMETERS = dict(parameters)

'''
  if hook.layer is HookLayer.TRANSFORMER_BLOCK:
    body = '''def build_replacement(original, layer_index: int, model, context: Mapping[str, Any]):
  """Return a callable with signature (x, start_pos, freqs_cis, mask).

  `context` describes the activation/validation state. Runtime execution-state
  selection is enforced outside this module by the signed hook manifest.
  Replace this identity implementation with direct tinygrad/custom-kernel code.
  """
  assert context.get("stage") in {"transition_oracle", "decode_benchmark", "first_token", "decode"}

  def replacement(x, start_pos, freqs_cis, mask):
    # TODO(agent): implement the model- and gfx1100-specific block/subgraph.
    # Keep `original` as the correctness fallback while iterating.
    return original(x, start_pos, freqs_cis, mask)

  return replacement
'''
  else:
    body = '''def build_kernel(*args, **kwargs):
  """Build the direct target-specific implementation for this hook contract."""
  raise NotImplementedError("agent must generate a direct implementation")
'''
  footer = f'''\n\n# Machine-readable context copied from the durable spec for code-review tools.\nFORGE_SPEC_CONTEXT = {_doc({
    "shapes":dict(spec.shapes), "dtypes":dict(spec.dtypes), "invariants":list(spec.invariants),
    "hook":{"layer":hook.layer.value, "target":hook.target, "mode":hook.mode.value,
            "adapter":hook.adapter, "selector":dict(hook.selector), "when":hook.when.signature},
    "search":spec.metadata.get("search", {}),
  })}\n'''
  return header + body + footer
