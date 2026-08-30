from __future__ import annotations

import hashlib, itertools, json, os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..command_backend import CommandHarness
from ..contracts import Candidate, Objective, WorkloadContract
from ..ledger import ExperimentLedger
from ..tuner import TuningSummary, successive_halving
from .hooks import HookDescriptor
from .workspace import CandidateWorkspace


@dataclass(frozen=True)
class SearchPlan:
  axes: Mapping[str, tuple[int | float | str | bool, ...]]
  budgets: tuple[int, ...] = (3, 10, 30)
  reduction: int = 3
  max_candidates: int = 128
  timeout_seconds: int = 900

  @classmethod
  def from_mapping(cls, value: Mapping[str, Any]) -> SearchPlan:
    raw = dict(value)
    axes_value = raw.get("axes", {})
    if not isinstance(axes_value, Mapping) or not axes_value: raise ValueError("autotune search requires a non-empty axes table")
    axes: dict[str, tuple[int | float | str | bool, ...]] = {}
    for name, values in axes_value.items():
      if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)) or not values:
        raise ValueError(f"search axis {name!r} must be a non-empty array")
      converted = tuple(values)
      if not all(isinstance(item, (int, float, str, bool)) for item in converted):
        raise ValueError(f"search axis {name!r} contains an unsupported value")
      axes[str(name)] = converted
    budgets = tuple(int(x) for x in raw.get("budgets", (3, 10, 30)))
    if not budgets or any(x <= 0 for x in budgets): raise ValueError("autotune budgets must be positive")
    reduction = int(raw.get("reduction", 3))
    max_candidates = int(raw.get("max_candidates", 128))
    timeout_seconds = int(raw.get("timeout_seconds", 900))
    if reduction < 2 or max_candidates < 1 or timeout_seconds < 1: raise ValueError("invalid autotune search limits")
    plan = cls(axes, budgets, reduction, max_candidates, timeout_seconds)
    if plan.cardinality > max_candidates: raise ValueError(f"search has {plan.cardinality} candidates, maximum is {max_candidates}")
    return plan

  @property
  def cardinality(self) -> int:
    total = 1
    for values in self.axes.values(): total *= len(values)
    return total

  def parameters(self) -> list[dict[str, int | float | str | bool]]:
    names = tuple(self.axes)
    return [dict(zip(names, values)) for values in itertools.product(*(self.axes[name] for name in names))]


def _candidate_id(base_id: str, parameters: Mapping[str, Any]) -> str:
  digest = hashlib.sha256(json.dumps(parameters, sort_keys=True, default=str).encode()).hexdigest()[:12]
  return f"{base_id}-tune-{digest}"


def _contract(spec) -> WorkloadContract:
  acceptance = spec.metadata.get("recipe_acceptance", spec.metadata.get("acceptance", {}))
  if not isinstance(acceptance, Mapping): acceptance = {}
  objective_text = str(spec.metadata.get("search_objective", "p95_latency_us"))
  try: objective = Objective(objective_text)
  except ValueError: objective = Objective.P95_LATENCY_US
  return WorkloadContract(
    name=f"{spec.name}:{HookDescriptor.from_spec(spec).when.signature}", target=spec.target, objective=objective,
    max_abs_error=float(acceptance.get("max_abs_error", acceptance.get("maximum_error", 1e-3))),
    max_rel_error=float(acceptance.get("max_rel_error", 1e-3)),
    max_vram_bytes=int(acceptance["max_vram_bytes"]) if acceptance.get("max_vram_bytes") is not None else None,
    max_vgprs=int(acceptance["max_vgprs"]) if acceptance.get("max_vgprs") is not None else None,
    max_lds_bytes=int(acceptance["max_lds_bytes"]) if acceptance.get("max_lds_bytes") is not None else None,
    forbid_spills=bool(acceptance.get("forbid_spills", True)),
    metadata={"hook": asdict(HookDescriptor.from_spec(spec)), "objective_text": spec.objective},
  )


def run_autotune(workspace: CandidateWorkspace, candidate_id: str, project_root: str | Path,
                 plan: SearchPlan, ledger_path: str | Path | None = None) -> tuple[Any, TuningSummary]:
  """Tune one structural implementation without allowing speed to bypass correctness."""
  record = workspace.load_candidate(candidate_id)
  if record.status != "mockgpu_passed": raise ValueError("candidate must pass MockGPU before stage-specific W7900 autotuning")
  spec = workspace.load_spec(record.spec_id)
  if not spec.hardware_command: raise ValueError("spec has no hardware command for autotuning")
  root = Path(project_root).resolve()
  command = workspace.render_command(spec.hardware_command, spec, record, root)
  cwd = str(spec.metadata.get("recipe_bundle", root))
  environment = {"DEV": "AMD", "RADEON_FORGE_CANDIDATE": record.source_path,
                 "RADEON_FORGE_RECIPE_BUNDLE": str(spec.metadata.get("recipe_bundle", "")),
                 "RADEON_FORGE_EXECUTION_STAGES": ",".join(x.value for x in HookDescriptor.from_spec(spec).when.stages),
                 "PYTHONUNBUFFERED": "1"}
  harness = CommandHarness(command, cwd=cwd, env={**os.environ, **environment}, timeout_seconds=plan.timeout_seconds)
  candidates = [Candidate(_candidate_id(record.candidate_id, parameters), spec.name, parameters,
                          source_path=record.source_path, hypothesis=record.hypothesis, parent_id=record.candidate_id)
                for parameters in plan.parameters()]
  ledger = ExperimentLedger(ledger_path or (workspace.root / "autotune" / f"{record.candidate_id}.jsonl"))
  summary = successive_halving(candidates, harness, _contract(spec), plan.budgets, plan.reduction, ledger)
  evidence = {"search_plan": asdict(plan), "rounds": summary.rounds, "evaluated_trials": len(summary.evaluated),
              "ledger": str(ledger.path), "execution_hook": asdict(HookDescriptor.from_spec(spec)),
              "winner": summary.winner.to_dict() if summary.winner is not None else None,
              "rejections": [{"candidate_id": result.candidate.candidate_id, "parameters": dict(result.candidate.parameters),
                              "reason": result.rejected_reason or result.correctness.reason,
                              "feasible": result.is_feasible(_contract(spec))} for result in summary.evaluated]}
  if summary.winner is None:
    updated = workspace.update(record.candidate_id, "autotune_failed", {"autotune": evidence})
  else:
    updated = workspace.update(record.candidate_id, "hardware_passed", {"autotune": evidence,
      "selected_parameters": dict(summary.winner.candidate.parameters), "hardware": summary.winner.to_dict()})
  return updated, summary
