from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class Objective(str, Enum):
  MEDIAN_LATENCY_US = "median_latency_us"
  P95_LATENCY_US = "p95_latency_us"
  END_TO_END_P95_MS = "end_to_end_p95_ms"


@dataclass(frozen=True)
class WorkloadContract:
  name: str
  target: str = "gfx1100"
  objective: Objective = Objective.P95_LATENCY_US
  max_abs_error: float = 1e-3
  max_rel_error: float = 1e-3
  max_vram_bytes: int | None = None
  max_vgprs: int | None = None
  max_lds_bytes: int | None = None
  forbid_spills: bool = True
  metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
  candidate_id: str
  family: str
  parameters: Mapping[str, int | float | str | bool]
  source_path: str | None = None
  hypothesis: str = ""
  parent_id: str | None = None

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class ResourceReport:
  vgprs: int | None = None
  sgprs: int | None = None
  lds_bytes: int | None = None
  scratch_bytes: int | None = None
  occupancy: int | None = None
  spilled_vgprs: int = 0
  spilled_sgprs: int = 0

  @property
  def has_spills(self) -> bool:
    return self.spilled_vgprs > 0 or self.spilled_sgprs > 0


@dataclass(frozen=True)
class CorrectnessReport:
  passed: bool
  max_abs_error: float = 0.0
  max_rel_error: float = 0.0
  checked_values: int = 0
  reason: str = ""


@dataclass(frozen=True)
class TrialResult:
  candidate: Candidate
  correctness: CorrectnessReport
  samples_us: tuple[float, ...] = ()
  median_latency_us: float | None = None
  p95_latency_us: float | None = None
  end_to_end_p95_ms: float | None = None
  resources: ResourceReport = field(default_factory=ResourceReport)
  compile_ok: bool = True
  stable: bool = True
  rejected_reason: str = ""
  evidence: Mapping[str, Any] = field(default_factory=dict)

  def metric(self, objective: Objective) -> float:
    value = {
      Objective.MEDIAN_LATENCY_US: self.median_latency_us,
      Objective.P95_LATENCY_US: self.p95_latency_us,
      Objective.END_TO_END_P95_MS: self.end_to_end_p95_ms,
    }[objective]
    return float("inf") if value is None else value

  def is_feasible(self, contract: WorkloadContract) -> bool:
    if not self.compile_ok or not self.stable or not self.correctness.passed: return False
    if contract.forbid_spills and self.resources.has_spills: return False
    if contract.max_vgprs is not None and self.resources.vgprs is not None and self.resources.vgprs > contract.max_vgprs: return False
    if contract.max_lds_bytes is not None and self.resources.lds_bytes is not None and self.resources.lds_bytes > contract.max_lds_bytes: return False
    if self.correctness.max_abs_error > contract.max_abs_error: return False
    if self.correctness.max_rel_error > contract.max_rel_error: return False
    return self.metric(contract.objective) != float("inf")

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)
