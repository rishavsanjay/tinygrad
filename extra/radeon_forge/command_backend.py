from __future__ import annotations

import json
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import Candidate, CorrectnessReport, ResourceReport, TrialResult


def _percentile(values: Sequence[float], q: float) -> float | None:
  if not values: return None
  ordered = sorted(values)
  index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q + 0.999999)))
  return ordered[index]


@dataclass(frozen=True)
class CommandHarness:
  """Evaluate candidates through a workload-specific executable.

  The executable receives the candidate and budget through environment
  variables and must print one JSON object on its final non-empty stdout line.
  This keeps Forge independent of any one kernel launch abstraction while the
  reference implementation and benchmark protocol remain explicit.
  """

  command: Sequence[str]
  cwd: str | Path | None = None
  env: Mapping[str, str] | None = None
  timeout_seconds: int = 300

  def __call__(self, candidate: Candidate, budget: int) -> TrialResult:
    run_env = dict(os.environ)
    if self.env is not None: run_env.update(self.env)
    run_env["RADEON_FORGE_CANDIDATE_JSON"] = json.dumps(candidate.to_dict(), sort_keys=True)
    run_env["RADEON_FORGE_BUDGET"] = str(budget)
    proc = subprocess.run(list(self.command), cwd=self.cwd, env=run_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=self.timeout_seconds, check=False)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
      reason = f"command failed with exit {proc.returncode}: {proc.stderr[-2000:]}"
      return TrialResult(candidate, CorrectnessReport(False, reason=reason), compile_ok=False, stable=False,
                         rejected_reason="command_failed", evidence={"stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]})
    try:
      payload: dict[str, Any] = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
      return TrialResult(candidate, CorrectnessReport(False, reason=f"invalid JSON result: {exc}"), compile_ok=False, stable=False,
                         rejected_reason="invalid_result", evidence={"stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:]})

    correctness_data = payload.get("correctness", {})
    resource_data = payload.get("resources", {})
    samples = tuple(float(value) for value in payload.get("samples_us", ()))
    correctness = CorrectnessReport(
      passed=bool(correctness_data.get("passed", False)),
      max_abs_error=float(correctness_data.get("max_abs_error", 0.0)),
      max_rel_error=float(correctness_data.get("max_rel_error", 0.0)),
      checked_values=int(correctness_data.get("checked_values", 0)),
      reason=str(correctness_data.get("reason", "")),
    )
    resources = ResourceReport(
      vgprs=resource_data.get("vgprs"), sgprs=resource_data.get("sgprs"), lds_bytes=resource_data.get("lds_bytes"),
      scratch_bytes=resource_data.get("scratch_bytes"), occupancy=resource_data.get("occupancy"),
      spilled_vgprs=int(resource_data.get("spilled_vgprs", 0)), spilled_sgprs=int(resource_data.get("spilled_sgprs", 0)),
    )
    return TrialResult(
      candidate=candidate,
      correctness=correctness,
      samples_us=samples,
      median_latency_us=float(payload["median_latency_us"]) if "median_latency_us" in payload else (statistics.median(samples) if samples else None),
      p95_latency_us=float(payload["p95_latency_us"]) if "p95_latency_us" in payload else _percentile(samples, 0.95),
      end_to_end_p95_ms=float(payload["end_to_end_p95_ms"]) if "end_to_end_p95_ms" in payload else None,
      resources=resources,
      compile_ok=bool(payload.get("compile_ok", True)),
      stable=bool(payload.get("stable", True)),
      rejected_reason=str(payload.get("rejected_reason", "")),
      evidence={"stdout": proc.stdout, "stderr": proc.stderr, **payload.get("evidence", {})},
    )
