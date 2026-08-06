from __future__ import annotations

import math, statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from ..runtime.events import TraceEvent


@dataclass(frozen=True)
class Finding:
  severity: str
  status: str
  title: str
  evidence: tuple[str, ...]
  recommendation: str


def _percentile(values: list[float], percentile: float) -> float | None:
  if not values: return None
  ordered = sorted(values)
  idx = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
  return ordered[idx]


def _number(value: Any, default: float = 0.0) -> float:
  try: return float(value)
  except (TypeError, ValueError): return default


def _rows(grouped: dict[str, list[float]]) -> tuple[list[dict[str, Any]], float]:
  total = sum(sum(values) for values in grouped.values())
  rows = [{"name": name, "calls": len(values), "total_ms": sum(values), "mean_ms": statistics.mean(values),
           "p95_ms": _percentile(values, 0.95), "share": sum(values) / max(total, 1e-12)} for name, values in grouped.items()]
  rows.sort(key=lambda row: (-row["total_ms"], row["name"]))
  return rows, total


def _kernel_summary(events: list[TraceEvent]) -> tuple[list[dict[str, Any]], float, dict[str, int], dict[str, list[dict[str, Any]]]]:
  grouped: dict[str, list[float]] = defaultdict(list)
  by_stage: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
  stage_counts: dict[str, int] = defaultdict(int)
  for event in events:
    if event.kind != "kernel": continue
    duration = _number(event.attributes.get("duration_ms"), event.duration_ms or 0.0)
    if duration <= 0: continue
    name = str(event.name or event.attributes.get("name") or event.attributes.get("kernel_name") or "unknown_kernel")
    stage = str(event.attributes.get("stage", "unknown"))
    grouped[name].append(duration)
    by_stage[stage][name].append(duration)
    stage_counts[stage] += 1
  rows, total = _rows(grouped)
  stage_rows = {stage: _rows(values)[0] for stage, values in by_stage.items()}
  return rows, total, dict(stage_counts), stage_rows


def build_profile_report(events: Iterable[TraceEvent]) -> dict[str, Any]:
  evs = list(events)
  durations = {kind: sum(x.duration_ms or 0.0 for x in evs if x.kind == kind) for kind in {x.kind for x in evs}}
  token_events = [x for x in evs if x.name == "token"]
  token_wall = [_number(x.attributes.get("wall_ms")) for x in token_events if _number(x.attributes.get("wall_ms")) > 0]
  token_gpu = [_number(x.attributes.get("gpu_ms")) for x in token_events if _number(x.attributes.get("gpu_ms")) > 0]
  kernels_per_token = [int(_number(x.attributes.get("kernel_count"))) for x in token_events]
  token_wall_by_stage: dict[str, list[float]] = defaultdict(list)
  for event in token_events:
    wall = _number(event.attributes.get("wall_ms"))
    if wall > 0: token_wall_by_stage[str(event.attributes.get("stage", "decode"))].append(wall)
  expected_decode_profile = sum(int(_number(x.attributes.get("profile_kernel_events"))) for x in token_events)
  prefill = [x for x in evs if x.name == "prefill"]
  prefill_wall = sum(_number(x.attributes.get("wall_ms")) for x in prefill)
  expected_prefill_profile = sum(int(_number(x.attributes.get("profile_kernel_events"))) for x in prefill)
  tools = [x for x in evs if x.kind == "tool" and x.duration_ms is not None and x.name != "tool_result"]
  metric_events = [x for x in evs if x.name == "metric"]
  truncated = [x for x in metric_events if x.attributes.get("name") == "profile_truncated"]
  hook_events = [x for x in evs if x.name == "hook"]
  hook_rollbacks = [x for x in hook_events if x.attributes.get("rolled_back")]
  top_kernels, profiled_kernel_ms, stage_kernel_counts, top_kernels_by_stage = _kernel_summary(evs)
  captured_profile = sum(stage_kernel_counts.values())
  findings: list[Finding] = []

  if kernels_per_token and statistics.mean(kernels_per_token) >= 20:
    findings.append(Finding("high", "observed", "High kernel-launch count per decoded token",
      (f"mean launches/token={statistics.mean(kernels_per_token):.1f}", f"profiled token-stage ranges={sum(v for k,v in stage_kernel_counts.items() if k != 'prefill')}"),
      "Inspect the dominant launch sequence separately for first-token and steady decode, then test a fused subgraph or persistent megakernel."))
  if token_wall and token_gpu and sum(token_gpu) / max(sum(token_wall), 1e-9) < 0.65:
    findings.append(Finding("high", "inferred", "GPU execution explains only part of decode wall time",
      (f"GPU/wall ratio={sum(token_gpu)/sum(token_wall):.2f}",),
      "Profile CPU submission, synchronization, sampling and launch bubbles before optimizing arithmetic throughput."))
  if prefill and prefill_wall > sum(token_wall):
    findings.append(Finding("medium", "observed", "Prefill dominates this agent turn",
      (f"prefill_ms={prefill_wall:.2f}", f"token_generation_ms={sum(token_wall):.2f}"),
      "Use a prefill-specific recipe: increase stable-prefix reuse, compact repeated tool schemas, or tune prefill kernels independently of decode."))
  if tools and sum(x.duration_ms or 0 for x in tools) > sum(token_wall):
    findings.append(Finding("medium", "observed", "Tool execution dominates model decode",
      (f"tool_ms={sum(x.duration_ms or 0 for x in tools):.2f}",),
      "Use an agent-loop or tool-execution-stage optimization; a faster model kernel alone will not materially improve task latency."))

  first_token = token_wall_by_stage.get("first_token", [])
  steady_decode = token_wall_by_stage.get("decode", [])
  if first_token and steady_decode and statistics.median(first_token) > statistics.median(steady_decode) * 1.4:
    findings.append(Finding("medium", "observed", "First-token and steady-decode latency are materially different",
      (f"first_token_p50={statistics.median(first_token):.2f}ms", f"steady_decode_p50={statistics.median(steady_decode):.2f}ms"),
      "Do not force one kernel schedule across both states. Keep separate first-token and steady-decode hook predicates and validate each independently."))

  stage_dominants = {stage: rows[0] for stage, rows in top_kernels_by_stage.items() if rows}
  distinct_dominants = {row["name"] for row in stage_dominants.values()}
  if len(distinct_dominants) > 1:
    findings.append(Finding("high", "observed", "Different execution stages have different dominant kernels",
      tuple(f"{stage}: {row['name']} ({row['share']*100:.1f}% of captured {stage} GPU time)" for stage, row in sorted(stage_dominants.items())),
      "Create separate stage-scoped optimization recipes rather than a single global replacement. Preserve a shared oracle but tune each state independently."))

  if top_kernels:
    dominant = top_kernels[0]
    if dominant["share"] >= 0.25:
      findings.append(Finding("high", "observed", "One kernel family dominates captured GPU time overall",
        (f"kernel={dominant['name']}", f"share={dominant['share']*100:.1f}%", f"calls={dominant['calls']}",
         f"total_ms={dominant['total_ms']:.2f}"),
        "Map this kernel to both its model subgraph and execution stage, then generate a bounded structural alternative and retune it."))
    tiny = [row for row in top_kernels if row["mean_ms"] < 0.05]
    tiny_calls = sum(row["calls"] for row in tiny)
    if captured_profile and tiny_calls / max(captured_profile, 1) >= 0.35:
      findings.append(Finding("medium", "inferred", "Captured execution is fragmented across many very small kernels",
        (f"sub-50us calls={tiny_calls}", f"captured kernel calls={captured_profile}"),
        "Inspect adjacency and materialization boundaries within the same execution stage. Fusion is promising only where the numerical oracle covers the combined subgraph."))
  else:
    findings.append(Finding("info", "unknown", "No kernel-level trace is attached",
      ("Only agent/model aggregate counters are available.",),
      "Enable the tinygrad PROFILE event adapter or capture ROCm/SQTT evidence before making a causal hardware diagnosis."))

  if hook_rollbacks:
    findings.append(Finding("high", "observed", "An execution-stage optimization was rolled back",
      tuple(f"stage={x.attributes.get('stage')} error={x.attributes.get('error')}" for x in hook_rollbacks[-4:]),
      "Keep the trusted baseline active, mark the candidate failed for this execution state, and regenerate from the preserved intent and oracle."))

  expected_profile = expected_decode_profile + expected_prefill_profile
  if truncated:
    findings.append(Finding("medium", "observed", "Kernel trace was truncated",
      tuple(f"{x.attributes.get('stage')} captured={x.attributes.get('captured')} available={x.attributes.get('available')}" for x in truncated[:4]),
      "Reduce the profiled workload or raise the capture budget before using kernel shares as complete attribution."))
  elif expected_profile and captured_profile < expected_profile:
    findings.append(Finding("info", "unknown", "Some backend profile ranges were not attached to the unified trace",
      (f"backend reported={expected_profile}", f"trace captured={captured_profile}"),
      "Treat per-kernel attribution as partial until the transport discrepancy is resolved."))

  stage_latency = {stage: {"count": len(values), "p50_ms": statistics.median(values) if values else None,
                           "p95_ms": _percentile(values, 0.95)} for stage, values in token_wall_by_stage.items()}
  return {
    "summary": {
      "event_count": len(evs), "durations_ms_by_kind": durations, "decode_tokens": len(token_events),
      "token_wall_ms_p50": statistics.median(token_wall) if token_wall else None,
      "token_wall_ms_p95": _percentile(token_wall, 0.95),
      "token_gpu_ms_p50": statistics.median(token_gpu) if token_gpu else None,
      "token_latency_by_stage": stage_latency,
      "mean_kernel_count_per_token": statistics.mean(kernels_per_token) if kernels_per_token else None,
      "tool_time_ms": sum(x.duration_ms or 0 for x in tools), "prefill_wall_ms": prefill_wall,
      "profiled_kernel_time_ms": profiled_kernel_ms, "profiled_kernel_calls_by_stage": stage_kernel_counts,
      "kernel_profile_complete": bool(captured_profile and not truncated and captured_profile >= expected_profile),
      "top_kernels": top_kernels[:20], "top_kernels_by_stage": {key: value[:10] for key, value in top_kernels_by_stage.items()},
      "hook_transitions": len(hook_events), "hook_rollbacks": len(hook_rollbacks),
    },
    "findings": [asdict(x) for x in findings],
  }
