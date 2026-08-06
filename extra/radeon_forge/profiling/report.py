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


def _kernel_summary(events: list[TraceEvent]) -> tuple[list[dict[str, Any]], float, int, int]:
  grouped: dict[str, list[float]] = defaultdict(list)
  decode_count, prefill_count = 0, 0
  for event in events:
    if event.name != "kernel": continue
    duration = _number(event.attributes.get("duration_ms"))
    if duration <= 0: continue
    name = str(event.attributes.get("name") or event.attributes.get("kernel_name") or "unknown_kernel")
    grouped[name].append(duration)
    stage = str(event.attributes.get("stage", "unknown"))
    if stage == "decode": decode_count += 1
    elif stage == "prefill": prefill_count += 1
  total = sum(sum(values) for values in grouped.values())
  rows = [{"name": name, "calls": len(values), "total_ms": sum(values), "mean_ms": statistics.mean(values),
           "p95_ms": _percentile(values, 0.95), "share": sum(values) / max(total, 1e-12)} for name, values in grouped.items()]
  rows.sort(key=lambda row: (-row["total_ms"], row["name"]))
  return rows, total, decode_count, prefill_count


def build_profile_report(events: Iterable[TraceEvent]) -> dict[str, Any]:
  evs = list(events)
  durations = {kind: sum(x.duration_ms or 0.0 for x in evs if x.kind == kind) for kind in {x.kind for x in evs}}
  token_events = [x for x in evs if x.name == "token"]
  token_wall = [_number(x.attributes.get("wall_ms")) for x in token_events if _number(x.attributes.get("wall_ms")) > 0]
  token_gpu = [_number(x.attributes.get("gpu_ms")) for x in token_events if _number(x.attributes.get("gpu_ms")) > 0]
  kernels_per_token = [int(_number(x.attributes.get("kernel_count"))) for x in token_events]
  expected_decode_profile = sum(int(_number(x.attributes.get("profile_kernel_events"))) for x in token_events)
  prefill = [x for x in evs if x.name == "prefill"]
  prefill_wall = sum(_number(x.attributes.get("wall_ms")) for x in prefill)
  expected_prefill_profile = sum(int(_number(x.attributes.get("profile_kernel_events"))) for x in prefill)
  tools = [x for x in evs if x.kind == "tool" and x.duration_ms is not None and x.name != "tool_result"]
  metric_events = [x for x in evs if x.name == "metric"]
  truncated = [x for x in metric_events if x.attributes.get("name") == "profile_truncated"]
  top_kernels, profiled_kernel_ms, decode_profile_count, prefill_profile_count = _kernel_summary(evs)
  findings: list[Finding] = []

  if kernels_per_token and statistics.mean(kernels_per_token) >= 20:
    findings.append(Finding("high", "observed", "High kernel-launch count per decoded token",
      (f"mean launches/token={statistics.mean(kernels_per_token):.1f}", f"profiled decode ranges={decode_profile_count}"),
      "Inspect the dominant launch sequence and test a fused subgraph or persistent megakernel; do not assume arithmetic is the bottleneck."))
  if token_wall and token_gpu and sum(token_gpu) / max(sum(token_wall), 1e-9) < 0.65:
    findings.append(Finding("high", "inferred", "GPU execution explains only part of decode wall time",
      (f"GPU/wall ratio={sum(token_gpu)/sum(token_wall):.2f}",),
      "Profile CPU submission, synchronization, sampling and launch bubbles before optimizing arithmetic throughput."))
  if prefill and prefill_wall > sum(token_wall):
    findings.append(Finding("medium", "observed", "Prefill dominates this agent turn",
      (f"prefill_ms={prefill_wall:.2f}", f"decode_ms={sum(token_wall):.2f}"),
      "Increase stable-prefix reuse and compact repeated tool schemas or repository context before generating lower-level kernels."))
  if tools and sum(x.duration_ms or 0 for x in tools) > sum(token_wall):
    findings.append(Finding("medium", "observed", "Tool execution dominates model decode",
      (f"tool_ms={sum(x.duration_ms or 0 for x in tools):.2f}",),
      "Parallelize independent tools or reduce tool round trips; a faster inference kernel alone will not materially improve task latency."))

  if top_kernels:
    dominant = top_kernels[0]
    if dominant["share"] >= 0.25:
      findings.append(Finding("high", "observed", "One kernel family dominates captured GPU time",
        (f"kernel={dominant['name']}", f"share={dominant['share']*100:.1f}%", f"calls={dominant['calls']}",
         f"total_ms={dominant['total_ms']:.2f}"),
        "Map this kernel back to the model subgraph, inspect its shapes and resource metadata, then generate a bounded structural alternative and retune it."))
    tiny = [row for row in top_kernels if row["mean_ms"] < 0.05]
    tiny_calls = sum(row["calls"] for row in tiny)
    if decode_profile_count and tiny_calls / max(decode_profile_count + prefill_profile_count, 1) >= 0.35:
      findings.append(Finding("medium", "inferred", "Captured execution is fragmented across many very small kernels",
        (f"sub-50us calls={tiny_calls}", f"captured kernel calls={decode_profile_count+prefill_profile_count}"),
        "Inspect adjacency and data materialization boundaries. Fusion is promising only where the independent numerical oracle covers the combined subgraph."))
  else:
    findings.append(Finding("info", "unknown", "No kernel-level trace is attached",
      ("Only agent/model aggregate counters are available.",),
      "Enable the tinygrad PROFILE event adapter or capture ROCm/SQTT evidence before making a causal hardware diagnosis."))

  expected_profile = expected_decode_profile + expected_prefill_profile
  captured_profile = decode_profile_count + prefill_profile_count
  if truncated:
    findings.append(Finding("medium", "observed", "Kernel trace was truncated",
      tuple(f"{x.attributes.get('stage')} captured={x.attributes.get('captured')} available={x.attributes.get('available')}" for x in truncated[:4]),
      "Reduce the profiled workload or raise the capture budget before using kernel shares as complete attribution."))
  elif expected_profile and captured_profile < expected_profile:
    findings.append(Finding("info", "unknown", "Some backend profile ranges were not attached to the unified trace",
      (f"backend reported={expected_profile}", f"trace captured={captured_profile}"),
      "Treat per-kernel attribution as partial until the transport discrepancy is resolved."))

  return {
    "summary": {
      "event_count": len(evs), "durations_ms_by_kind": durations, "decode_tokens": len(token_events),
      "token_wall_ms_p50": statistics.median(token_wall) if token_wall else None,
      "token_wall_ms_p95": _percentile(token_wall, 0.95),
      "token_gpu_ms_p50": statistics.median(token_gpu) if token_gpu else None,
      "mean_kernel_count_per_token": statistics.mean(kernels_per_token) if kernels_per_token else None,
      "tool_time_ms": sum(x.duration_ms or 0 for x in tools), "prefill_wall_ms": prefill_wall,
      "profiled_kernel_time_ms": profiled_kernel_ms, "profiled_decode_kernel_calls": decode_profile_count,
      "profiled_prefill_kernel_calls": prefill_profile_count, "kernel_profile_complete": bool(captured_profile and not truncated and captured_profile >= expected_profile),
      "top_kernels": top_kernels[:20],
    },
    "findings": [asdict(x) for x in findings],
  }
