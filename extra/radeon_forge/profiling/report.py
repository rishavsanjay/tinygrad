from __future__ import annotations

import math, statistics
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


def build_profile_report(events: Iterable[TraceEvent]) -> dict[str, Any]:
  evs = list(events)
  durations = {kind: sum(x.duration_ms or 0.0 for x in evs if x.kind == kind) for kind in {x.kind for x in evs}}
  token_events = [x for x in evs if x.name == "token"]
  token_wall = [float(x.attributes.get("wall_ms", 0.0)) for x in token_events if float(x.attributes.get("wall_ms", 0.0)) > 0]
  token_gpu = [float(x.attributes.get("gpu_ms", 0.0)) for x in token_events if float(x.attributes.get("gpu_ms", 0.0)) > 0]
  kernels = [int(x.attributes.get("kernel_count", 0)) for x in token_events]
  prefill = [x for x in evs if x.name == "prefill"]
  tools = [x for x in evs if x.kind == "tool" and x.duration_ms is not None]
  findings: list[Finding] = []

  if kernels and statistics.mean(kernels) >= 20:
    findings.append(Finding("high", "observed", "High kernel-launch count per decoded token",
      (f"mean launches/token={statistics.mean(kernels):.1f}",),
      "Inspect launch gaps and fuse the dominant decode subgraph or generate a persistent megakernel candidate."))
  if token_wall and token_gpu and sum(token_gpu) / max(sum(token_wall), 1e-9) < 0.65:
    findings.append(Finding("high", "inferred", "GPU execution explains only part of decode wall time",
      (f"GPU/wall ratio={sum(token_gpu)/sum(token_wall):.2f}",),
      "Profile CPU submission, synchronization and launch bubbles before optimizing arithmetic throughput."))
  if prefill and sum(x.duration_ms or 0 for x in prefill) > sum(token_wall):
    findings.append(Finding("medium", "observed", "Prefill dominates this agent turn",
      (f"prefill_ms={sum(x.duration_ms or 0 for x in prefill):.2f}", f"decode_ms={sum(token_wall):.2f}"),
      "Increase stable-prefix reuse and compact repeated tool schemas/repository context."))
  if tools and sum(x.duration_ms or 0 for x in tools) > sum(token_wall):
    findings.append(Finding("medium", "observed", "Tool execution dominates model decode",
      (f"tool_ms={sum(x.duration_ms or 0 for x in tools):.2f}",),
      "Parallelize independent tools or reduce tool round trips; a faster kernel will not materially improve task latency."))
  if not any(x.name == "kernel" for x in evs):
    findings.append(Finding("info", "unknown", "No kernel-level trace is attached",
      ("Only agent/model aggregate counters are available.",),
      "Capture tinygrad PROFILE events or ROCm/SQTT evidence before making a causal hardware diagnosis."))

  return {
    "summary": {
      "event_count": len(evs), "durations_ms_by_kind": durations, "decode_tokens": len(token_events),
      "token_wall_ms_p50": statistics.median(token_wall) if token_wall else None,
      "token_wall_ms_p95": _percentile(token_wall, 0.95),
      "token_gpu_ms_p50": statistics.median(token_gpu) if token_gpu else None,
      "mean_kernel_count_per_token": statistics.mean(kernels) if kernels else None,
      "tool_time_ms": sum(x.duration_ms or 0 for x in tools),
    },
    "findings": [asdict(x) for x in findings],
  }
