from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .contracts import ResourceReport


_PATTERNS: dict[str, tuple[str, ...]] = {
  "vgprs": (r"\bNumVgprs:\s*(\d+)", r"\.vgpr_count:\s*(\d+)", r"\bvgpr_count:\s*(\d+)"),
  "sgprs": (r"\bNumSgprs:\s*(\d+)", r"\.sgpr_count:\s*(\d+)", r"\bsgpr_count:\s*(\d+)"),
  "scratch_bytes": (r"\bScratchSize:\s*(\d+)", r"\.private_segment_fixed_size:\s*(\d+)", r"\bprivate_segment_fixed_size:\s*(\d+)"),
  "lds_bytes": (r"\bLDSByteSize:\s*(\d+)", r"\.group_segment_fixed_size:\s*(\d+)", r"\bgroup_segment_fixed_size:\s*(\d+)"),
  "occupancy": (r"\bOccupancy:\s*(\d+)",),
  "spilled_vgprs": (r"\.vgpr_spill_count:\s*(\d+)", r"\bvgpr_spill_count:\s*(\d+)"),
  "spilled_sgprs": (r"\.sgpr_spill_count:\s*(\d+)", r"\bsgpr_spill_count:\s*(\d+)"),
}


def _first_int(text: str, patterns: tuple[str, ...]) -> int | None:
  for pattern in patterns:
    if match := re.search(pattern, text, flags=re.IGNORECASE): return int(match.group(1))
  return None


def resource_report_from_text(text: str) -> ResourceReport:
  values = {name: _first_int(text, patterns) for name, patterns in _PATTERNS.items()}
  return ResourceReport(
    vgprs=values["vgprs"], sgprs=values["sgprs"], lds_bytes=values["lds_bytes"], scratch_bytes=values["scratch_bytes"],
    occupancy=values["occupancy"], spilled_vgprs=values["spilled_vgprs"] or 0, spilled_sgprs=values["spilled_sgprs"] or 0,
  )


def _as_int(value: Any) -> int | None:
  if value is None: return None
  try: return int(value)
  except (TypeError, ValueError): return None


def resource_report_from_comgr(metadata: Mapping[str, Any], kernel_index: int = 0) -> ResourceReport:
  """Parse COMGR executable metadata such as the structure exposed in tinygrad PR #3641."""
  kernels = metadata.get("amdhsa.kernels")
  if not isinstance(kernels, list) or kernel_index >= len(kernels) or not isinstance(kernels[kernel_index], Mapping):
    raise ValueError("COMGR metadata does not contain the requested amdhsa kernel")
  kernel = kernels[kernel_index]
  return ResourceReport(
    vgprs=_as_int(kernel.get(".vgpr_count", kernel.get("vgpr_count"))),
    sgprs=_as_int(kernel.get(".sgpr_count", kernel.get("sgpr_count"))),
    lds_bytes=_as_int(kernel.get(".group_segment_fixed_size", kernel.get("group_segment_fixed_size"))),
    scratch_bytes=_as_int(kernel.get(".private_segment_fixed_size", kernel.get("private_segment_fixed_size"))),
    spilled_vgprs=_as_int(kernel.get(".vgpr_spill_count", kernel.get("vgpr_spill_count"))) or 0,
    spilled_sgprs=_as_int(kernel.get(".sgpr_spill_count", kernel.get("sgpr_spill_count"))) or 0,
  )
