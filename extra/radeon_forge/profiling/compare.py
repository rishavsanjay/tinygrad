from __future__ import annotations

import csv, json, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class NumericDelta:
  baseline: float
  candidate: float
  absolute: float
  relative: float | None


@dataclass(frozen=True)
class KernelDelta:
  kernel: str
  metric: str
  baseline_count: int
  candidate_count: int
  baseline_mean: float
  candidate_mean: float
  absolute: float
  relative: float | None


class IncomparableCaptures(ValueError): pass


def _manifest(value: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path | None]:
  if isinstance(value, Mapping): return dict(value), None
  path = Path(value)
  if path.is_dir(): path = path / "forge_capture_manifest.json"
  if not path.is_file(): raise FileNotFoundError(path)
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict): raise ValueError(f"capture manifest {path} is not an object")
  return payload, path.parent


def _normalized_command(value: Any) -> tuple[str, ...]:
  if not isinstance(value, list): return ()
  return tuple(str(x) for x in value)


def _counter_set(manifest: Mapping[str, Any]) -> tuple[str, ...]:
  summary = manifest.get("summary", {})
  if not isinstance(summary, Mapping): return ()
  counters = summary.get("counters", ())
  return tuple(sorted(str(x) for x in counters)) if isinstance(counters, list) else ()


def _workload_identity(manifest: Mapping[str, Any]) -> str:
  summary = manifest.get("summary", {})
  metadata = summary.get("capture_metadata", {}) if isinstance(summary, Mapping) else {}
  if isinstance(metadata, Mapping):
    for key in ("workload_hash", "task_suite_hash", "input_hash", "workload_id"):
      if metadata.get(key): return f"{key}:{metadata[key]}"
  command = _normalized_command(manifest.get("target_command"))
  return "command:" + json.dumps(command)


def compatibility_report(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
  reasons: list[str] = []
  base_summary = baseline.get("summary", {}) if isinstance(baseline.get("summary", {}), Mapping) else {}
  cand_summary = candidate.get("summary", {}) if isinstance(candidate.get("summary", {}), Mapping) else {}
  checks = {
    "kind": (baseline.get("kind"), candidate.get("kind")),
    "stage": (base_summary.get("stage"), cand_summary.get("stage")),
    "target_command": (_normalized_command(baseline.get("target_command")), _normalized_command(candidate.get("target_command"))),
    "workload_identity": (_workload_identity(baseline), _workload_identity(candidate)),
  }
  if baseline.get("kind") == "counters" or candidate.get("kind") == "counters":
    checks["counters"] = (_counter_set(baseline), _counter_set(candidate))
  for name, (left, right) in checks.items():
    if left != right: reasons.append(f"{name} differs: baseline={left!r} candidate={right!r}")
  if not baseline.get("passed"): reasons.append("baseline capture did not pass")
  if not candidate.get("passed"): reasons.append("candidate capture did not pass")
  return {"comparable": not reasons, "reasons": reasons, "checks": checks}


def _relative(base: float, candidate: float) -> float | None:
  if base == 0: return None
  return (candidate - base) / base


def _numeric_summary(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
  summary = manifest.get("summary", {})
  parsed = summary.get("parsed_csv", {}) if isinstance(summary, Mapping) else {}
  numeric = parsed.get("numeric_columns", {}) if isinstance(parsed, Mapping) else {}
  return numeric if isinstance(numeric, Mapping) else {}


def _numeric_deltas(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
  left, right = _numeric_summary(baseline), _numeric_summary(candidate)
  ret: dict[str, dict[str, Any]] = {}
  for key in sorted(set(left) & set(right)):
    if not isinstance(left[key], Mapping) or not isinstance(right[key], Mapping): continue
    if "mean" not in left[key] or "mean" not in right[key]: continue
    base, cand = float(left[key]["mean"]), float(right[key]["mean"])
    ret[str(key)] = asdict(NumericDelta(base, cand, cand-base, _relative(base, cand)))
  return ret


def _artifact_paths(manifest: Mapping[str, Any], root: Path | None, suffix: str) -> list[Path]:
  paths = []
  for item in manifest.get("artifacts", []):
    if not isinstance(item, Mapping) or str(item.get("suffix", "")).lower() != suffix: continue
    path = Path(str(item.get("path", "")))
    if not path.is_absolute() and root is not None: path = root / path
    if path.is_file(): paths.append(path)
  return paths


def _kernel_column(fieldnames: Iterable[str]) -> str | None:
  names = list(fieldnames)
  preferred = ("Kernel_Name", "KernelName", "Kernel", "Name", "kernel_name", "kernel")
  return next((name for name in preferred if name in names), None)


def _kernel_metrics(manifest: Mapping[str, Any], root: Path | None) -> dict[tuple[str, str], list[float]]:
  grouped: dict[tuple[str, str], list[float]] = {}
  for path in _artifact_paths(manifest, root, ".csv"):
    try:
      with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        key_column = _kernel_column(reader.fieldnames or ())
        if key_column is None: continue
        for row in reader:
          kernel = str(row.get(key_column, "")).strip()
          if not kernel: continue
          for key, value in row.items():
            if key == key_column: continue
            try: number = float(value)
            except (TypeError, ValueError): continue
            if math.isfinite(number): grouped.setdefault((kernel, str(key)), []).append(number)
    except OSError: continue
  return grouped


def _kernel_deltas(baseline: Mapping[str, Any], base_root: Path | None,
                   candidate: Mapping[str, Any], cand_root: Path | None) -> list[dict[str, Any]]:
  left, right = _kernel_metrics(baseline, base_root), _kernel_metrics(candidate, cand_root)
  ret = []
  for kernel, metric in sorted(set(left) & set(right)):
    base_values, cand_values = left[(kernel, metric)], right[(kernel, metric)]
    base_mean, cand_mean = sum(base_values)/len(base_values), sum(cand_values)/len(cand_values)
    ret.append(asdict(KernelDelta(kernel, metric, len(base_values), len(cand_values), base_mean, cand_mean,
                                  cand_mean-base_mean, _relative(base_mean, cand_mean))))
  return ret


def compare_captures(baseline_value: str | Path | Mapping[str, Any],
                     candidate_value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
  baseline, base_root = _manifest(baseline_value)
  candidate, cand_root = _manifest(candidate_value)
  compatibility = compatibility_report(baseline, candidate)
  if not compatibility["comparable"]: raise IncomparableCaptures("; ".join(compatibility["reasons"]))
  return {
    "baseline_capture_id": baseline.get("capture_id"),
    "candidate_capture_id": candidate.get("capture_id"),
    "kind": baseline.get("kind"),
    "stage": (baseline.get("summary", {}) or {}).get("stage"),
    "compatibility": compatibility,
    "numeric_column_deltas": _numeric_deltas(baseline, candidate),
    "per_kernel_deltas": _kernel_deltas(baseline, base_root, candidate, cand_root),
    "interpretation_contract": {
      "lower_is_better_metrics": [],
      "higher_is_better_metrics": [],
      "note": "Deltas are descriptive only. Metric direction and causal meaning must come from the profiler schema or domain knowledge, not guessed from column names."
    }
  }
