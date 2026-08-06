from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .hooks import HookDescriptor
from .recipe import export_recipe
from .workspace import CandidateWorkspace


def _toml_value(value: Any) -> str:
  if isinstance(value, bool): return "true" if value else "false"
  if isinstance(value, (int, float)): return str(value)
  if isinstance(value, str): return json.dumps(value, ensure_ascii=False)
  if isinstance(value, (list, tuple)): return "[" + ", ".join(_toml_value(x) for x in value) + "]"
  raise TypeError(f"unsupported portable hook value: {type(value).__name__}")


def _append_table(lines: list[str], name: str, values: Mapping[str, Any]) -> None:
  if not values: return
  lines.append(f"\n[{name}]")
  for key, value in values.items():
    if isinstance(value, Mapping): continue
    lines.append(f"{json.dumps(str(key))} = {_toml_value(value)}")
  for key, value in values.items():
    if isinstance(value, Mapping): _append_table(lines, f"{name}.{key}", value)


def export_recipe_with_hook(workspace: CandidateWorkspace, spec_id: str, output: str | Path,
                            candidate_id: str | None = None) -> Path:
  """Export one recipe while preserving its concrete runtime interception point."""
  path = export_recipe(workspace, spec_id, output, candidate_id)
  spec = workspace.load_spec(spec_id)
  descriptor = HookDescriptor.from_spec(spec)
  hook = {
    "layer": descriptor.layer.value,
    "target": descriptor.target,
    "mode": descriptor.mode.value,
    "adapter": descriptor.adapter,
    "priority": descriptor.priority,
    "exclusive_group": descriptor.exclusive_group,
    "description": descriptor.description,
    "selector": dict(descriptor.selector),
  }
  lines = path.read_text(encoding="utf-8").rstrip().splitlines()
  _append_table(lines, "metadata.hook", hook)
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path
