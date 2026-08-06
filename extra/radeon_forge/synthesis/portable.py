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
  filtered = {key: value for key, value in values.items() if value is not None and value != {}}
  if not filtered: return
  lines.append(f"\n[{name}]")
  for key, value in filtered.items():
    if isinstance(value, Mapping): continue
    lines.append(f"{json.dumps(str(key))} = {_toml_value(value)}")
  for key, value in filtered.items():
    if isinstance(value, Mapping): _append_table(lines, f"{name}.{key}", value)


def export_recipe_with_hook(workspace: CandidateWorkspace, spec_id: str, output: str | Path,
                            candidate_id: str | None = None) -> Path:
  """Export placement, execution-state applicability and empirical search knowledge."""
  path = export_recipe(workspace, spec_id, output, candidate_id)
  spec = workspace.load_spec(spec_id)
  descriptor = HookDescriptor.from_spec(spec)
  predicate = descriptor.when
  when = {
    "stages": [stage.value for stage in predicate.stages],
    "min_context_tokens": predicate.min_context_tokens,
    "max_context_tokens": predicate.max_context_tokens,
    "min_prompt_tokens": predicate.min_prompt_tokens,
    "max_prompt_tokens": predicate.max_prompt_tokens,
    "min_generated_token_index": predicate.min_generated_token_index,
    "max_generated_token_index": predicate.max_generated_token_index,
    "batch_sizes": list(predicate.batch_sizes),
    "prefix_cache": predicate.prefix_cache,
    "warm": predicate.warm,
    "conditions": dict(predicate.conditions),
  }
  hook = {
    "layer": descriptor.layer.value,
    "target": descriptor.target,
    "mode": descriptor.mode.value,
    "adapter": descriptor.adapter,
    "priority": descriptor.priority,
    "exclusive_group": descriptor.exclusive_group,
    "description": descriptor.description,
    "selector": dict(descriptor.selector),
    "when": when,
  }
  lines = path.read_text(encoding="utf-8").rstrip().splitlines()
  _append_table(lines, "metadata.hook", hook)

  # The search space is portable knowledge, not a mandatory implementation
  # abstraction. A receiving agent may reuse, shrink or replace it, but keeping
  # it next to the stage predicate makes prior hardware exploration reproducible.
  search = spec.metadata.get("search", {})
  if isinstance(search, Mapping) and search: _append_table(lines, "metadata.search", search)

  selected_parameters = {}
  if candidate_id is not None:
    candidate = workspace.load_candidate(candidate_id)
    selected_parameters = candidate.evidence.get("selected_parameters", {})
  if isinstance(selected_parameters, Mapping) and selected_parameters:
    _append_table(lines, "metadata.exported_winner", dict(selected_parameters))

  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path
