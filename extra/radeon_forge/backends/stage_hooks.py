from __future__ import annotations

import hashlib, importlib.util, json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tinygrad import TinyJit

from ..synthesis.hooks import ExecutionContext, StagePredicate


class StageHookError(RuntimeError): pass


def _matching_hooks(hooks: Sequence[Mapping[str, Any]], context: ExecutionContext) -> list[Mapping[str, Any]]:
  """Resolve the highest-priority matching hook in each exclusive group."""
  selected: dict[str, Mapping[str, Any]] = {}
  for hook in hooks:
    descriptor = hook.get("descriptor", {})
    if not isinstance(descriptor, Mapping): continue
    try: predicate = StagePredicate.from_mapping(descriptor.get("when", {}))
    except Exception: continue
    if not predicate.matches(context): continue
    group = str(descriptor.get("exclusive_group", f"{descriptor.get('layer')}:{descriptor.get('target')}"))
    previous = selected.get(group)
    if previous is None or int(descriptor.get("priority", 0)) > int(previous.get("descriptor", {}).get("priority", 0)):
      selected[group] = hook
  return sorted(selected.values(), key=lambda x: -int(x.get("descriptor", {}).get("priority", 0)))


def _signature(hook: Mapping[str, Any]) -> str:
  parameters = hook.get("parameters", {})
  digest = hashlib.sha256(json.dumps(parameters, sort_keys=True, default=str).encode()).hexdigest()[:10]
  return f"{hook.get('activation_id')}:{digest}"


class ModelStageHookRuntime:
  """Apply validated Python model hooks only in their declared execution states.

  Candidate code runs inside the already-isolated local model subprocess. The
  original model objects are retained and restored on every transition or
  failure. This adapter never treats activation metadata as validation evidence.
  """
  def __init__(self, model: Any):
    self.model = model
    self._original_layers = tuple(getattr(model, "layers", ()))
    self._active_signature: tuple[str, ...] = ()
    self._loaded_modules: dict[tuple[str, str], Any] = {}

  def _reset(self) -> None:
    if self._original_layers and hasattr(self.model, "layers"): self.model.layers = list(self._original_layers)
    if hasattr(self.model, "forward_jit") and self.model.forward_jit is not None: self.model.forward_jit = TinyJit(self.model.forward)
    self._active_signature = ()

  @staticmethod
  def _indices(selector: Mapping[str, Any], count: int) -> list[int]:
    value = selector.get("indices", "all")
    if value == "all": return list(range(count))
    if isinstance(value, int): values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)): values = [int(x) for x in value]
    else: raise StageHookError("transformer block selector indices must be 'all', an integer, or an array")
    if any(index < 0 or index >= count for index in values): raise StageHookError("transformer block selector is out of range")
    return values

  def _module(self, hook: Mapping[str, Any]) -> Any:
    path = Path(str(hook["source_path"]))
    expected = str(hook["source_sha256"])
    if not path.is_file(): raise StageHookError(f"hook source is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected: raise StageHookError("hook source changed after activation")
    key = (str(path), expected)
    if key in self._loaded_modules: return self._loaded_modules[key]
    spec = importlib.util.spec_from_file_location(f"radeon_forge_hook_{expected[:16]}", path)
    if spec is None or spec.loader is None: raise StageHookError(f"cannot import hook source {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    self._loaded_modules[key] = module
    return module

  def apply(self, hooks: Sequence[Mapping[str, Any]], context: ExecutionContext) -> dict[str, Any]:
    selected = [hook for hook in _matching_hooks(hooks, context)
                if hook.get("descriptor", {}).get("adapter") == "python_transformer_block"]
    signature = tuple(_signature(hook) for hook in selected)
    if signature == self._active_signature:
      return {"changed": False, "active": [str(hook.get("activation_id")) for hook in selected], "stage": context.stage.value}
    self._reset()
    if not selected: return {"changed": True, "active": [], "stage": context.stage.value, "baseline": True}
    try:
      layers = list(self._original_layers)
      selected_parameters: dict[str, Mapping[str, Any]] = {}
      for hook in selected:
        descriptor = hook["descriptor"]
        if descriptor.get("mode", "replace") != "replace": raise StageHookError("python_transformer_block currently supports replace mode only")
        module = self._module(hook)
        parameters = hook.get("parameters", {})
        if not isinstance(parameters, Mapping): raise StageHookError("hook parameters must be a table")
        parameters = dict(parameters)
        setattr(module, "RADEON_FORGE_PARAMETERS", parameters)
        configure = getattr(module, "configure", None)
        if configure is not None:
          if not callable(configure): raise StageHookError("optional configure attribute must be callable")
          configure(parameters)
        factory = getattr(module, "build_replacement", None)
        if not callable(factory): raise StageHookError("hook must define build_replacement(original, layer_index, model, context)")
        selector = descriptor.get("selector", {})
        for index in self._indices(selector if isinstance(selector, Mapping) else {}, len(layers)):
          replacement = factory(layers[index], index, self.model, asdict(context))
          if not callable(replacement): raise StageHookError(f"replacement for layer {index} is not callable")
          layers[index] = replacement
        selected_parameters[str(hook.get("activation_id"))] = parameters
      self.model.layers = layers
      if hasattr(self.model, "forward_jit") and self.model.forward_jit is not None: self.model.forward_jit = TinyJit(self.model.forward)
      self._active_signature = signature
      return {"changed": True, "active": [str(hook.get("activation_id")) for hook in selected],
              "parameters": selected_parameters, "stage": context.stage.value, "baseline": False}
    except Exception as exc:
      self._reset()
      return {"changed": True, "active": [], "stage": context.stage.value, "baseline": True,
              "rolled_back": True, "error": str(exc), "error_type": type(exc).__name__}

  def close(self) -> None: self._reset()
