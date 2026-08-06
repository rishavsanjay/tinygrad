from __future__ import annotations

import hashlib, json, threading, time, uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .workspace import CandidateWorkspace, KernelSpec


class HookLayer(str, Enum):
  AGENT_LOOP = "agent_loop"
  SCHEDULER = "scheduler"
  MODEL = "model"
  TRANSFORMER_BLOCK = "transformer_block"
  SUBGRAPH = "subgraph"
  KERNEL = "kernel"
  KV_CACHE = "kv_cache"
  SAMPLER = "sampler"


class HookMode(str, Enum):
  BEFORE = "before"
  AFTER = "after"
  REPLACE = "replace"
  WRAP = "wrap"


class ExecutionStage(str, Enum):
  """Coarse states in one agent/inference execution.

  Stage is deliberately orthogonal to HookLayer. A kernel or block replacement
  can be valid for decode but harmful for prefill; a scheduler optimization may
  only apply while resuming after a tool call; a KV optimization may only apply
  while appending to a warm cache.
  """
  ANY = "any"
  SESSION_START = "session_start"
  PREFILL = "prefill"
  FIRST_TOKEN = "first_token"
  DECODE = "decode"
  TOOL_EXECUTION = "tool_execution"
  TOOL_RESUME = "tool_resume"
  KV_APPEND = "kv_append"
  SAMPLING = "sampling"
  SESSION_END = "session_end"


@dataclass(frozen=True)
class ExecutionContext:
  stage: ExecutionStage
  batch_size: int = 1
  prompt_tokens: int = 0
  context_tokens: int = 0
  generated_token_index: int = -1
  prefix_reused_tokens: int = 0
  tool_round: int = 0
  warm: bool = False
  attributes: Mapping[str, Any] = field(default_factory=dict)

  @classmethod
  def from_mapping(cls, value: Mapping[str, Any]) -> ExecutionContext:
    raw = dict(value)
    stage = ExecutionStage(str(raw.pop("stage")))
    known = {key: raw.pop(key, default) for key, default in (
      ("batch_size", 1), ("prompt_tokens", 0), ("context_tokens", 0), ("generated_token_index", -1),
      ("prefix_reused_tokens", 0), ("tool_round", 0), ("warm", False))}
    return cls(stage, int(known["batch_size"]), int(known["prompt_tokens"]), int(known["context_tokens"]),
               int(known["generated_token_index"]), int(known["prefix_reused_tokens"]), int(known["tool_round"]),
               bool(known["warm"]), raw)

  def flattened(self) -> dict[str, Any]:
    return {"stage": self.stage.value, "batch_size": self.batch_size, "prompt_tokens": self.prompt_tokens,
            "context_tokens": self.context_tokens, "generated_token_index": self.generated_token_index,
            "prefix_reused_tokens": self.prefix_reused_tokens, "tool_round": self.tool_round, "warm": self.warm,
            "prefix_cache": "hit" if self.prefix_reused_tokens > 0 else "miss", **dict(self.attributes)}


@dataclass(frozen=True)
class StagePredicate:
  stages: tuple[ExecutionStage, ...] = (ExecutionStage.ANY,)
  min_context_tokens: int | None = None
  max_context_tokens: int | None = None
  min_prompt_tokens: int | None = None
  max_prompt_tokens: int | None = None
  min_generated_token_index: int | None = None
  max_generated_token_index: int | None = None
  batch_sizes: tuple[int, ...] = ()
  prefix_cache: str = "any"  # any, hit, miss
  warm: bool | None = None
  conditions: Mapping[str, Any] = field(default_factory=dict)

  @classmethod
  def from_mapping(cls, value: Mapping[str, Any] | None) -> StagePredicate:
    raw = dict(value or {})
    stage_value = raw.pop("stages", raw.pop("stage", ["any"]))
    if isinstance(stage_value, str): stage_value = [stage_value]
    if not isinstance(stage_value, Sequence): raise ValueError("hook stages must be a string or array")
    stages = tuple(ExecutionStage(str(x)) for x in stage_value)
    batch_value = raw.pop("batch_sizes", raw.pop("batch_size", ()))
    if isinstance(batch_value, int): batch_value = [batch_value]
    if not isinstance(batch_value, Sequence): raise ValueError("hook batch_sizes must be an integer or array")
    prefix_cache = str(raw.pop("prefix_cache", "any"))
    if prefix_cache not in {"any", "hit", "miss"}: raise ValueError("prefix_cache must be any, hit, or miss")
    warm = raw.pop("warm", None)
    if warm is not None and not isinstance(warm, bool): raise ValueError("warm must be boolean")
    fields = {}
    for name in ("min_context_tokens", "max_context_tokens", "min_prompt_tokens", "max_prompt_tokens",
                 "min_generated_token_index", "max_generated_token_index"):
      item = raw.pop(name, None)
      fields[name] = None if item is None else int(item)
    explicit_conditions = raw.pop("conditions", {})
    if not isinstance(explicit_conditions, Mapping): raise ValueError("hook conditions must be a table")
    conditions = {**raw, **dict(explicit_conditions)}
    return cls(stages, **fields, batch_sizes=tuple(int(x) for x in batch_value), prefix_cache=prefix_cache, warm=warm,
               conditions=conditions)

  @property
  def signature(self) -> str:
    return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:12]

  def matches(self, context: ExecutionContext) -> bool:
    if ExecutionStage.ANY not in self.stages and context.stage not in self.stages: return False
    if self.min_context_tokens is not None and context.context_tokens < self.min_context_tokens: return False
    if self.max_context_tokens is not None and context.context_tokens > self.max_context_tokens: return False
    if self.min_prompt_tokens is not None and context.prompt_tokens < self.min_prompt_tokens: return False
    if self.max_prompt_tokens is not None and context.prompt_tokens > self.max_prompt_tokens: return False
    if self.min_generated_token_index is not None and context.generated_token_index < self.min_generated_token_index: return False
    if self.max_generated_token_index is not None and context.generated_token_index > self.max_generated_token_index: return False
    if self.batch_sizes and context.batch_size not in self.batch_sizes: return False
    if self.prefix_cache != "any" and context.flattened()["prefix_cache"] != self.prefix_cache: return False
    if self.warm is not None and context.warm is not self.warm: return False
    flat = context.flattened()
    return all(_matches(expected, flat.get(str(key))) for key, expected in self.conditions.items())


@dataclass(frozen=True)
class HookDescriptor:
  layer: HookLayer
  target: str
  mode: HookMode = HookMode.REPLACE
  adapter: str = "request_metadata"
  selector: Mapping[str, Any] = field(default_factory=dict)
  when: StagePredicate = field(default_factory=StagePredicate)
  priority: int = 0
  exclusive_group: str = ""
  description: str = ""

  @classmethod
  def from_spec(cls, spec: KernelSpec) -> HookDescriptor:
    raw = spec.metadata.get("hook", {})
    if not isinstance(raw, Mapping): raise ValueError("spec metadata hook must be a table")
    layer = HookLayer(str(raw.get("layer", "kernel")))
    target = str(raw.get("target", spec.operation)).strip()
    if not target: raise ValueError("hook target must not be empty")
    selector = raw.get("selector", {})
    if not isinstance(selector, Mapping): raise ValueError("hook selector must be a table")
    when_value = raw.get("when", {key: raw[key] for key in ("stage", "stages") if key in raw})
    predicate = StagePredicate.from_mapping(when_value if isinstance(when_value, Mapping) else {"stages": when_value})
    group = str(raw.get("exclusive_group", f"{layer.value}:{target}"))
    return cls(layer, target, HookMode(str(raw.get("mode", "replace"))), str(raw.get("adapter", "request_metadata")),
               dict(selector), predicate, int(raw.get("priority", 0)), group, str(raw.get("description", "")))

  @property
  def slot_key(self) -> str: return f"{self.exclusive_group}:{self.when.signature}"


@dataclass(frozen=True)
class RuntimeFingerprint:
  architecture: str = ""
  gpu: str = ""
  runtime: str = ""
  runtime_revision: str = ""
  model_family: str = ""
  model_hash: str = ""
  dtype: str = ""
  shapes: Mapping[str, Any] = field(default_factory=dict)
  extra: Mapping[str, Any] = field(default_factory=dict)

  @classmethod
  def from_mapping(cls, value: Mapping[str, Any] | None) -> RuntimeFingerprint:
    raw = dict(value or {})
    known = {key: raw.pop(key, "") for key in ("architecture", "gpu", "runtime", "runtime_revision", "model_family", "model_hash", "dtype")}
    shapes = raw.pop("shapes", {})
    return cls(**known, shapes=dict(shapes) if isinstance(shapes, Mapping) else {}, extra=raw)

  def flattened(self) -> dict[str, Any]:
    ret = {"architecture": self.architecture, "gpu": self.gpu, "runtime": self.runtime,
           "runtime_revision": self.runtime_revision, "model_family": self.model_family,
           "model_hash": self.model_hash, "dtype": self.dtype}
    ret.update({f"shape.{key}": value for key, value in self.shapes.items()})
    ret.update(self.extra)
    return ret


@dataclass(frozen=True)
class CompatibilityReport:
  compatible: bool
  exact: tuple[str, ...]
  unknown: tuple[str, ...]
  mismatches: tuple[str, ...]


def _matches(expected: Any, actual: Any) -> bool:
  if expected in (None, "", "*"): return True
  if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)): return actual in expected
  if isinstance(expected, str) and expected.endswith("*"): return str(actual).startswith(expected[:-1])
  return expected == actual


def check_compatibility(spec: KernelSpec, fingerprint: RuntimeFingerprint) -> CompatibilityReport:
  expected = spec.metadata.get("recipe_compatibility", spec.metadata.get("compatibility", {}))
  if not isinstance(expected, Mapping): raise ValueError("compatibility contract must be a table")
  actual = fingerprint.flattened()
  exact, unknown, mismatches = [], [], []
  if spec.target:
    arch = actual.get("architecture", "")
    if not arch: unknown.append("architecture")
    elif _matches(spec.target, arch): exact.append("architecture")
    else: mismatches.append(f"architecture expected={spec.target!r} actual={arch!r}")
  for key, value in expected.items():
    key = str(key)
    current = actual.get(key)
    if current in (None, ""): unknown.append(key)
    elif _matches(value, current): exact.append(key)
    else: mismatches.append(f"{key} expected={value!r} actual={current!r}")
  return CompatibilityReport(not mismatches, tuple(exact), tuple(unknown), tuple(mismatches))


@dataclass(frozen=True)
class ActiveHook:
  activation_id: str
  candidate_id: str
  spec_id: str
  source_path: str
  source_sha256: str
  descriptor: HookDescriptor
  compatibility: CompatibilityReport
  activated_at_s: float
  reason: str
  previous_activation_id: str | None = None


class HookActivationError(RuntimeError): pass


def _descriptor_from_dict(desc: Mapping[str, Any]) -> HookDescriptor:
  when = desc.get("when", {})
  return HookDescriptor(HookLayer(desc["layer"]), str(desc["target"]), HookMode(desc["mode"]), str(desc["adapter"]),
                        dict(desc.get("selector", {})), StagePredicate.from_mapping(when), int(desc.get("priority", 0)),
                        str(desc.get("exclusive_group", "")), str(desc.get("description", "")))


class HookRegistry:
  """Persistent, rollback-safe, execution-state-aware optimization registry."""
  SUPPORTED_ADAPTERS = frozenset({"request_metadata", "python_transformer_block"})

  def __init__(self, workspace: CandidateWorkspace):
    self.workspace = workspace
    self.root = workspace.root / "hooks"
    self.root.mkdir(parents=True, exist_ok=True)
    self.active_path, self.history_path = self.root / "active.json", self.root / "history.jsonl"
    self._lock = threading.RLock()
    if not self.active_path.exists(): self.active_path.write_text("[]\n", encoding="utf-8")

  def _load_active(self) -> list[ActiveHook]:
    try: payload = json.loads(self.active_path.read_text(encoding="utf-8"))
    except Exception: payload = []
    ret = []
    for item in payload:
      try:
        comp = item["compatibility"]
        ret.append(ActiveHook(item["activation_id"], item["candidate_id"], item["spec_id"], item["source_path"],
                              item["source_sha256"], _descriptor_from_dict(item["descriptor"]),
                              CompatibilityReport(bool(comp["compatible"]), tuple(comp.get("exact", ())),
                              tuple(comp.get("unknown", ())), tuple(comp.get("mismatches", ()))), float(item["activated_at_s"]),
                              item["reason"], item.get("previous_activation_id")))
      except Exception: continue
    return ret

  def _save_active(self, hooks: Sequence[ActiveHook]) -> None:
    temp = self.active_path.with_suffix(".tmp")
    temp.write_text(json.dumps([asdict(x) for x in hooks], indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(self.active_path)

  def _history(self, event: str, **data: Any) -> None:
    with self.history_path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps({"event": event, "timestamp_s": time.time(), **data}, default=str) + "\n")

  def active(self) -> list[ActiveHook]:
    with self._lock: return sorted(self._load_active(), key=lambda x: (-x.descriptor.priority, x.activated_at_s))

  def resolve(self, context: ExecutionContext) -> list[ActiveHook]:
    """Select the highest-priority matching implementation per logical hook group."""
    selected: dict[str, ActiveHook] = {}
    for hook in self.active():
      if not hook.descriptor.when.matches(context): continue
      group = hook.descriptor.exclusive_group
      current = selected.get(group)
      if current is None or hook.descriptor.priority > current.descriptor.priority: selected[group] = hook
    return sorted(selected.values(), key=lambda x: (-x.descriptor.priority, x.activated_at_s))

  def _required_status(self, spec: KernelSpec) -> frozenset[str]:
    heldout = spec.metadata.get("heldout_command", ())
    return frozenset({"heldout_passed", "accepted"}) if heldout else frozenset({"hardware_passed", "heldout_passed", "accepted"})

  def activate(self, candidate_id: str, fingerprint: RuntimeFingerprint, reason: str) -> ActiveHook:
    if not reason.strip(): raise HookActivationError("activation requires a reason")
    with self._lock:
      candidate = self.workspace.load_candidate(candidate_id)
      spec = self.workspace.load_spec(candidate.spec_id)
      descriptor = HookDescriptor.from_spec(spec)
      if descriptor.adapter not in self.SUPPORTED_ADAPTERS: raise HookActivationError(f"runtime adapter {descriptor.adapter!r} is not installed")
      if candidate.status not in self._required_status(spec):
        raise HookActivationError(f"candidate status {candidate.status!r} has not passed the required hardware/held-out oracle")
      source_path = Path(candidate.source_path)
      if not source_path.is_file(): raise HookActivationError("candidate source is missing")
      source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
      if source_sha != candidate.source_sha256: raise HookActivationError("candidate source hash changed after validation")
      compatibility = check_compatibility(spec, fingerprint)
      if not compatibility.compatible: raise HookActivationError("incompatible runtime: " + "; ".join(compatibility.mismatches))
      current = self._load_active()
      previous = next((x for x in current if x.descriptor.slot_key == descriptor.slot_key), None)
      current = [x for x in current if x.descriptor.slot_key != descriptor.slot_key]
      active = ActiveHook(uuid.uuid4().hex, candidate.candidate_id, spec.spec_id, candidate.source_path, source_sha, descriptor,
                          compatibility, time.time(), reason.strip(), previous.activation_id if previous else None)
      current.append(active)
      self._save_active(current)
      self._history("hook_activated", activation=asdict(active), replaced=asdict(previous) if previous else None)
      return active

  def deactivate(self, activation_id: str, reason: str) -> ActiveHook:
    if not reason.strip(): raise HookActivationError("deactivation requires a reason")
    with self._lock:
      current = self._load_active()
      target = next((x for x in current if x.activation_id == activation_id), None)
      if target is None: raise KeyError(f"unknown active hook {activation_id}")
      self._save_active([x for x in current if x.activation_id != activation_id])
      self._history("hook_deactivated", activation=asdict(target), reason=reason.strip())
      return target

  def clear(self, reason: str) -> list[ActiveHook]:
    with self._lock:
      current = self._load_active()
      self._save_active([])
      self._history("hooks_cleared", activations=[asdict(x) for x in current], reason=reason.strip())
      return current

  @staticmethod
  def _runtime_item(active: ActiveHook) -> dict[str, Any] | None:
    source = Path(active.source_path)
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != active.source_sha256: return None
    return {"activation_id": active.activation_id, "candidate_id": active.candidate_id, "spec_id": active.spec_id,
            "source_path": active.source_path, "source_sha256": active.source_sha256,
            "descriptor": asdict(active.descriptor)}

  def runtime_metadata(self, context: ExecutionContext | None = None) -> dict[str, Any]:
    source = self.resolve(context) if context is not None else self.active()
    hooks = [item for active in source if (item := self._runtime_item(active)) is not None]
    return {"active_hooks": hooks, "execution_context": asdict(context) if context is not None else None}
