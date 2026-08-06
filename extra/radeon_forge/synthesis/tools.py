from __future__ import annotations

import os, subprocess, time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from ..oracles.mockgpu import MockGPUOracle
from ..permissions import Action
from ..runtime.tools import ToolRegistry, ToolSpec
from .hooks import HookRegistry, RuntimeFingerprint, check_compatibility
from .portable import export_recipe_with_hook
from .recipe import RecipeLibrary
from .workspace import CandidateWorkspace


class OptimizationTools:
  def __init__(self, workspace: CandidateWorkspace, project_root: str | Path, hooks: HookRegistry | None = None,
               fingerprint_provider: Callable[[], RuntimeFingerprint] | None = None):
    self.workspace = workspace
    self.project_root = Path(project_root).resolve()
    self.mockgpu = MockGPUOracle(self.project_root)
    self.recipes = RecipeLibrary(workspace)
    self.hooks, self.fingerprint_provider = hooks, fingerprint_provider

  def _project_path(self, value: str, *, must_exist: bool = False) -> Path:
    path = (self.project_root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if path != self.project_root and self.project_root not in path.parents: raise ValueError("path must stay inside the private workspace")
    if must_exist and not path.is_file(): raise FileNotFoundError(path)
    return path

  def _fingerprint(self) -> RuntimeFingerprint:
    return self.fingerprint_provider() if self.fingerprint_provider is not None else RuntimeFingerprint()

  def list_specs(self, args: Mapping[str, Any]) -> Any:
    return {"specs": [asdict(x) | {"spec_id": x.spec_id} for x in self.workspace.specs()]}

  def list_candidates(self, args: Mapping[str, Any]) -> Any:
    return {"candidates": [asdict(x) for x in self.workspace.candidates(args.get("spec_id"))]}

  def stage_candidate(self, args: Mapping[str, Any]) -> Any:
    record = self.workspace.create_candidate(str(args["spec_id"]), str(args["source"]), str(args["hypothesis"]), args.get("parent_id"))
    return asdict(record)

  def validate_mockgpu(self, args: Mapping[str, Any]) -> Any:
    candidate = self.workspace.load_candidate(str(args["candidate_id"]))
    spec = self.workspace.load_spec(candidate.spec_id)
    if not spec.mockgpu_command: raise ValueError("spec has no MockGPU oracle command")
    command = self.workspace.render_command(spec.mockgpu_command, spec, candidate, self.project_root)
    result = self.mockgpu.run(command, env={"RADEON_FORGE_CANDIDATE": candidate.source_path,
                                            "RADEON_FORGE_RECIPE_BUNDLE": str(spec.metadata.get("recipe_bundle", ""))})
    updated = self.workspace.update(candidate.candidate_id, "mockgpu_passed" if result.passed else "mockgpu_failed", {"mockgpu": result.to_dict()})
    return asdict(updated)

  def _run_hardware_command(self, candidate_id: str, command_value, timeout_seconds: int, evidence_key: str,
                            pass_status: str, fail_status: str) -> Any:
    candidate = self.workspace.load_candidate(candidate_id)
    spec = self.workspace.load_spec(candidate.spec_id)
    command = self.workspace.render_command(command_value, spec, candidate, self.project_root)
    env = {**os.environ, "DEV": "AMD", "RADEON_FORGE_CANDIDATE": candidate.source_path,
           "RADEON_FORGE_RECIPE_BUNDLE": str(spec.metadata.get("recipe_bundle", "")), "PYTHONUNBUFFERED": "1"}
    started = time.perf_counter_ns()
    proc = subprocess.run(command, cwd=str(spec.metadata.get("recipe_bundle", self.project_root)), env=env, text=True, capture_output=True,
                          timeout=timeout_seconds)
    evidence = {"command": command, "returncode": proc.returncode, "elapsed_ms": (time.perf_counter_ns()-started)/1e6,
                "stdout": proc.stdout[-50000:], "stderr": proc.stderr[-50000:], "target": spec.target}
    updated = self.workspace.update(candidate.candidate_id, pass_status if proc.returncode == 0 else fail_status, {evidence_key: evidence})
    return asdict(updated)

  def benchmark_hardware(self, args: Mapping[str, Any]) -> Any:
    candidate = self.workspace.load_candidate(str(args["candidate_id"]))
    if candidate.status != "mockgpu_passed" and not bool(args.get("allow_without_mockgpu", False)):
      raise ValueError("candidate must pass MockGPU before W7900 benchmarking")
    spec = self.workspace.load_spec(candidate.spec_id)
    if not spec.hardware_command: raise ValueError("spec has no hardware benchmark command")
    return self._run_hardware_command(candidate.candidate_id, spec.hardware_command, int(args.get("timeout_seconds", 900)),
                                      "hardware", "hardware_passed", "hardware_failed")

  def validate_heldout(self, args: Mapping[str, Any]) -> Any:
    candidate = self.workspace.load_candidate(str(args["candidate_id"]))
    if candidate.status != "hardware_passed": raise ValueError("candidate must pass the real hardware benchmark first")
    spec = self.workspace.load_spec(candidate.spec_id)
    command = tuple(str(x) for x in spec.metadata.get("heldout_command", ()))
    if not command: raise ValueError("spec has no held-out validation command")
    return self._run_hardware_command(candidate.candidate_id, command, int(args.get("timeout_seconds", 900)),
                                      "heldout", "heldout_passed", "heldout_failed")

  def list_recipes(self, args: Mapping[str, Any]) -> Any:
    return {"recipes": [asdict(x) for x in self.recipes.installed()]}

  def inspect_recipe(self, args: Mapping[str, Any]) -> Any:
    return self.recipes.inspect(str(args["recipe_id"]))

  def import_recipe(self, args: Mapping[str, Any]) -> Any:
    path = self._project_path(str(args["path"]), must_exist=True)
    return asdict(self.recipes.install_file(path))

  def export_recipe_file(self, args: Mapping[str, Any]) -> Any:
    output = self._project_path(str(args["output"]))
    path = export_recipe_with_hook(self.workspace, str(args["spec_id"]), output,
                                   str(args["candidate_id"]) if args.get("candidate_id") else None)
    return {"path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(), "portable": True,
            "includes_execution_stage_hook": True}

  def list_active_hooks(self, args: Mapping[str, Any]) -> Any:
    if self.hooks is None: return {"hooks": [], "available": False}
    return {"hooks": [asdict(x) for x in self.hooks.active()], "available": True,
            "runtime_fingerprint": asdict(self._fingerprint())}

  def activate_hook(self, args: Mapping[str, Any]) -> Any:
    if self.hooks is None: raise RuntimeError("hook registry is unavailable")
    candidate = self.workspace.load_candidate(str(args["candidate_id"]))
    spec = self.workspace.load_spec(candidate.spec_id)
    fingerprint = self._fingerprint()
    allow_unknown = bool(args.get("allow_unknown_runtime", False))
    report = check_compatibility(spec, fingerprint)
    if report.mismatches: raise ValueError("runtime is incompatible: " + "; ".join(report.mismatches))
    if report.unknown and not allow_unknown:
      raise ValueError("runtime compatibility is incomplete: missing " + ", ".join(report.unknown) +
                       ". Regenerate locally or explicitly approve an unknown-runtime deployment.")
    active = self.hooks.activate(candidate.candidate_id, fingerprint,
                                 str(args.get("reason", "Approved stage-specific local deployment")))
    return asdict(active)

  def deactivate_hook(self, args: Mapping[str, Any]) -> Any:
    if self.hooks is None: raise RuntimeError("hook registry is unavailable")
    return asdict(self.hooks.deactivate(str(args["activation_id"]), str(args.get("reason", "User-requested rollback"))))

  def install(self, registry: ToolRegistry) -> None:
    registry.register(ToolSpec("list_kernel_specs", "List typed megakernel and fused-kernel contracts", {"type":"object","properties":{}}), self.list_specs)
    registry.register(ToolSpec("list_kernel_candidates", "List staged generated implementations and their oracle status", {"type":"object","properties":{"spec_id":{"type":"string"}}}), self.list_candidates)
    registry.register(ToolSpec("stage_kernel_candidate", "Write one disposable implementation for a typed kernel specification", {"type":"object","required":["spec_id","source","hypothesis"],"properties":{"spec_id":{"type":"string"},"source":{"type":"string"},"hypothesis":{"type":"string"},"parent_id":{"type":"string"}}}, Action.WRITE_GENERATED_SOURCE), self.stage_candidate)
    registry.register(ToolSpec("validate_candidate_mockgpu", "Execute a generated AMD candidate through tinygrad's RDNA3 MockGPU semantic oracle", {"type":"object","required":["candidate_id"],"properties":{"candidate_id":{"type":"string"}}}, Action.COMPILE), self.validate_mockgpu)
    registry.register(ToolSpec("benchmark_candidate_w7900", "Benchmark a MockGPU-passing candidate on the real local W7900", {"type":"object","required":["candidate_id"],"properties":{"candidate_id":{"type":"string"},"timeout_seconds":{"type":"integer"},"allow_without_mockgpu":{"type":"boolean"}}}, Action.BENCHMARK), self.benchmark_hardware)
    registry.register(ToolSpec("validate_candidate_heldout", "Run a hardware-passing candidate on its held-out correctness and task-quality suite", {"type":"object","required":["candidate_id"],"properties":{"candidate_id":{"type":"string"},"timeout_seconds":{"type":"integer"}}}, Action.BENCHMARK), self.validate_heldout)
    registry.register(ToolSpec("list_forge_recipes", "List portable installed optimization recipes", {"type":"object","properties":{}}), self.list_recipes)
    registry.register(ToolSpec("inspect_forge_recipe", "Read the intent, invariants, execution-stage hook, oracle and artifact inventory of an installed optimization recipe", {"type":"object","required":["recipe_id"],"properties":{"recipe_id":{"type":"string"}}}), self.inspect_recipe)
    registry.register(ToolSpec("import_forge_recipe", "Install one portable .forge.toml optimization recipe from the private workspace", {"type":"object","required":["path"],"properties":{"path":{"type":"string"}}}, Action.WRITE_GENERATED_SOURCE), self.import_recipe)
    registry.register(ToolSpec("export_forge_recipe", "Export a contract, execution-stage hook and optional implementation cache as one shareable .forge.toml file", {"type":"object","required":["spec_id","output"],"properties":{"spec_id":{"type":"string"},"candidate_id":{"type":"string"},"output":{"type":"string"}}}, Action.WRITE_GENERATED_SOURCE), self.export_recipe_file)
    registry.register(ToolSpec("list_active_optimization_hooks", "List deployed optimizations and the execution states in which each can run", {"type":"object","properties":{}}), self.list_active_hooks)
    registry.register(ToolSpec("activate_optimization_hook", "Deploy a validated optimization into its declared execution stages; activation is compatibility-gated and rollback-safe", {"type":"object","required":["candidate_id"],"properties":{"candidate_id":{"type":"string"},"reason":{"type":"string"},"allow_unknown_runtime":{"type":"boolean"}}}, Action.DEPLOY), self.activate_hook)
    registry.register(ToolSpec("deactivate_optimization_hook", "Rollback one active optimization hook", {"type":"object","required":["activation_id"],"properties":{"activation_id":{"type":"string"},"reason":{"type":"string"}}}, Action.DEPLOY), self.deactivate_hook)
