from __future__ import annotations

import os, subprocess, time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..oracles.mockgpu import MockGPUOracle
from ..permissions import Action
from ..runtime.tools import ToolRegistry, ToolSpec
from .workspace import CandidateWorkspace


class OptimizationTools:
  def __init__(self, workspace: CandidateWorkspace, project_root: str | Path):
    self.workspace = workspace
    self.project_root = Path(project_root).resolve()
    self.mockgpu = MockGPUOracle(self.project_root)

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
    result = self.mockgpu.run(command, env={"RADEON_FORGE_CANDIDATE": candidate.source_path})
    updated = self.workspace.update(candidate.candidate_id, "mockgpu_passed" if result.passed else "mockgpu_failed", {"mockgpu": result.to_dict()})
    return asdict(updated)

  def benchmark_hardware(self, args: Mapping[str, Any]) -> Any:
    candidate = self.workspace.load_candidate(str(args["candidate_id"]))
    if candidate.status != "mockgpu_passed" and not bool(args.get("allow_without_mockgpu", False)):
      raise ValueError("candidate must pass MockGPU before W7900 benchmarking")
    spec = self.workspace.load_spec(candidate.spec_id)
    if not spec.hardware_command: raise ValueError("spec has no hardware benchmark command")
    command = self.workspace.render_command(spec.hardware_command, spec, candidate, self.project_root)
    env = {**os.environ, "DEV": "AMD", "RADEON_FORGE_CANDIDATE": candidate.source_path, "PYTHONUNBUFFERED": "1"}
    started = time.perf_counter_ns()
    proc = subprocess.run(command, cwd=self.project_root, env=env, text=True, capture_output=True, timeout=int(args.get("timeout_seconds", 900)))
    evidence = {"command": command, "returncode": proc.returncode, "elapsed_ms": (time.perf_counter_ns()-started)/1e6,
                "stdout": proc.stdout[-50000:], "stderr": proc.stderr[-50000:], "target": "gfx1100"}
    updated = self.workspace.update(candidate.candidate_id, "hardware_passed" if proc.returncode == 0 else "hardware_failed", {"hardware": evidence})
    return asdict(updated)

  def install(self, registry: ToolRegistry) -> None:
    registry.register(ToolSpec("list_kernel_specs", "List typed megakernel and fused-kernel contracts", {"type":"object","properties":{}}), self.list_specs)
    registry.register(ToolSpec("list_kernel_candidates", "List staged generated implementations and their oracle status", {"type":"object","properties":{"spec_id":{"type":"string"}}}), self.list_candidates)
    registry.register(ToolSpec("stage_kernel_candidate", "Write one disposable implementation for a typed kernel specification", {"type":"object","required":["spec_id","source","hypothesis"],"properties":{"spec_id":{"type":"string"},"source":{"type":"string"},"hypothesis":{"type":"string"},"parent_id":{"type":"string"}}}, Action.WRITE_GENERATED_SOURCE), self.stage_candidate)
    registry.register(ToolSpec("validate_candidate_mockgpu", "Execute a generated AMD candidate through tinygrad's RDNA3 MockGPU semantic oracle", {"type":"object","required":["candidate_id"],"properties":{"candidate_id":{"type":"string"}}}, Action.COMPILE), self.validate_mockgpu)
    registry.register(ToolSpec("benchmark_candidate_w7900", "Benchmark a MockGPU-passing candidate on the real local W7900", {"type":"object","required":["candidate_id"],"properties":{"candidate_id":{"type":"string"},"timeout_seconds":{"type":"integer"},"allow_without_mockgpu":{"type":"boolean"}}}, Action.BENCHMARK), self.benchmark_hardware)
