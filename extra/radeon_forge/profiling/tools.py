from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..permissions import Action
from ..runtime.tool_context import current_tool_context
from ..runtime.tools import ToolRegistry, ToolSpec
from .capture import CaptureKind, CaptureRequest, Rocprofv3Adapter


class ProfilingTools:
  def __init__(self, project_root: str | Path, evidence_root: str | Path):
    self.adapter = Rocprofv3Adapter(project_root, evidence_root)

  @staticmethod
  def _context() -> dict[str, Any]:
    context = current_tool_context()
    return asdict(context) if context is not None else {}

  @staticmethod
  def _command(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
      raise ValueError("command must be a non-empty string array")
    return tuple(value)

  def probe(self, args: Mapping[str, Any]) -> Any: return self.adapter.probe()

  def _capture(self, kind: CaptureKind, args: Mapping[str, Any]) -> Any:
    context = self._context()
    request = CaptureRequest(
      kind=kind,
      command=self._command(args.get("command")),
      stage=str(args.get("stage", "unknown")),
      session_id=str(context.get("session_id", args.get("session_id", ""))),
      trace_id=str(context.get("trace_id", args.get("trace_id", ""))),
      agent_step=int(context.get("agent_step", args.get("agent_step", 0))),
      kernel_regex=str(args.get("kernel_regex", "")),
      counters=tuple(str(x) for x in args.get("counters", ())),
      timeout_seconds=int(args.get("timeout_seconds", 900)),
      metadata={"tool_call_id":context.get("tool_call_id", ""), "tool_name":context.get("tool_name", ""),
                **dict(args.get("metadata", {}))},
    )
    return self.adapter.capture(request).to_dict()

  def capture_counters(self, args: Mapping[str, Any]) -> Any: return self._capture(CaptureKind.COUNTERS, args)
  def capture_att(self, args: Mapping[str, Any]) -> Any: return self._capture(CaptureKind.ATT, args)

  def list_captures(self, args: Mapping[str, Any]) -> Any:
    captures = []
    for path in sorted(self.adapter.evidence_root.glob("*/forge_capture_manifest.json"), reverse=True):
      try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        captures.append({"capture_id":payload.get("capture_id", path.parent.name), "kind":payload.get("kind"),
          "passed":payload.get("passed"), "elapsed_ms":payload.get("elapsed_ms"), "output_directory":str(path.parent),
          "summary":payload.get("summary", {}), "artifact_count":len(payload.get("artifacts", [])),
          "error":payload.get("error", "")})
      except Exception: continue
    return {"captures":captures[:max(1, min(int(args.get("limit", 50)), 500))]}

  def inspect_capture(self, args: Mapping[str, Any]) -> Any:
    capture_id = str(args["capture_id"])
    if "/" in capture_id or ".." in capture_id: raise ValueError("invalid capture id")
    path = self.adapter.evidence_root / capture_id / "forge_capture_manifest.json"
    if not path.is_file(): raise KeyError(f"unknown capture {capture_id}")
    return json.loads(path.read_text(encoding="utf-8"))

  def install(self, registry: ToolRegistry) -> None:
    registry.register(ToolSpec("probe_rocm_profiler", "Inspect the installed local rocprofv3/ATT capabilities without running a workload",
      {"type":"object","properties":{}}), self.probe)
    registry.register(ToolSpec("capture_rocm_counters", "Capture selected hardware performance counters for one local command and execution state",
      {"type":"object","required":["command","stage","counters"],"properties":{
        "command":{"type":"array","items":{"type":"string"}}, "stage":{"type":"string"},
        "counters":{"type":"array","items":{"type":"string"}}, "kernel_regex":{"type":"string"},
        "timeout_seconds":{"type":"integer"}, "metadata":{"type":"object"}}}, Action.BENCHMARK), self.capture_counters)
    registry.register(ToolSpec("capture_rocm_att", "Capture AMD Advanced Thread Trace/SQTT evidence for one local command and execution state",
      {"type":"object","required":["command","stage"],"properties":{
        "command":{"type":"array","items":{"type":"string"}}, "stage":{"type":"string"},
        "kernel_regex":{"type":"string"}, "timeout_seconds":{"type":"integer"},
        "metadata":{"type":"object"}}}, Action.BENCHMARK), self.capture_att)
    registry.register(ToolSpec("list_profile_captures", "List immutable local ROCm counter and ATT capture manifests",
      {"type":"object","properties":{"limit":{"type":"integer"}}}), self.list_captures)
    registry.register(ToolSpec("inspect_profile_capture", "Inspect one profile manifest, artifact hashes and normalized evidence summary",
      {"type":"object","required":["capture_id"],"properties":{"capture_id":{"type":"string"}}}), self.inspect_capture)
