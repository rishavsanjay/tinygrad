import tempfile
import unittest
from pathlib import Path

from extra.radeon_forge.backends.fake import ScriptedBackend
from extra.radeon_forge.permissions import PermissionDenied
from extra.radeon_forge.profiling.report import build_profile_report
from extra.radeon_forge.runtime import ForgeEngine
from extra.radeon_forge.runtime.backend import BackendCapabilities, GenerationEvent
from extra.radeon_forge.runtime.session import SessionState
from extra.radeon_forge.runtime.tools import WorkspaceTools
from extra.radeon_forge.synthesis.workspace import CandidateWorkspace, KernelSpec


class KernelEvidenceBackend:
  @property
  def name(self): return "kernel-evidence-local"

  @property
  def capabilities(self): return BackendCapabilities(prefix_cache=True, persistent_kv=True, kernel_metrics=True)

  @property
  def runtime_metadata(self): return {"runtime":"test", "architecture":"gfx1100", "model_family":"llama"}

  def stream(self, request):
    yield GenerationEvent("prefill", metrics={"wall_ms": 8.0, "gpu_ms": 5.0, "prompt_tokens": 24,
                                               "prefix_reused_tokens": 12, "profile_kernel_events": 1})
    yield GenerationEvent("kernel", metrics={"name": "rmsnorm_fused", "duration_ms": 0.40, "stage": "prefill", "device": "AMD"})
    yield GenerationEvent("token", "first", metrics={"index": 0, "stage": "first_token", "wall_ms": 4.0, "gpu_ms": 2.5,
                                                       "kernel_count": 24, "profile_kernel_events": 1})
    yield GenerationEvent("kernel", metrics={"name": "first_token_projection", "duration_ms": 0.80,
                                              "stage": "first_token", "device": "AMD"})
    yield GenerationEvent("token", "ok", metrics={"index": 1, "stage": "decode", "wall_ms": 2.0, "gpu_ms": 1.4,
                                                    "kernel_count": 24, "profile_kernel_events": 2})
    yield GenerationEvent("kernel", metrics={"name": "decode_gemv", "duration_ms": 0.90, "stage": "decode", "device": "AMD"})
    yield GenerationEvent("kernel", metrics={"name": "decode_gemv", "duration_ms": 0.70, "stage": "decode", "device": "AMD"})
    yield GenerationEvent("done", finish_reason="stop", metrics={"generated_tokens": 2})

  def close(self): pass


class TestRadeonForgeRuntime(unittest.TestCase):
  def test_kernel_evidence_survives_unified_trace_and_is_ranked_by_stage(self):
    with tempfile.TemporaryDirectory() as directory:
      engine = ForgeEngine(KernelEvidenceBackend(), directory)
      session = engine.create_session()
      session.send("profile two execution states")
      report = build_profile_report(session.trace.events())
      top = report["summary"]["top_kernels"]
      self.assertEqual(top[0]["name"], "decode_gemv")
      self.assertEqual(top[0]["calls"], 2)
      self.assertAlmostEqual(top[0]["total_ms"], 1.6)
      self.assertEqual(report["summary"]["top_kernels_by_stage"]["prefill"][0]["name"], "rmsnorm_fused")
      self.assertEqual(report["summary"]["top_kernels_by_stage"]["first_token"][0]["name"], "first_token_projection")
      self.assertEqual(report["summary"]["top_kernels_by_stage"]["decode"][0]["name"], "decode_gemv")
      self.assertTrue(report["summary"]["kernel_profile_complete"])
      titles = {x["title"] for x in report["findings"]}
      self.assertIn("One kernel family dominates captured GPU time overall", titles)
      self.assertIn("Different execution stages have different dominant kernels", titles)
      self.assertIn("First-token and steady-decode latency are materially different", titles)
      kernel_events = [event for event in session.trace.events() if event.kind == "kernel"]
      self.assertEqual([event.name for event in kernel_events], ["rmsnorm_fused", "first_token_projection", "decode_gemv", "decode_gemv"])
      engine.close()

  def test_permissioned_tool_round_trip(self):
    responses = [
      '<tool_call>{"name":"read_file","arguments":{"path":"hello.txt"}}</tool_call>',
      "The private file was read successfully.",
    ]
    with tempfile.TemporaryDirectory() as directory:
      Path(directory, "hello.txt").write_text("local only", encoding="utf-8")
      engine = ForgeEngine(ScriptedBackend(responses), directory)
      session = engine.create_session()
      session.send("read hello.txt")
      self.assertEqual(session.state, SessionState.AWAITING_TOOL_APPROVAL)
      with self.assertRaises(PermissionDenied): session.approve_tool("not-a-real-grant")
      self.assertEqual(session.state, SessionState.AWAITING_TOOL_APPROVAL)
      token = engine.grant_for_pending_tool(session.session_id, "approve one local read")
      session.approve_tool(token)
      self.assertEqual(session.state, SessionState.COMPLETED)
      self.assertEqual(session.messages[-1]["content"], "The private file was read successfully.")
      self.assertTrue(any(event.kind == "tool_result" for event in session.events))
      engine.close()

  def test_workspace_paths_cannot_escape(self):
    with tempfile.TemporaryDirectory() as directory:
      tools = WorkspaceTools(directory)
      with self.assertRaisesRegex(ValueError, "escapes workspace"):
        tools.read_file({"path": "../outside.txt"})
      with self.assertRaisesRegex(ValueError, "not allowlisted"):
        tools.run_command({"command": "curl https://example.com"})

  def test_generated_candidates_are_content_addressed(self):
    with tempfile.TemporaryDirectory() as directory:
      workspace = CandidateWorkspace(directory)
      spec = workspace.save_spec(KernelSpec(name="decode-gemv", operation="gemv", shapes={"hidden": 4096},
                                            invariants=("numerically_matches_reference",)))
      first = workspace.create_candidate(spec.spec_id, "def build_kernel():\n  return 1\n", "baseline schedule")
      second = workspace.create_candidate(spec.spec_id, "def build_kernel():\n  return 1\n", "same implementation, new explanation")
      self.assertEqual(first.candidate_id, second.candidate_id)
      self.assertEqual(first.source_sha256, second.source_sha256)
      self.assertTrue(Path(second.source_path).is_file())


if __name__ == "__main__": unittest.main()
