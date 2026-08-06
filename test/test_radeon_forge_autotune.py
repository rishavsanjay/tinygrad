import json
import sys
import tempfile
import unittest
from pathlib import Path

from extra.radeon_forge.synthesis.autotune import SearchPlan, run_autotune
from extra.radeon_forge.synthesis.workspace import CandidateWorkspace, KernelSpec


class TestStageAutotune(unittest.TestCase):
  def test_fast_invalid_configuration_cannot_win(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      runner = root / "runner.py"
      runner.write_text('''import json, os
candidate = json.loads(os.environ["RADEON_FORGE_CANDIDATE_JSON"])
tile = candidate["parameters"]["TILE"]
budget = int(os.environ["RADEON_FORGE_BUDGET"])
payload = {
  "samples_us": ([1.0] if tile == 1 else [2.0]) * budget,
  "correctness": {"passed": True, "max_abs_error": 0.0, "max_rel_error": 0.0, "checked_values": 128},
  "resources": {"vgprs": 128 if tile == 1 else 64, "lds_bytes": 4096, "spilled_vgprs": 0, "spilled_sgprs": 0},
  "compile_ok": True,
  "stable": True
}
print(json.dumps(payload))
''', encoding="utf-8")
      workspace = CandidateWorkspace(root / "forge")
      spec = workspace.save_spec(KernelSpec(
        name="decode-search", operation="batch-one decode subgraph", target="gfx1100", invariants=("match reference",),
        hardware_command=(sys.executable, str(runner)),
        metadata={"recipe_acceptance":{"max_vgprs":96, "forbid_spills":True},
                  "hook":{"layer":"subgraph", "target":"decode.projection", "adapter":"request_metadata",
                          "when":{"stages":["decode"], "batch_sizes":[1]}, "exclusive_group":"decode-projection"}},
      ))
      candidate = workspace.create_candidate(spec.spec_id, "# parameterized candidate\n", "search tile size")
      workspace.update(candidate.candidate_id, "mockgpu_passed", {"mockgpu":{"passed":True}})
      updated, summary = run_autotune(workspace, candidate.candidate_id, root,
                                      SearchPlan.from_mapping({"axes":{"TILE":[1,2]}, "budgets":[1,2], "reduction":2}))
      self.assertIsNotNone(summary.winner)
      self.assertEqual(summary.winner.candidate.parameters["TILE"], 2)
      self.assertEqual(updated.status, "hardware_passed")
      self.assertEqual(updated.evidence["selected_parameters"], {"TILE":2})
      persisted = workspace.load_candidate(candidate.candidate_id)
      self.assertEqual(persisted.evidence["selected_parameters"], {"TILE":2})
      self.assertTrue(Path(persisted.evidence["autotune"]["ledger"]).is_file())

  def test_search_space_is_bounded(self):
    with self.assertRaisesRegex(ValueError, "maximum"):
      SearchPlan.from_mapping({"axes":{"A":list(range(20)), "B":list(range(20))}, "max_candidates":128})


if __name__ == "__main__": unittest.main()
