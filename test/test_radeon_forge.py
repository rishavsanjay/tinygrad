import tempfile
import unittest
from pathlib import Path

from extra.radeon_forge.amd_metadata import resource_report_from_comgr, resource_report_from_text
from extra.radeon_forge.contracts import Candidate, CorrectnessReport, ResourceReport, TrialResult, WorkloadContract
from extra.radeon_forge.families import rdna3_rmsnorm_fp8_family
from extra.radeon_forge.ledger import ExperimentLedger
from extra.radeon_forge.tuner import successive_halving


class TestRadeonForge(unittest.TestCase):
  def test_metadata_text(self):
    text = """
; NumSgprs: 18
; NumVgprs: 54
; ScratchSize: 0
; LDSByteSize: 8192 bytes/workgroup
; Occupancy: 8
    .vgpr_spill_count: 0
    .sgpr_spill_count: 1
"""
    report = resource_report_from_text(text)
    self.assertEqual((report.sgprs, report.vgprs, report.lds_bytes, report.occupancy), (18, 54, 8192, 8))
    self.assertTrue(report.has_spills)

  def test_metadata_comgr(self):
    report = resource_report_from_comgr({"amdhsa.kernels": [{".vgpr_count": "44", ".sgpr_count": "20",
      ".group_segment_fixed_size": "4096", ".private_segment_fixed_size": "0", ".vgpr_spill_count": "0", ".sgpr_spill_count": "0"}]})
    self.assertEqual((report.vgprs, report.sgprs, report.lds_bytes), (44, 20, 4096))
    self.assertFalse(report.has_spills)

  def test_family_grid(self):
    family = rdna3_rmsnorm_fp8_family("/tmp/tinygrad", 4096 * 16, 4096)
    candidates = family.candidates()
    self.assertEqual(len(candidates), 12)
    self.assertEqual(len({candidate.candidate_id for candidate in candidates}), 12)
    self.assertTrue(all(candidate.parameters["HIDDEN"] == 4096 for candidate in candidates))

  def test_hard_gate_beats_fast_invalid_candidate(self):
    candidates = [Candidate("bad", "test", {}), Candidate("good", "test", {})]
    contract = WorkloadContract("unit", max_vgprs=96)

    def evaluate(candidate, budget):
      if candidate.candidate_id == "bad":
        return TrialResult(candidate, CorrectnessReport(True), samples_us=(1.0,) * budget, median_latency_us=1.0, p95_latency_us=1.0,
                           resources=ResourceReport(vgprs=128))
      return TrialResult(candidate, CorrectnessReport(True), samples_us=(2.0,) * budget, median_latency_us=2.0, p95_latency_us=2.0,
                         resources=ResourceReport(vgprs=64))

    with tempfile.TemporaryDirectory() as directory:
      ledger = ExperimentLedger(Path(directory) / "events.jsonl")
      summary = successive_halving(candidates, evaluate, contract, budgets=(1, 2), reduction=2, ledger=ledger)
      self.assertIsNotNone(summary.winner)
      self.assertEqual(summary.winner.candidate.candidate_id, "good")
      self.assertTrue(any(record["event"] == "winner_selected" for record in ledger.records()))


if __name__ == "__main__": unittest.main()
