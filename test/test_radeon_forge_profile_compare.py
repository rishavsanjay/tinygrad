import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from extra.radeon_forge.profiling.compare import IncomparableCaptures, compare_captures


class TestProfileCaptureComparison(unittest.TestCase):
  def _capture(self, root: Path, capture_id: str, stage: str, rows: list[dict[str, str]], *, command=None,
               counters=("SQ_WAVES", "DurationNs"), passed=True, workload_hash="suite-1") -> Path:
    directory = root / capture_id
    directory.mkdir()
    csv_path = directory / "counters.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=["Kernel_Name", "SQ_WAVES", "DurationNs"])
      writer.writeheader(); writer.writerows(rows)
    data = csv_path.read_bytes()
    numeric = {}
    for key in ("SQ_WAVES", "DurationNs"):
      values = [float(row[key]) for row in rows]
      numeric[key] = {"count":len(values), "min":min(values), "max":max(values), "mean":sum(values)/len(values)}
    payload = {
      "capture_id":capture_id, "kind":"counters", "passed":passed,
      "target_command":command or ["python3", "workload.py", "--case", "decode"],
      "artifacts":[{"path":str(csv_path), "size_bytes":len(data), "sha256":hashlib.sha256(data).hexdigest(), "suffix":".csv"}],
      "summary":{"stage":stage, "counters":list(counters), "capture_metadata":{"workload_hash":workload_hash},
                 "parsed_csv":{"csv_files":1, "csv_rows":len(rows), "numeric_columns":numeric}}
    }
    manifest = directory / "forge_capture_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest

  def test_compatible_captures_produce_global_and_per_kernel_deltas(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)
      baseline=self._capture(root,"base","decode",[
        {"Kernel_Name":"decode_gemv","SQ_WAVES":"64","DurationNs":"1000"},
        {"Kernel_Name":"decode_gemv","SQ_WAVES":"64","DurationNs":"1200"},
      ])
      candidate=self._capture(root,"candidate","decode",[
        {"Kernel_Name":"decode_gemv","SQ_WAVES":"72","DurationNs":"800"},
        {"Kernel_Name":"decode_gemv","SQ_WAVES":"72","DurationNs":"900"},
      ])
      result=compare_captures(baseline,candidate)
      self.assertTrue(result["compatibility"]["comparable"])
      self.assertEqual(result["numeric_column_deltas"]["DurationNs"]["baseline"],1100.0)
      self.assertEqual(result["numeric_column_deltas"]["DurationNs"]["candidate"],850.0)
      row=next(item for item in result["per_kernel_deltas"] if item["kernel"]=="decode_gemv" and item["metric"]=="DurationNs")
      self.assertEqual(row["absolute"],-250.0)
      self.assertAlmostEqual(row["relative"],-250/1100)
      self.assertEqual(result["interpretation_contract"]["lower_is_better_metrics"],[])

  def test_stage_or_workload_mismatch_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)
      rows=[{"Kernel_Name":"k","SQ_WAVES":"1","DurationNs":"1"}]
      baseline=self._capture(root,"base","prefill",rows)
      candidate=self._capture(root,"candidate","decode",rows)
      with self.assertRaisesRegex(IncomparableCaptures,"stage differs"):
        compare_captures(baseline,candidate)
      other=self._capture(root,"other","prefill",rows,workload_hash="suite-2")
      with self.assertRaisesRegex(IncomparableCaptures,"workload_identity differs"):
        compare_captures(baseline,other)

  def test_counter_set_or_failed_capture_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)
      rows=[{"Kernel_Name":"k","SQ_WAVES":"1","DurationNs":"1"}]
      baseline=self._capture(root,"base","decode",rows)
      different=self._capture(root,"different","decode",rows,counters=("SQ_WAVES",))
      with self.assertRaisesRegex(IncomparableCaptures,"counters differs"):
        compare_captures(baseline,different)
      failed=self._capture(root,"failed","decode",rows,passed=False)
      with self.assertRaisesRegex(IncomparableCaptures,"candidate capture did not pass"):
        compare_captures(baseline,failed)


if __name__=="__main__": unittest.main()
