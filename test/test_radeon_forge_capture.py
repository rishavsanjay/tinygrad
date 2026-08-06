import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extra.radeon_forge.profiling.capture import CaptureKind, CaptureRequest, Rocprofv3Adapter
from extra.radeon_forge.profiling.tools import ProfilingTools
from extra.radeon_forge.runtime.tool_context import ToolExecutionContext, bind_tool_context


FAKE_ROCPROF = '''#!/usr/bin/env python3
import csv, pathlib, sys
if "--version" in sys.argv:
  print("rocprofv3 fake 1.0")
  raise SystemExit(0)
if "--help" in sys.argv:
  print("--att --att-simd-select --kernel-include-regex --pmc --output-directory -d")
  raise SystemExit(0)
args=sys.argv[1:]
flag="--output-directory" if "--output-directory" in args else "-d"
out=pathlib.Path(args[args.index(flag)+1])
out.mkdir(parents=True, exist_ok=True)
if "--att" in args:
  (out/"thread_trace.att").write_bytes(b"ATT-EVIDENCE")
else:
  with (out/"counters.csv").open("w", newline="") as f:
    writer=csv.DictWriter(f, fieldnames=["Kernel_Name","SQ_WAVES","DurationNs"])
    writer.writeheader()
    writer.writerow({"Kernel_Name":"decode_gemv","SQ_WAVES":"64","DurationNs":"900"})
print("capture complete")
'''


class TestRocprofCapture(unittest.TestCase):
  def _adapter(self, root: Path):
    binary=root/"bin"/"rocprofv3"
    binary.parent.mkdir()
    binary.write_text(FAKE_ROCPROF, encoding="utf-8")
    binary.chmod(0o755)
    environment={**os.environ, "PATH":str(binary.parent)+os.pathsep+os.environ.get("PATH","")}
    return Rocprofv3Adapter(root, root/"evidence"), environment

  def test_counter_capture_writes_manifest_hashes_and_summary(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)
      adapter, environment=self._adapter(root)
      with patch.dict(os.environ, environment, clear=True):
        probe=adapter.probe()
        self.assertTrue(probe["available"])
        self.assertTrue(probe["supports_pmc"])
        result=adapter.capture(CaptureRequest(CaptureKind.COUNTERS, ("python3","-c","print('x')"), "decode",
          session_id="s1", trace_id="t1", agent_step=2, counters=("SQ_WAVES","DurationNs")))
      self.assertTrue(result.passed)
      self.assertEqual(result.summary["stage"], "decode")
      self.assertEqual(result.summary["session_id"], "s1")
      self.assertEqual(result.summary["parsed_csv"]["csv_rows"], 1)
      self.assertEqual(result.summary["parsed_csv"]["numeric_columns"]["SQ_WAVES"]["mean"], 64.0)
      self.assertTrue(all(len(artifact.sha256)==64 for artifact in result.artifacts))
      self.assertTrue((Path(result.output_directory)/"forge_capture_manifest.json").is_file())

  def test_att_capture_is_separate_and_provenance_is_bound(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)
      adapter, environment=self._adapter(root)
      tools=ProfilingTools(root, root/"evidence")
      context=ToolExecutionContext("session-7","trace-9",3,"call-2","capture_rocm_att",{"purpose":"decode diagnosis"})
      with patch.dict(os.environ, environment, clear=True), bind_tool_context(context):
        result=tools.capture_att({"command":["python3","-c","print('decode')"], "stage":"first_token",
                                  "kernel_regex":"decode_.*", "timeout_seconds":30})
      self.assertTrue(result["passed"])
      self.assertEqual(result["summary"]["session_id"], "session-7")
      self.assertEqual(result["summary"]["trace_id"], "trace-9")
      self.assertEqual(result["summary"]["agent_step"], 3)
      self.assertEqual(result["summary"]["kernel_regex"], "decode_.*")
      self.assertTrue(any(item["suffix"]==".att" for item in result["artifacts"]))
      listed=tools.list_captures({"limit":10})
      self.assertEqual(listed["captures"][0]["capture_id"], result["capture_id"])

  def test_profile_target_is_allowlisted_and_shell_free(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)
      adapter, environment=self._adapter(root)
      with patch.dict(os.environ, environment, clear=True):
        with self.assertRaisesRegex(ValueError, "not allowlisted"):
          adapter.capture(CaptureRequest(CaptureKind.ATT, ("bash","-lc","rm -rf /"), "decode"))


if __name__ == "__main__": unittest.main()
