import unittest

from extra.radeon_forge.profiling.report import build_profile_report
from extra.radeon_forge.runtime.events import TraceRecorder


class TestHardwareProfileEvidence(unittest.TestCase):
  def test_att_and_counter_captures_are_observed_evidence(self):
    trace=TraceRecorder("trace-1")
    trace.point("profile","hardware_capture",capture_id="att-1",capture_kind="att",passed=True,stage="decode",
                artifact_count=3,output_directory="/tmp/att",parsed_csv={},error="")
    trace.point("profile","hardware_capture",capture_id="counter-1",capture_kind="counters",passed=True,stage="first_token",
                artifact_count=2,output_directory="/tmp/counters",parsed_csv={"csv_rows":8},error="")
    report=build_profile_report(trace.events())
    self.assertEqual(report["summary"]["hardware_capture_count"],2)
    self.assertEqual(report["summary"]["att_capture_count"],1)
    self.assertEqual(report["summary"]["counter_capture_count"],1)
    titles={finding["title"] for finding in report["findings"]}
    self.assertIn("AMD ATT/SQTT evidence is attached",titles)
    self.assertIn("ROCm hardware counter evidence is attached",titles)
    self.assertNotIn("No ROCm counter or ATT capture is attached to this trace",titles)

  def test_failed_capture_is_visible(self):
    trace=TraceRecorder("trace-2")
    trace.point("profile","hardware_capture",capture_id="att-bad",capture_kind="att",passed=False,stage="decode",
                artifact_count=0,output_directory="/tmp/att-bad",parsed_csv={},error="rocprofv3 missing ATT support")
    report=build_profile_report(trace.events())
    self.assertEqual(report["summary"]["failed_capture_count"],1)
    finding=next(item for item in report["findings"] if item["title"]=="One or more hardware profiling captures failed")
    self.assertEqual(finding["status"],"observed")
    self.assertIn("missing ATT support",finding["evidence"][0])

  def test_absence_remains_unknown(self):
    report=build_profile_report(())
    finding=next(item for item in report["findings"] if item["title"]=="No ROCm counter or ATT capture is attached to this trace")
    self.assertEqual(finding["status"],"unknown")


if __name__=="__main__": unittest.main()
