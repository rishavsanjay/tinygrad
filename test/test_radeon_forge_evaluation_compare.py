import unittest

from extra.radeon_forge.evaluation.compare import IncomparableSuites,compare_evaluations


def result(suite_hash="same",success=1.0,tools=1.0,p95=100.0,passed=(True,True),latencies=(80.0,120.0)):
  return {"suite_name":"private","suite_hash":suite_hash,"task_success_rate":success,"tool_call_validity_rate":tools,
    "p50_latency_ms":90.0,"p95_latency_ms":p95,"mean_latency_ms":100.0,
    "results":[
      {"task_id":"a","passed":passed[0],"latency_ms":latencies[0],"observed_tools":["read_file"],"failure_reasons":[]},
      {"task_id":"b","passed":passed[1],"latency_ms":latencies[1],"observed_tools":[],"failure_reasons":[]},
    ]}


class TestEvaluationComparison(unittest.TestCase):
  def test_faster_quality_preserving_candidate_is_deployable(self):
    baseline=result(p95=120.0)
    candidate=result(p95=80.0,latencies=(60.0,90.0))
    comparison=compare_evaluations(baseline,candidate)
    self.assertTrue(comparison["gate"]["passed"])
    self.assertTrue(comparison["deployable_speedup"])
    self.assertEqual(comparison["latency"]["p95_latency_ms"]["absolute"],-40.0)
    self.assertEqual(len(comparison["per_task"]),2)

  def test_speed_cannot_hide_task_or_tool_regression(self):
    baseline=result(p95=120.0)
    candidate=result(success=0.5,tools=0.5,p95=50.0,passed=(True,False),latencies=(30.0,40.0))
    comparison=compare_evaluations(baseline,candidate)
    self.assertFalse(comparison["gate"]["passed"])
    self.assertFalse(comparison["deployable_speedup"])
    self.assertIn("task success regressed",comparison["gate"]["reasons"][0])
    self.assertTrue(any("baseline-passing tasks regressed" in reason for reason in comparison["gate"]["reasons"]))

  def test_different_suite_or_task_ids_are_rejected(self):
    with self.assertRaisesRegex(IncomparableSuites,"suite_hash differs"):
      compare_evaluations(result("a"),result("b"))
    candidate=result();candidate["results"]=candidate["results"][:1]
    with self.assertRaisesRegex(IncomparableSuites,"task ids differ"):
      compare_evaluations(result(),candidate)

  def test_explicit_tolerance_is_visible_and_bounded(self):
    comparison=compare_evaluations(result(success=1.0),result(success=0.99),max_task_success_drop=0.02)
    self.assertTrue(comparison["gate"]["task_success_preserved"])


if __name__=="__main__":unittest.main()
