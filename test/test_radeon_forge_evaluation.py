import json
import tempfile
import unittest
from pathlib import Path

from extra.radeon_forge.backends.fake import ScriptedBackend
from extra.radeon_forge.evaluation.suite import AgentTask, AgentTaskSuite, run_suite
from extra.radeon_forge.runtime import ForgeEngine


class TestAgentEvaluation(unittest.TestCase):
  def test_suite_measures_complete_tool_loop(self):
    responses=[
      '<tool_call>{"name":"read_file","arguments":{"path":"marker.txt"}}</tool_call>',
      'The marker is ALPHA.',
      'MockGPU is a semantic emulator; only real W7900 hardware provides performance evidence.',
    ]
    with tempfile.TemporaryDirectory() as directory:
      Path(directory,"marker.txt").write_text("ALPHA",encoding="utf-8")
      suite=AgentTaskSuite("unit",(
        AgentTask("read","Read marker.txt and report the marker.",expected_tools=("read_file",),allowed_tools=("read_file",),
                  required_output_regex=("ALPHA",),timeout_seconds=10),
        AgentTask("explain","Why not use MockGPU timing?",required_output_regex=("real W7900|real.*hardware",),timeout_seconds=10),
      ))
      engine=ForgeEngine(ScriptedBackend(responses),directory)
      result=run_suite(engine,suite)
      self.assertEqual(result.passed_tasks,2)
      self.assertEqual(result.task_success_rate,1.0)
      self.assertEqual(result.tool_call_validity_rate,1.0)
      self.assertGreater(result.p50_latency_ms,0)
      self.assertEqual(result.results[0].observed_tools,("read_file",))
      self.assertTrue(result.results[0].trace_id)
      engine.close()

  def test_unexpected_tool_is_rejected_and_task_fails(self):
    with tempfile.TemporaryDirectory() as directory:
      suite=AgentTaskSuite("unit",(AgentTask("bad","Answer directly",expected_tools=(),allowed_tools=()),))
      backend=ScriptedBackend(['<tool_call>{"name":"list_files","arguments":{}}</tool_call>'])
      engine=ForgeEngine(backend,directory)
      result=run_suite(engine,suite)
      task=result.results[0]
      self.assertFalse(task.passed)
      self.assertFalse(task.tool_call_valid)
      self.assertIn("not allowed",task.failure_reasons[0])
      self.assertEqual(task.terminal_state,"idle")
      engine.close()

  def test_suite_hash_is_stable_and_file_load_validates_ids(self):
    suite=AgentTaskSuite("stable",(AgentTask("a","prompt"),),metadata={"x":1})
    self.assertEqual(suite.suite_hash,suite.suite_hash)
    with tempfile.TemporaryDirectory() as directory:
      path=Path(directory)/"suite.json"
      payload={"format_version":1,"name":"x","tasks":[{"task_id":"a","prompt":"one"},{"task_id":"a","prompt":"two"}]}
      path.write_text(json.dumps(payload),encoding="utf-8")
      with self.assertRaisesRegex(ValueError,"unique"):
        AgentTaskSuite.load(path)


if __name__=="__main__":unittest.main()
