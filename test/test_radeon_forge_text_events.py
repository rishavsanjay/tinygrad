import tempfile
import unittest

from extra.radeon_forge.profiling.report import build_profile_report
from extra.radeon_forge.runtime import BackendCapabilities, ForgeEngine, GenerationEvent
from extra.radeon_forge.ui.openai_api import collect_chat_completion, stream_chat_completion


class BufferedTextBackend:
  @property
  def name(self): return "buffered-text-local"

  @property
  def capabilities(self): return BackendCapabilities(streaming=True, local_only=True)

  @property
  def runtime_metadata(self): return {"architecture":"gfx1100", "runtime":"tinygrad", "model_family":"llama"}

  def stream(self, request):
    yield GenerationEvent("prefill", metrics={"prompt_tokens":8, "wall_ms":1.0})
    # A real model token was generated, but its characters were withheld while
    # Forge determined whether they were the prefix of a cross-token stop.
    yield GenerationEvent("token", "", metrics={"index":0, "stage":"first_token", "wall_ms":2.0,
                                                  "gpu_ms":1.5, "kernel_count":20})
    yield GenerationEvent("text", "visible answer", metrics={"stage":"first_token", "buffered_for_stop_sequence":True})
    yield GenerationEvent("done", finish_reason="stop", metrics={"generated_tokens":1, "stop_sequence":"<END>"})

  def close(self): pass


class TestBufferedTextTransport(unittest.TestCase):
  def test_agent_session_preserves_text_without_fake_token(self):
    with tempfile.TemporaryDirectory() as directory:
      engine = ForgeEngine(BufferedTextBackend(), directory)
      session = engine.create_session()
      engine.run_message(session.session_id, "answer until the stop marker")
      self.assertEqual(session.messages[-1]["content"], "visible answer")
      report = build_profile_report(session.trace.events())
      self.assertEqual(report["summary"]["decode_tokens"], 1)
      self.assertEqual(report["summary"]["token_wall_ms_p50"], 2.0)
      self.assertEqual(len([event for event in session.events if event.kind == "text"]), 1)
      engine.close()

  def test_openai_completion_and_stream_include_visible_text(self):
    payload = {"messages":[{"role":"user", "content":"answer"}], "stop":["<END>"]}
    with tempfile.TemporaryDirectory() as directory:
      engine = ForgeEngine(BufferedTextBackend(), directory)
      result = collect_chat_completion(engine, payload)
      self.assertEqual(result["choices"][0]["message"]["content"], "visible answer")
      self.assertEqual(result["usage"]["completion_tokens"], 1)
      stream = "".join(stream_chat_completion(engine, {**payload, "session_id":"stream-session"}))
      self.assertIn("visible answer", stream)
      self.assertIn('"finish_reason":"stop"', stream)
      engine.close()


if __name__ == "__main__": unittest.main()
