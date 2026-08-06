import json
import tempfile
import unittest

from extra.radeon_forge.backends.fake import ScriptedBackend
from extra.radeon_forge.runtime import BackendCapabilities, ForgeEngine, GenerationEvent
from extra.radeon_forge.ui.openai_api import OpenAIRequestError, collect_chat_completion, stream_chat_completion


class ToolBackend:
  RAW = '<tool_call>{"id":"call-1","name":"read_file","arguments":{"path":"README.md"}}</tool_call>'
  @property
  def name(self): return "tool-model"
  @property
  def capabilities(self): return BackendCapabilities(structured_tools=True, local_only=True)
  @property
  def runtime_metadata(self): return {"architecture":"gfx1100", "runtime":"tinygrad", "model_family":"llama"}
  def stream(self, request):
    yield GenerationEvent("prefill", metrics={"prompt_tokens":12, "wall_ms":2.0})
    yield GenerationEvent("token", self.RAW[:20], metrics={"stage":"first_token", "wall_ms":1.0})
    yield GenerationEvent("token", self.RAW[20:], metrics={"stage":"decode", "wall_ms":1.0})
    yield GenerationEvent("tool_call", tool_call={"id":"call-1", "name":"read_file",
      "arguments":{"path":"README.md"}, "raw":self.RAW})
    yield GenerationEvent("done", finish_reason="tool_call", metrics={"generated_tokens":2})
  def close(self): pass


class TestOpenAIAdapter(unittest.TestCase):
  def test_regular_completion_has_standard_shape_and_forge_metrics(self):
    with tempfile.TemporaryDirectory() as directory:
      engine = ForgeEngine(ScriptedBackend(["hello local world"]), directory)
      result = collect_chat_completion(engine, {"model":"local", "messages":[{"role":"user", "content":"hello"}],
                                                "user":"stable-session"})
      self.assertEqual(result["object"], "chat.completion")
      self.assertEqual(result["choices"][0]["message"]["content"], "hello local world")
      self.assertEqual(result["choices"][0]["finish_reason"], "stop")
      self.assertEqual(result["forge"]["session_id"], "stable-session")
      self.assertGreater(result["usage"]["completion_tokens"], 0)
      engine.close()

  def test_tool_completion_is_structured_and_raw_protocol_is_not_content(self):
    with tempfile.TemporaryDirectory() as directory:
      engine = ForgeEngine(ToolBackend(), directory)
      payload = {"messages":[{"role":"user", "content":"read README"}], "tools":[{"type":"function",
        "function":{"name":"read_file", "description":"read", "parameters":{"type":"object"}}}]}
      result = collect_chat_completion(engine, payload)
      message = result["choices"][0]["message"]
      self.assertIsNone(message["content"])
      self.assertEqual(message["tool_calls"][0]["function"]["name"], "read_file")
      self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"path":"README.md"})
      self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
      engine.close()

  def test_stream_suppresses_internal_tool_markup(self):
    with tempfile.TemporaryDirectory() as directory:
      engine = ForgeEngine(ToolBackend(), directory)
      payload = {"stream":True, "messages":[{"role":"user", "content":"read README"}], "tools":[{"type":"function",
        "function":{"name":"read_file", "description":"read", "parameters":{"type":"object"}}}]}
      stream = list(stream_chat_completion(engine, payload))
      joined = "".join(stream)
      self.assertNotIn("<tool_call>", joined)
      self.assertIn('"tool_calls"', joined)
      self.assertIn('"finish_reason":"tool_calls"', joined)
      self.assertEqual(stream[-1], "data: [DONE]\n\n")
      engine.close()

  def test_invalid_messages_fail_before_model_execution(self):
    with tempfile.TemporaryDirectory() as directory:
      engine = ForgeEngine(ScriptedBackend(), directory)
      with self.assertRaises(OpenAIRequestError): collect_chat_completion(engine, {"messages":[]})
      engine.close()


if __name__ == "__main__": unittest.main()
