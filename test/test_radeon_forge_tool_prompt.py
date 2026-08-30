import unittest

from extra.radeon_forge.runtime.tool_prompt import TOOL_PROTOCOL_MARKER, inject_tool_instruction, render_tool_instruction


class TestToolPrompt(unittest.TestCase):
  def test_tool_schema_is_rendered_for_local_model(self):
    tools = [{"type":"function", "function":{"name":"read_file", "description":"Read a file",
      "parameters":{"type":"object", "properties":{"path":{"type":"string"}}, "required":["path"]}}}]
    text = render_tool_instruction(tools)
    self.assertIn(TOOL_PROTOCOL_MARKER, text)
    self.assertIn('"name": "read_file"', text)
    self.assertIn('<tool_call>', text)

  def test_instruction_merges_into_existing_system_message_once(self):
    tools = [{"type":"function", "function":{"name":"read_file", "parameters":{"type":"object"}}}]
    original = [{"role":"system", "content":"Be precise."}, {"role":"user", "content":"Read it"}]
    injected = inject_tool_instruction(original, tools)
    self.assertEqual(len(injected), 2)
    self.assertTrue(injected[0]["content"].startswith("Be precise."))
    self.assertEqual(injected[0]["content"].count(TOOL_PROTOCOL_MARKER), 1)
    reinjected = inject_tool_instruction(injected, tools)
    self.assertEqual(reinjected[0]["content"].count(TOOL_PROTOCOL_MARKER), 1)
    self.assertEqual(original[0]["content"], "Be precise.")

  def test_instruction_is_prepended_when_no_system_message_exists(self):
    tools = [{"type":"function", "function":{"name":"search_text", "parameters":{"type":"object"}}}]
    injected = inject_tool_instruction([{"role":"user", "content":"Search"}], tools)
    self.assertEqual(injected[0]["role"], "system")
    self.assertIn(TOOL_PROTOCOL_MARKER, injected[0]["content"])


if __name__ == "__main__": unittest.main()
