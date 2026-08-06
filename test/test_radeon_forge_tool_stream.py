import unittest

from extra.radeon_forge.runtime.tool_stream import ToolProtocolError, ToolStreamParser


class TestToolStreamParser(unittest.TestCase):
  def test_incremental_valid_tool_call(self):
    parser = ToolStreamParser(["read_file", "search_text"])
    self.assertIsNone(parser.feed('<tool_call>{"name":"read_'))
    call = parser.feed('file","arguments":{"path":"README.md"}}</tool_call>')
    self.assertEqual(call.name, "read_file")
    self.assertEqual(call.arguments, {"path":"README.md"})
    self.assertEqual(call.raw, '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>')

  def test_unknown_tool_fails_closed(self):
    parser = ToolStreamParser(["read_file"])
    with self.assertRaisesRegex(ToolProtocolError, "unknown or unavailable"):
      parser.feed('<tool_call>{"name":"curl","arguments":{}}</tool_call>')

  def test_malformed_or_incomplete_calls_fail_closed(self):
    parser = ToolStreamParser(["read_file"])
    with self.assertRaisesRegex(ToolProtocolError, "exactly one"):
      parser.feed('Sure. <tool_call>{"name":"read_file","arguments":{}}</tool_call>')
    incomplete = ToolStreamParser(["read_file"])
    incomplete.feed('<tool_call>{"name":"read_file"')
    with self.assertRaisesRegex(ToolProtocolError, "incomplete"):
      incomplete.finalize()

  def test_stream_size_is_bounded(self):
    parser = ToolStreamParser(["read_file"], max_bytes=8)
    with self.assertRaisesRegex(ToolProtocolError, "size limit"):
      parser.feed("123456789")


if __name__ == "__main__": unittest.main()
