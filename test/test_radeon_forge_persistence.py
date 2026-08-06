import tempfile
import unittest
from pathlib import Path

from extra.radeon_forge.backends.fake import ScriptedBackend
from extra.radeon_forge.runtime import ForgeEngine
from extra.radeon_forge.runtime.session import AgentSession, SessionState
from extra.radeon_forge.runtime.store import SessionStore
from extra.radeon_forge.runtime.tools import ToolRegistry
from extra.radeon_forge.permissions import PermissionController


class TestSessionPersistence(unittest.TestCase):
  def test_completed_session_restores_and_continues(self):
    with tempfile.TemporaryDirectory() as directory:
      first=ForgeEngine(ScriptedBackend(["first local answer"]),directory)
      session=first.create_session(); session_id=session.session_id
      first.run_message(session_id,"first question")
      trace_id=session.trace.trace_id
      first.close()

      second=ForgeEngine(ScriptedBackend(["continued after restart"]),directory)
      restored=second.session(session_id)
      self.assertEqual(restored.state,SessionState.COMPLETED)
      self.assertEqual(restored.trace.trace_id,trace_id)
      self.assertEqual(restored.messages[-1]["content"],"first local answer")
      second.run_message(session_id,"continue")
      self.assertEqual(restored.messages[-1]["content"],"continued after restart")
      self.assertGreater(len(restored.trace.events()),0)
      second.close()

  def test_pending_tool_approval_restores_without_execution(self):
    with tempfile.TemporaryDirectory() as directory:
      Path(directory,"hello.txt").write_text("private",encoding="utf-8")
      first=ForgeEngine(ScriptedBackend(['<tool_call>{"name":"read_file","arguments":{"path":"hello.txt"}}</tool_call>']),directory)
      session=first.create_session(); session_id=session.session_id
      first.run_message(session_id,"read the file")
      self.assertEqual(session.state,SessionState.AWAITING_TOOL_APPROVAL)
      first.close()

      second=ForgeEngine(ScriptedBackend(["read completed after approval"]),directory)
      restored=second.session(session_id)
      self.assertEqual(restored.state,SessionState.AWAITING_TOOL_APPROVAL)
      self.assertEqual(restored.pending_tool_call.name,"read_file")
      self.assertFalse(any(event.kind=="tool_started" for event in restored.events))
      second.run_tool_approval(session_id,"approve restored local read")
      self.assertEqual(restored.state,SessionState.COMPLETED)
      self.assertEqual(restored.messages[-1]["content"],"read completed after approval")
      second.close()

  def test_inflight_checkpoint_recovers_to_idle_without_replay(self):
    with tempfile.TemporaryDirectory() as directory:
      root=Path(directory)/"sessions"
      store=SessionStore(root)
      backend=ScriptedBackend(["unused"])
      session=AgentSession(backend,ToolRegistry(PermissionController()),"system")
      session.messages.append({"role":"user","content":"incomplete turn"})
      session.state=SessionState.GENERATING
      session._emit("generation_started",backend="fake",step=1,capabilities={})
      store.save(session)
      restored=store.restore(session.session_id,backend,ToolRegistry(PermissionController()),"system")
      self.assertEqual(restored.state,SessionState.IDLE)
      recovery=next(event for event in restored.events if event.kind=="session_recovered")
      self.assertEqual(recovery.data["prior_state"],"generating")
      self.assertFalse(any(message.get("role")=="assistant" for message in restored.messages))

  def test_checkpoint_path_and_format_are_local_and_atomic(self):
    with tempfile.TemporaryDirectory() as directory:
      store=SessionStore(directory)
      session=AgentSession(ScriptedBackend(),ToolRegistry(PermissionController()),"system")
      path=store.save(session)
      self.assertEqual(path.parent.resolve(),Path(directory).resolve())
      self.assertTrue(path.name.endswith(".json"))
      self.assertFalse(list(Path(directory).glob("*.tmp")))
      payload=store.load_payload(session.session_id)
      self.assertEqual(payload["format_version"],SessionStore.FORMAT_VERSION)


if __name__=="__main__": unittest.main()
