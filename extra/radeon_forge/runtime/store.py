from __future__ import annotations

import json, os, tempfile, time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from .backend import InferenceBackend
from .events import TraceEvent, TraceRecorder
from .session import AgentSession, SessionEvent, SessionState
from .tools import ToolCall, ToolRegistry


class SessionStore:
  """Atomic, local-only persistence for agent sessions and unified traces."""
  FORMAT_VERSION = 1

  def __init__(self, root: str | Path):
    self.root = Path(root).resolve()
    self.root.mkdir(parents=True, exist_ok=True)

  def _path(self, session_id: str) -> Path:
    if not session_id or any(ch not in "0123456789abcdef" for ch in session_id.lower()): raise ValueError("invalid session id")
    return self.root / f"{session_id}.json"

  @staticmethod
  def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
      with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
      os.replace(temporary, path)
    finally:
      try: os.unlink(temporary)
      except FileNotFoundError: pass

  def save(self, session: AgentSession) -> Path:
    snapshot = session.snapshot()
    payload = {"format_version":self.FORMAT_VERSION, "saved_at_s":time.time(), "session":snapshot,
               "trace":session.trace.to_dict()}
    path = self._path(session.session_id)
    self._atomic_json(path, payload)
    return path

  def delete(self, session_id: str) -> None:
    try: self._path(session_id).unlink()
    except FileNotFoundError: pass

  def list(self) -> list[dict[str, Any]]:
    ret=[]
    for path in sorted(self.root.glob("*.json"), key=lambda item:item.stat().st_mtime, reverse=True):
      try:
        payload=json.loads(path.read_text(encoding="utf-8")); session=payload["session"]
        ret.append({"session_id":session["session_id"], "state":session.get("state","unknown"),
                    "saved_at_s":payload.get("saved_at_s"), "message_count":len(session.get("messages",[])),
                    "event_count":len(session.get("events",[])), "path":str(path)})
      except Exception: continue
    return ret

  def load_payload(self, session_id: str) -> dict[str, Any]:
    path=self._path(session_id)
    payload=json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("format_version",0)) != self.FORMAT_VERSION: raise ValueError("unsupported session checkpoint version")
    return payload

  def restore(self, session_id: str, backend: InferenceBackend, tools: ToolRegistry, system_prompt: str,
              metadata_provider: Callable[[AgentSession, int], Mapping[str, Any]] | None = None) -> AgentSession:
    payload=self.load_payload(session_id); saved=payload["session"]
    trace_payload=payload.get("trace", {})
    trace=TraceRecorder(str(trace_payload.get("trace_id") or saved.get("trace_id") or ""))
    trace_events=[]
    for item in trace_payload.get("events", []):
      try:
        trace_events.append(TraceEvent(str(item["event_id"]), str(item["trace_id"]), str(item["kind"]), str(item["name"]),
          int(item["start_ns"]), int(item["end_ns"]) if item.get("end_ns") is not None else None,
          str(item["parent_id"]) if item.get("parent_id") is not None else None, dict(item.get("attributes", {}))))
      except Exception: continue
    trace.extend(trace_events)
    session=AgentSession(backend,tools,system_prompt,session_id=str(saved["session_id"]),trace=trace,metadata_provider=metadata_provider)
    session.messages=[dict(item) for item in saved.get("messages", [])]
    if not session.messages: session.messages=[{"role":"system","content":session._system_prompt()}]
    session.events=[]
    for item in saved.get("events", []):
      try: session.events.append(SessionEvent(int(item["sequence"]),str(item["kind"]),dict(item.get("data",{})),float(item.get("timestamp_s",time.time()))))
      except Exception: continue
    session._sequence=max((item.sequence for item in session.events),default=0)
    pending=saved.get("pending_tool_call")
    session.pending_tool_call=(ToolCall(str(pending["call_id"]),str(pending["name"]),dict(pending.get("arguments",{})))
                               if isinstance(pending,Mapping) else None)
    session.pending_assistant_content=str(saved.get("pending_assistant_content", ""))
    session.partial_output=""
    session.last_finish_reason=saved.get("last_finish_reason")
    prior_state=SessionState(str(saved.get("state",SessionState.IDLE.value)))
    if prior_state in {SessionState.GENERATING,SessionState.RUNNING_TOOL}:
      session.state=SessionState.IDLE
      session._emit("session_recovered", prior_state=prior_state.value, action="incomplete operation was not replayed")
    elif prior_state is SessionState.AWAITING_TOOL_APPROVAL and session.pending_tool_call is None:
      session.state=SessionState.IDLE
      session._emit("session_recovered", prior_state=prior_state.value, action="missing pending tool was cleared")
    else: session.state=prior_state
    return session
