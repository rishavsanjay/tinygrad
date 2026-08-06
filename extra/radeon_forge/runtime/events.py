from __future__ import annotations

import contextlib, json, threading, time, uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping


def now_ns() -> int: return time.perf_counter_ns()


@dataclass(frozen=True)
class TraceEvent:
  event_id: str
  trace_id: str
  kind: str
  name: str
  start_ns: int
  end_ns: int | None = None
  parent_id: str | None = None
  attributes: Mapping[str, Any] = field(default_factory=dict)

  @property
  def duration_ms(self) -> float | None:
    return None if self.end_ns is None else (self.end_ns - self.start_ns) / 1e6

  def to_dict(self) -> dict[str, Any]:
    ret = asdict(self)
    ret["duration_ms"] = self.duration_ms
    return ret


class TraceRecorder:
  """Thread-safe hierarchical trace recorder for agent, model, tool and GPU events."""
  def __init__(self, trace_id: str | None = None):
    self.trace_id = trace_id or uuid.uuid4().hex
    self._events: list[TraceEvent] = []
    self._lock = threading.RLock()

  def point(self, kind: str, event_name: str, parent_id: str | None = None, **attributes: Any) -> TraceEvent:
    """Record an instantaneous event.

    `event_name` is deliberately not called `name`: backend evidence commonly
    carries a `name` attribute for the concrete GPU kernel. Keeping those two
    namespaces separate lets an event be named `kernel` while retaining the
    real kernel symbol in its attributes.
    """
    ev = TraceEvent(uuid.uuid4().hex, self.trace_id, kind, event_name, now_ns(), now_ns(), parent_id, attributes)
    with self._lock: self._events.append(ev)
    return ev

  def duration(self, kind: str, event_name: str, duration_ms: float, parent_id: str | None = None, **attributes: Any) -> TraceEvent:
    """Record a duration measured by another clock domain.

    GPU profile timestamps are device-local and cannot be placed exactly on the
    CPU wall-clock axis without calibration. We preserve their measured duration
    and causal parent while anchoring the range at ingestion time. The attributes
    retain source/stage/order for later calibrated timeline adapters.
    """
    duration = max(0.0, float(duration_ms))
    end = now_ns()
    start = end - int(duration * 1e6)
    ev = TraceEvent(uuid.uuid4().hex, self.trace_id, kind, event_name, start, end, parent_id, attributes)
    with self._lock: self._events.append(ev)
    return ev

  @contextlib.contextmanager
  def span(self, kind: str, event_name: str, parent_id: str | None = None, **attributes: Any) -> Iterator[TraceEvent]:
    start = now_ns()
    provisional = TraceEvent(uuid.uuid4().hex, self.trace_id, kind, event_name, start, None, parent_id, attributes)
    try: yield provisional
    except BaseException as exc:
      attrs = dict(attributes)
      attrs.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
      with self._lock: self._events.append(TraceEvent(provisional.event_id, self.trace_id, kind, event_name, start, now_ns(), parent_id, attrs))
      raise
    else:
      attrs = dict(attributes)
      attrs.setdefault("status", "ok")
      with self._lock: self._events.append(TraceEvent(provisional.event_id, self.trace_id, kind, event_name, start, now_ns(), parent_id, attrs))

  def extend(self, events: list[TraceEvent]) -> None:
    with self._lock: self._events.extend(events)

  def events(self) -> tuple[TraceEvent, ...]:
    with self._lock: return tuple(sorted(self._events, key=lambda x: (x.start_ns, x.event_id)))

  def to_dict(self) -> dict[str, Any]: return {"trace_id": self.trace_id, "events": [x.to_dict() for x in self.events()]}

  def write_json(self, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
    return out

  def chrome_trace(self) -> dict[str, Any]:
    events = []
    for ev in self.events():
      if ev.end_ns is None: continue
      events.append({"name": ev.name, "cat": ev.kind, "ph": "X", "ts": ev.start_ns / 1000,
                     "dur": (ev.end_ns - ev.start_ns) / 1000, "pid": 1, "tid": ev.kind,
                     "args": dict(ev.attributes)})
    return {"traceEvents": events, "displayTimeUnit": "ms"}