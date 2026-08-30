from __future__ import annotations

import threading, time, uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class JobSnapshot:
  job_id: str
  kind: str
  state: str
  session_id: str
  created_at_s: float
  started_at_s: float | None = None
  completed_at_s: float | None = None
  error: str | None = None
  error_type: str | None = None


class LocalJobManager:
  """Small in-process job manager for non-blocking local UI operations."""
  def __init__(self):
    self._jobs: dict[str, JobSnapshot] = {}
    self._lock = threading.RLock()

  def _set(self, snapshot: JobSnapshot) -> None:
    with self._lock: self._jobs[snapshot.job_id] = snapshot

  def submit(self, kind: str, session_id: str, fn: Callable[[], Any]) -> JobSnapshot:
    job_id, created = uuid.uuid4().hex, time.time()
    initial = JobSnapshot(job_id, kind, "queued", session_id, created)
    self._set(initial)

    def run():
      started = time.time()
      self._set(JobSnapshot(job_id, kind, "running", session_id, created, started_at_s=started))
      try:
        fn()
      except BaseException as exc:
        self._set(JobSnapshot(job_id, kind, "failed", session_id, created, started, time.time(), str(exc), type(exc).__name__))
      else:
        self._set(JobSnapshot(job_id, kind, "completed", session_id, created, started, time.time()))

    threading.Thread(target=run, name=f"radeon-forge-{kind}-{job_id[:8]}", daemon=True).start()
    return initial

  def snapshot(self, job_id: str) -> JobSnapshot:
    with self._lock:
      try: return self._jobs[job_id]
      except KeyError as exc: raise KeyError(f"unknown job {job_id}") from exc

  def snapshots(self, session_id: str | None = None) -> list[dict[str, Any]]:
    with self._lock:
      jobs = list(self._jobs.values())
    if session_id is not None: jobs = [job for job in jobs if job.session_id == session_id]
    return [asdict(job) for job in sorted(jobs, key=lambda x: x.created_at_s, reverse=True)]
