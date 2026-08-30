from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class ExperimentLedger:
  """Append-only JSONL evidence store.

  Generated implementations are disposable; this ledger preserves hypotheses,
  contracts, measurements, rejections, and approvals needed to reproduce why a
  candidate was selected.
  """

  def __init__(self, path: str | os.PathLike[str]):
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)

  @staticmethod
  def _jsonable(value: Any) -> Any:
    if is_dataclass(value): return asdict(value)
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): ExperimentLedger._jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [ExperimentLedger._jsonable(v) for v in value]
    return value

  def append(self, event: str, payload: Any) -> None:
    record = {
      "schema_version": 1,
      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "event": event,
      "payload": self._jsonable(payload),
    }
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
      os.write(fd, line.encode("utf-8"))
      os.fsync(fd)
    finally:
      os.close(fd)

  def records(self) -> Iterable[dict[str, Any]]:
    if not self.path.exists(): return ()
    with self.path.open("r", encoding="utf-8") as f:
      return tuple(json.loads(line) for line in f if line.strip())
