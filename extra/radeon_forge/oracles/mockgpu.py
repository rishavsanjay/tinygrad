from __future__ import annotations

import json, os, shlex, subprocess, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MockGPUResult:
  passed: bool
  returncode: int
  elapsed_ms: float
  command: tuple[str, ...]
  stdout: str
  stderr: str
  target: str = "gfx1100"
  backend: str = "DEV=MOCK+AMD"

  def to_dict(self): return asdict(self)


class MockGPUOracle:
  """Runs candidate correctness checks through tinygrad's integrated RDNA3 mock GPU.

  This is a semantic/instruction support gate, never a performance oracle.
  Real W7900 execution remains authoritative for latency and resource behavior.
  """
  def __init__(self, root: str | Path, timeout_seconds: int = 600):
    self.root = Path(root).resolve()
    self.timeout_seconds = timeout_seconds

  def run(self, command: Sequence[str] | str, env: Mapping[str, str] | None = None) -> MockGPUResult:
    argv = tuple(shlex.split(command) if isinstance(command, str) else (str(x) for x in command))
    if not argv: raise ValueError("command must not be empty")
    child_env = os.environ.copy()
    child_env.update({"DEV": "MOCK+AMD", "PYTHONUNBUFFERED": "1"})
    child_env.update(env or {})
    started = time.perf_counter_ns()
    proc = subprocess.run(argv, cwd=self.root, env=child_env, text=True, capture_output=True, timeout=self.timeout_seconds)
    return MockGPUResult(proc.returncode == 0, proc.returncode, (time.perf_counter_ns() - started) / 1e6, argv,
                         proc.stdout[-50000:], proc.stderr[-50000:])

  def run_json(self, command: Sequence[str] | str, env: Mapping[str, str] | None = None) -> str:
    return json.dumps(self.run(command, env).to_dict(), default=str)
