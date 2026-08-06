from __future__ import annotations

import signal, threading


class SignalCancellation:
  """Process-local cooperative cancellation set by SIGUSR1."""
  def __init__(self):
    self._event = threading.Event()
    self.installed = False

  def install(self) -> bool:
    if not hasattr(signal, "SIGUSR1"): return False
    signal.signal(signal.SIGUSR1, lambda _signum, _frame: self._event.set())
    self.installed = True
    return True

  def reset(self) -> None: self._event.clear()
  def request(self) -> None: self._event.set()
  @property
  def requested(self) -> bool: return self._event.is_set()
