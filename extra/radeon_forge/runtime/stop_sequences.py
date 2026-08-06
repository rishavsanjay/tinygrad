from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StopMatch:
  text: str
  matched: str | None = None


class StopSequenceMatcher:
  """Incrementally hides stop strings even when they cross token boundaries."""
  def __init__(self, stops: Iterable[str]):
    values = tuple(dict.fromkeys(str(stop) for stop in stops))
    if any(not stop for stop in values): raise ValueError("stop sequences must not be empty")
    self.stops = values
    self.buffer = ""
    self.done = False

  def feed(self, text: str) -> StopMatch:
    if self.done: raise RuntimeError("stop matcher is already complete")
    self.buffer += text
    earliest: tuple[int, str] | None = None
    for stop in self.stops:
      index = self.buffer.find(stop)
      if index >= 0 and (earliest is None or index < earliest[0] or (index == earliest[0] and len(stop) > len(earliest[1]))):
        earliest = (index, stop)
    if earliest is not None:
      index, stop = earliest
      output = self.buffer[:index]
      self.buffer = ""
      self.done = True
      return StopMatch(output, stop)

    keep = 0
    for stop in self.stops:
      limit = min(len(stop) - 1, len(self.buffer))
      for length in range(limit, 0, -1):
        if self.buffer.endswith(stop[:length]):
          keep = max(keep, length)
          break
    if keep:
      output, self.buffer = self.buffer[:-keep], self.buffer[-keep:]
    else:
      output, self.buffer = self.buffer, ""
    return StopMatch(output)

  def finalize(self) -> str:
    if self.done: return ""
    output, self.buffer = self.buffer, ""
    self.done = True
    return output
