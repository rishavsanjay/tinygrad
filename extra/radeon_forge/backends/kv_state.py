from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class KVReuseLedger:
  """Tracks the only prefix that is safe to reuse from one resident KV cache.

  Generated output is not reusable until it has been fed back through the model
  as a decode input. This distinction matters at length stops and tool
  boundaries: emitted text and materialized KV state are not always identical.
  """
  active_session: str | None = None
  cached_tokens: list[int] = field(default_factory=list)

  def begin(self, session_id: str, prompt: Sequence[int]) -> int:
    if self.active_session != session_id:
      self.active_session = session_id
      self.cached_tokens.clear()
    common = 0
    for old, new in zip(self.cached_tokens, prompt):
      if old != new: break
      common += 1
    return common

  def commit_prefill(self, prompt_without_pending_decode_token: Sequence[int]) -> None:
    self.cached_tokens = list(prompt_without_pending_decode_token)

  def commit_decode_input(self, token: int, expected_position: int) -> None:
    if len(self.cached_tokens) != expected_position:
      raise RuntimeError(f"KV ledger position mismatch: cached={len(self.cached_tokens)} expected={expected_position}")
    self.cached_tokens.append(int(token))

  def reset(self) -> None:
    self.active_session = None
    self.cached_tokens.clear()
