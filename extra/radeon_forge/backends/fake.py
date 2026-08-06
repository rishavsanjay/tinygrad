from __future__ import annotations

from typing import Iterator
from ..runtime.backend import BackendCapabilities, GenerationEvent, GenerationRequest


class ScriptedBackend:
  """Deterministic backend for UI development and runtime tests without a GPU."""
  def __init__(self, responses: list[str] | None = None): self.responses = responses or ["Radeon Forge is ready."]
  @property
  def name(self) -> str: return "scripted-local"
  @property
  def capabilities(self) -> BackendCapabilities: return BackendCapabilities(kernel_metrics=True)
  def stream(self, request: GenerationRequest) -> Iterator[GenerationEvent]:
    text = self.responses.pop(0) if self.responses else "Done."
    yield GenerationEvent("prefill", metrics={"wall_ms": 4.0, "prompt_tokens": 32, "prefix_reused_tokens": 24})
    parts = text.split(" ")
    for index, token in enumerate(parts):
      yield GenerationEvent("token", token + (" " if index < len(parts) - 1 else ""),
                            metrics={"index": index, "wall_ms": 2.0, "gpu_ms": 1.4, "kernel_count": 24})
    yield GenerationEvent("done", finish_reason="stop", metrics={"generated_tokens": len(parts)})
  def close(self) -> None: pass
