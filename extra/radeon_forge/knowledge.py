from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_TOKEN = re.compile(r"[A-Za-z0-9_+#.-]+")


@dataclass(frozen=True)
class KnowledgeChunk:
  source: str
  start_line: int
  end_line: int
  text: str


@dataclass(frozen=True)
class RetrievalHit:
  chunk: KnowledgeChunk
  score: float

  @property
  def citation(self) -> str:
    return f"{self.chunk.source}:{self.chunk.start_line}-{self.chunk.end_line}"


class LocalKnowledgeBase:
  """Small deterministic local retriever for architecture and experiment knowledge.

  This deliberately has no network path. It is not intended to replace a full
  embedding index; its job is to provide auditable local excerpts and citations
  to the planning model for the supported project corpus.
  """

  def __init__(self, chunks: Sequence[KnowledgeChunk]):
    self.chunks = tuple(chunks)
    self._term_counts = [Counter(self._tokens(chunk.text)) for chunk in self.chunks]
    document_frequency: Counter[str] = Counter()
    for counts in self._term_counts: document_frequency.update(counts.keys())
    self._idf = {term: math.log((1 + len(self.chunks)) / (1 + frequency)) + 1.0 for term, frequency in document_frequency.items()}

  @staticmethod
  def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]

  @classmethod
  def from_paths(cls, paths: Iterable[str | Path], lines_per_chunk: int = 40, overlap: int = 5) -> "LocalKnowledgeBase":
    if lines_per_chunk <= 0 or overlap < 0 or overlap >= lines_per_chunk: raise ValueError("invalid chunk dimensions")
    chunks: list[KnowledgeChunk] = []
    step = lines_per_chunk - overlap
    for path_value in paths:
      path = Path(path_value)
      lines = path.read_text(encoding="utf-8").splitlines()
      for start in range(0, len(lines), step):
        selected = lines[start:start + lines_per_chunk]
        if not selected: continue
        chunks.append(KnowledgeChunk(str(path), start + 1, start + len(selected), "\n".join(selected)))
    return cls(chunks)

  def search(self, query: str, top_k: int = 5) -> tuple[RetrievalHit, ...]:
    if top_k <= 0: raise ValueError("top_k must be positive")
    query_counts = Counter(self._tokens(query))
    if not query_counts: return ()
    hits: list[RetrievalHit] = []
    for chunk, counts in zip(self.chunks, self._term_counts):
      length_norm = max(1.0, math.sqrt(sum(value * value for value in counts.values())))
      score = sum(query_count * counts.get(term, 0) * self._idf.get(term, 1.0) for term, query_count in query_counts.items()) / length_norm
      if score > 0: hits.append(RetrievalHit(chunk, score))
    hits.sort(key=lambda hit: (-hit.score, hit.chunk.source, hit.chunk.start_line))
    return tuple(hits[:top_k])
