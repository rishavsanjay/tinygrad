from __future__ import annotations

import hashlib, json, time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


def _hash(payload: Any) -> str:
  return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class KernelSpec:
  name: str
  operation: str
  target: str = "gfx1100"
  language: str = "python"
  extension: str = ".py"
  entrypoint: str = "build_kernel"
  shapes: Mapping[str, int | str] = field(default_factory=dict)
  dtypes: Mapping[str, str] = field(default_factory=dict)
  invariants: tuple[str, ...] = ()
  objective: str = "minimize_p95_latency"
  mockgpu_command: tuple[str, ...] = ()
  hardware_command: tuple[str, ...] = ()
  metadata: Mapping[str, Any] = field(default_factory=dict)

  @property
  def spec_id(self) -> str: return f"{self.name}-{_hash(asdict(self))[:12]}"


@dataclass(frozen=True)
class CandidateRecord:
  candidate_id: str
  spec_id: str
  source_path: str
  source_sha256: str
  parent_id: str | None
  hypothesis: str
  status: str
  created_at_s: float
  evidence: Mapping[str, Any] = field(default_factory=dict)


class CandidateWorkspace:
  """Content-addressed store where generated implementations are disposable artifacts."""
  def __init__(self, root: str | Path):
    self.root = Path(root).resolve()
    self.spec_root, self.candidate_root = self.root / "specs", self.root / "candidates"
    self.spec_root.mkdir(parents=True, exist_ok=True)
    self.candidate_root.mkdir(parents=True, exist_ok=True)

  def save_spec(self, spec: KernelSpec) -> KernelSpec:
    path = self.spec_root / f"{spec.spec_id}.json"
    path.write_text(json.dumps(asdict(spec), indent=2, default=str), encoding="utf-8")
    return spec

  def load_spec(self, spec_id: str) -> KernelSpec:
    payload = json.loads((self.spec_root / f"{spec_id}.json").read_text(encoding="utf-8"))
    for key in ("invariants", "mockgpu_command", "hardware_command"): payload[key] = tuple(payload.get(key, ()))
    return KernelSpec(**payload)

  def specs(self) -> list[KernelSpec]:
    ret = []
    for path in sorted(self.spec_root.glob("*.json")):
      try: ret.append(self.load_spec(path.stem))
      except Exception: continue
    return ret

  def create_candidate(self, spec_id: str, source: str, hypothesis: str, parent_id: str | None = None) -> CandidateRecord:
    spec = self.load_spec(spec_id)
    if not source.strip(): raise ValueError("candidate source must not be empty")
    if len(source.encode()) > 2_000_000: raise ValueError("candidate source exceeds 2 MB")
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    candidate_id = f"{spec.name}-{source_sha[:12]}"
    directory = self.candidate_root / candidate_id
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / f"candidate{spec.extension}"
    source_path.write_text(source, encoding="utf-8")
    record = CandidateRecord(candidate_id, spec_id, str(source_path), source_sha, parent_id, hypothesis.strip(), "staged", time.time())
    (directory / "manifest.json").write_text(json.dumps(asdict(record), indent=2, default=str), encoding="utf-8")
    return record

  def load_candidate(self, candidate_id: str) -> CandidateRecord:
    payload = json.loads((self.candidate_root / candidate_id / "manifest.json").read_text(encoding="utf-8"))
    return CandidateRecord(**payload)

  def candidates(self, spec_id: str | None = None) -> list[CandidateRecord]:
    ret = []
    for path in sorted(self.candidate_root.glob("*/manifest.json")):
      try:
        record = CandidateRecord(**json.loads(path.read_text(encoding="utf-8")))
        if spec_id is None or record.spec_id == spec_id: ret.append(record)
      except Exception: continue
    return sorted(ret, key=lambda x: x.created_at_s, reverse=True)

  def update(self, candidate_id: str, status: str, evidence: Mapping[str, Any]) -> CandidateRecord:
    record = self.load_candidate(candidate_id)
    merged = {**dict(record.evidence), **dict(evidence)}
    updated = replace(record, status=status, evidence=merged)
    path = self.candidate_root / candidate_id / "manifest.json"
    path.write_text(json.dumps(asdict(updated), indent=2, default=str), encoding="utf-8")
    return updated

  @staticmethod
  def render_command(command: Sequence[str], spec: KernelSpec, candidate: CandidateRecord, root: Path) -> tuple[str, ...]:
    bundle = str(spec.metadata.get("recipe_bundle", root))
    replacements = {"{candidate}": candidate.source_path, "{candidate_id}": candidate.candidate_id,
                    "{spec_id}": spec.spec_id, "{root}": str(root), "{bundle}": bundle}
    rendered: list[str] = []
    for original in command:
      part = str(original)
      for key, value in replacements.items(): part = part.replace(key, value)
      rendered.append(part)
    return tuple(rendered)
