from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import Candidate


@dataclass(frozen=True)
class KernelFamily:
  name: str
  source_path: Path
  fixed_defines: Mapping[str, int | float | str | bool]
  search_space: Mapping[str, Sequence[int | float | str | bool]]
  hypothesis: str

  def candidates(self) -> list[Candidate]:
    names = tuple(self.search_space)
    values = [self.search_space[name] for name in names]
    candidates: list[Candidate] = []
    for combination in itertools.product(*values):
      params: dict[str, int | float | str | bool] = dict(self.fixed_defines)
      params.update(zip(names, combination))
      digest = hashlib.sha256(repr(sorted(params.items())).encode()).hexdigest()[:10]
      candidates.append(Candidate(candidate_id=f"{self.name}-{digest}", family=self.name, parameters=params,
                                  source_path=str(self.source_path), hypothesis=self.hypothesis))
    return candidates


def compiler_defines(parameters: Mapping[str, Any]) -> list[str]:
  defines: list[str] = []
  for key, value in sorted(parameters.items()):
    if isinstance(value, bool): value = int(value)
    defines.append(f"-D{key}={value}")
  return defines


def rdna3_rmsnorm_fp8_family(root: str | Path, n_elems: int, hidden: int, eps_literal: str = "1e-5f") -> KernelFamily:
  """Seed family from tinygrad's fused RMSNorm/multiply/FP8 kernel.

  The source was originally tuned for a different AMD target. Forge treats it
  as a pattern and recompiles every candidate for gfx1100; unsupported or
  resource-heavy candidates are rejected before benchmarking.
  """
  source = Path(root) / "extra/llama_kernels/fused_rmsnorm_mul_quantize_fp8/fused_rmsnorm_mul_quantize_fp8.cpp"
  return KernelFamily(
    name="rdna3-fused-rmsnorm-mul-quantize-fp8",
    source_path=source,
    fixed_defines={"N_ELEMS": n_elems, "HIDDEN": hidden, "EPS_LITERAL": eps_literal, "HAS_RESIDUAL": 0},
    search_space={"NUM_WG": (128, 256, 512, 1024), "THREADS_PER_WG": (64, 128, 256)},
    hypothesis="Tune work distribution for gfx1100 while preserving the fused single-HBM-pass algorithm.",
  )
