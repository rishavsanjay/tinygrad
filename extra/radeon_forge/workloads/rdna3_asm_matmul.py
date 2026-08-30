from __future__ import annotations

import json
import math
import os
import statistics

import numpy as np

from tinygrad import Context, Device, GlobalCounters, Tensor
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import Estimates, run_linear
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.gemm import amd_asm_matmul as gemm


_ORDER_BUILDERS = {
  "optimized": lambda: list(gemm.FMAC_PAIR_ORDER),
  "row_major": lambda: [(a, b) for a in range(4) for b in range(8)],
  "column_major": lambda: [(a, b) for b in range(8) for a in range(4)],
  "snake": lambda: [(a, b) for a in range(4) for b in (range(8) if a % 2 == 0 else range(7, -1, -1))],
}


def _candidate() -> tuple[dict, int]:
  candidate = json.loads(os.environ["RADEON_FORGE_CANDIDATE_JSON"])
  budget = int(os.environ.get("RADEON_FORGE_BUDGET", "5"))
  if budget <= 0: raise ValueError("RADEON_FORGE_BUDGET must be positive")
  return candidate, budget


def _install_order(name: str) -> None:
  if name not in _ORDER_BUILDERS: raise ValueError(f"unknown FMAC order {name!r}")
  order = _ORDER_BUILDERS[name]()
  expected = {(a, b) for a in range(4) for b in range(8)}
  if len(order) != 32 or set(order) != expected: raise ValueError(f"invalid FMAC order {name!r}")
  gemm.FMAC_PAIR_ORDER = order
  gemm.FMAC_PATTERN = gemm.derive_fmac_pattern(gemm.ACC_GRID, gemm.V_A_TILE_REGS, gemm.V_B_TILE_REGS)


def _percentile(values: list[float], q: float) -> float:
  ordered = sorted(values)
  return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def run() -> dict:
  candidate, budget = _candidate()
  parameters = candidate["parameters"]
  n = int(parameters["N"])
  limit_occ = int(parameters["LIMIT_OCC"])
  order_name = str(parameters["FMAC_ORDER"])
  if limit_occ <= 0: raise ValueError("LIMIT_OCC must be positive")
  _install_order(order_name)

  dev = Device[Device.DEFAULT]
  arch = getattr(getattr(dev, "renderer", None), "target", None)
  arch_name = getattr(arch, "arch", None)
  if arch_name != "gfx1100": raise RuntimeError(f"RDNA3 workload requires gfx1100, got {arch_name!r}")

  instructions = gemm.build_kernel(n)
  rng = np.random.default_rng(42)
  a = Tensor(rng.random((n, n), dtype=np.float32) - 0.5)
  b = Tensor(rng.random((n, n), dtype=np.float32) - 0.5)
  c = Tensor.empty(n, n)
  Tensor.realize(a, b, c)

  grid, local = (n // gemm.BLOCK_N, n // gemm.BLOCK_M, 1), (gemm.THREADS, 1, 1)
  lds_size = max(gemm.LDS_SIZE, 65536 // limit_occ)

  def asm_kernel(A: UOp, B: UOp, C: UOp) -> UOp:
    gidxs = [UOp.special(size, f"gidx{index}") for index, size in enumerate(grid)]
    lidxs = [UOp.special(size, f"lidx{index}") for index, size in enumerate(local)]
    lds = UOp.placeholder((lds_size,), dtypes.uint8, 0, AddrSpace.LOCAL)
    sink = UOp.sink(A.base, B.base, C.base, lds, *gidxs, *lidxs,
                    arg=KernelInfo(name=f"radeon_forge_{candidate['candidate_id']}", estimates=Estimates(ops=n*n*n*2, mem=n*n*4*3)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=instruction) for instruction in instructions))))

  c = Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2]
  linear = c.schedule_linear()
  with Context(DEBUG=0):
    run_linear(linear)  # warmup and realize all lazy dependencies
    samples_us: list[float] = []
    for _ in range(budget):
      start = GlobalCounters.time_sum_s
      run_linear(linear)
      elapsed = (GlobalCounters.time_sum_s - start) * 1e6
      if not math.isfinite(elapsed) or elapsed <= 0: raise RuntimeError(f"invalid timing sample {elapsed}")
      samples_us.append(elapsed)

  reference = (a @ b).realize()
  output_np, reference_np = c.numpy(), reference.numpy()
  difference = np.abs(output_np - reference_np)
  denominator = np.maximum(np.abs(reference_np), 1e-6)
  max_abs_error = float(difference.max())
  max_rel_error = float((difference / denominator).max())
  mse = float(np.square(output_np - reference_np).mean())
  stable = len(samples_us) < 3 or statistics.pstdev(samples_us) / statistics.mean(samples_us) <= 0.10

  return {
    "compile_ok": True,
    "stable": stable,
    "samples_us": samples_us,
    "median_latency_us": statistics.median(samples_us),
    "p95_latency_us": _percentile(samples_us, 0.95),
    "correctness": {
      "passed": bool(np.isfinite(mse) and mse <= 1e-6),
      "max_abs_error": max_abs_error,
      "max_rel_error": max_rel_error,
      "checked_values": int(output_np.size),
      "reason": "" if np.isfinite(mse) and mse <= 1e-6 else f"mean squared error {mse}",
    },
    "resources": {
      "vgprs": 179,
      "sgprs": 56,
      "lds_bytes": lds_size,
      "scratch_bytes": 0,
      "spilled_vgprs": 0,
      "spilled_sgprs": 0,
    },
    "evidence": {
      "target": arch_name,
      "reference": "tinygrad matmul",
      "mean_squared_error": mse,
      "fmac_order": order_name,
      "limit_occ": limit_occ,
      "resource_source": "static register assignments and explicit LDS allocation",
      "kernel_instruction_count": len(instructions),
    },
  }


if __name__ == "__main__": print(json.dumps(run(), sort_keys=True))
