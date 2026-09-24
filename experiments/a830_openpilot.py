#!/usr/bin/env python3
"""Benchmark OpenPilot's driving model on A830 with optional KGSL counters."""
from __future__ import annotations
import argparse, hashlib, json, math, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Match openpilot/selfdrive/modeld/SConscript. Callers may override any knob.
# Parsing a few scalar ONNX initializers realizes them immediately. Compile-only
# mode parses those on CPU, then moves the complete lazy graph to QCOM.
os.environ.setdefault("DEV", "CPU" if "--compile-only" in sys.argv else "QCOM")
os.environ.setdefault("IMAGE", "1")
os.environ.setdefault("FLOAT16", "1")
os.environ.setdefault("JIT_BATCH_SIZE", "0")
os.environ.setdefault("NOLOCALS", "1")
os.environ.setdefault("OPENPILOT_HACKS", "1")
if "--profile" in sys.argv: os.environ["PROFILE"] = "1"  # register tinygrad's atexit trace writer before imports

import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import Context, PROFILE, temp
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.uop.ops import Ops

def percentile(vals:list[float], q:float) -> float:
  ordered = sorted(vals)
  return ordered[min(len(ordered)-1, math.ceil(q * len(ordered))-1)]

def deterministic_inputs(runner:OnnxRunner, seed:int) -> dict[str, Tensor]:
  rng = np.random.default_rng(seed)
  ret:dict[str, Tensor] = {}
  input_placement_npy = bool(int(os.environ.get("INPUT_PLACEMENT_NPY", "0")))
  for name, spec in sorted(runner.graph_inputs.items()):
    shape = tuple(x if isinstance(x, int) else 1 for x in spec.shape)
    if spec.dtype is dtypes.uint8: arr = rng.integers(0, 256, shape, dtype=np.uint8)
    elif spec.dtype in (dtypes.float16, dtypes.float32): arr = rng.standard_normal(shape).astype(np.float32)
    else: raise RuntimeError(f"unsupported OpenPilot input dtype {spec.dtype} for {name}")
    ret[name] = Tensor(arr, dtype=spec.dtype, device=Device.DEFAULT if (not input_placement_npy or "img" in name) else "NPY").realize()
  return ret

def captured_manifest(run:TinyJit) -> tuple[str, list[dict]]:
  assert run.captured is not None
  records = []
  calls = [u for u in run.captured.linear.toposort(gate=lambda x: x.op is not Ops.PROGRAM)
           if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM]
  for call in calls:
    prg, elf = call.src[0], call.src[0].to_elf()
    records.append({"name": elf.name, "binary_sha256": hashlib.sha256(elf.lib).hexdigest(),
                    "signature": [(name, slot, str(dtype), list(shape)) for name,slot,dtype,shape in elf.signature],
                    "global_size": repr(prg.arg.global_size), "local_size": repr(prg.arg.local_size)})
  payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(payload).hexdigest(), records

def compile_only(model:str, chip_id:int) -> dict:
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.nir import IR3Renderer
  runner = OnnxRunner(model)
  def empty_inputs(device:str) -> dict[str, Tensor]:
    inputs = {}
    for name, spec in sorted(runner.graph_inputs.items()):
      shape = tuple(x if isinstance(x, int) else 1 for x in spec.shape)
      dtype = dtypes.float32 if spec.dtype is dtypes.float16 else spec.dtype
      inputs[name] = Tensor.empty(*shape, dtype=dtype, device=device)
    return inputs
  # Populate ONNX's Python-constant cache while scalar/shape tensors are still on CPU.
  runner(empty_inputs("CPU"))
  runner.to("QCOM")
  inputs = empty_inputs("QCOM")
  output = next(iter(runner(inputs).values())).cast(dtypes.float32)
  linear = output.schedule_linear()
  calls = [x for x in linear.src if x.op is Ops.CALL and x.src[0].op is Ops.SINK]
  renderer = IR3Renderer(Target("QCOM", "IR3", f"a830,chip_id={chip_id:#x}"))
  binary_sizes, nir_sizes = [], []
  for call in calls:
    program = to_program(call.src[0], renderer)
    nir_sizes.append(len(program.src[2].arg))
    binary_sizes.append(len(program.src[3].arg))
  return {"schema": 1, "mode": "compile_only", "mesa": "26.2.1", "chip_id": f"{chip_id:#x}",
          "model": os.path.basename(model), "model_bytes": os.path.getsize(model), "kernels": len(calls),
          "nir_bytes": sum(nir_sizes), "binary_bytes": sum(binary_sizes), "max_binary_bytes": max(binary_sizes, default=0)}

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("model", help="driving_supercombo.onnx")
  ap.add_argument("--warmup", type=int, default=5)
  ap.add_argument("--runs", type=int, default=20)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--counters", action="store_true", help="collect the existing kernel-managed KGSL counter window per run")
  ap.add_argument("--profile", action="store_true", help="capture one post-benchmark, per-kernel GPU timestamp trace")
  ap.add_argument("--diagnose-warmup", action="store_true", help="report eager/capture/replay output differences without timing")
  ap.add_argument("--save-output", help="write the validated replay output to a .npy file for cross-process comparison")
  ap.add_argument("--save-manifest", help="write the captured kernel binary/signature/launch manifest as JSON")
  ap.add_argument("--compile-only", action="store_true", help="host-side compile every model kernel for A830 without opening KGSL")
  ap.add_argument("--chip-id", type=lambda x:int(x, 0), default=0x44050001)
  args = ap.parse_args()
  if args.warmup < 4: ap.error("--warmup must be at least 4 to capture and validate TinyJit replays")
  if args.runs < 1: ap.error("--runs must be positive")
  if args.compile_only and args.profile: ap.error("--profile requires on-device execution")
  if args.compile_only:
    print("A830_OPENPILOT_RESULT=" + json.dumps(compile_only(args.model, args.chip_id), sort_keys=True), flush=True)
    return

  # PROFILE=1 at import registers the trace finalizer. Keep the TinyJit capture and timed runs uninstrumented.
  if args.profile: PROFILE.value = 0
  dev = Device.default
  if dev.device.split(":")[0] != "QCOM" or getattr(dev, "gen", None) != 8:
    raise RuntimeError(f"this benchmark requires an A8xx QCOM device, got {dev.device}")
  runner = OnnxRunner(args.model)
  inputs = deterministic_inputs(runner, args.seed)
  if not bool(int(os.environ.get("DISABLE_PRE_SYNC", "0"))): dev.synchronize()

  @TinyJit(prune=True)
  def run(**kwargs):
    if bool(int(os.environ.get("INPUT_PLACEMENT_NPY", "0"))):
      return next(iter(runner({name:value.to(Device.DEFAULT) for name,value in kwargs.items()}).values())).cast(dtypes.float32)
    return next(iter(runner(kwargs).values())).cast(dtypes.float32)

  warmup_ms:list[float] = []
  warmup_checks:list[dict] = []
  reference:np.ndarray|None = None
  for warmup_idx in range(args.warmup):
    st = time.perf_counter()
    out = run(**inputs)
    dev.synchronize()
    warmup_ms.append((time.perf_counter() - st) * 1e3)
    current = out.numpy()
    # The first two calls prime and capture; use the first ordinary replay as the production correctness reference.
    if warmup_idx == 2: reference = current.copy()
    compare = current if reference is None else reference
    diff = np.abs(current.astype(np.float64) - compare.astype(np.float64))
    check = {"iteration": warmup_idx, "phase": "eager" if warmup_idx == 0 else "capture" if warmup_idx == 1 else "replay",
             "sha256": hashlib.sha256(current.tobytes()).hexdigest(),
             "finite": bool(np.isfinite(current).all()), "exact": bool(np.array_equal(compare, current)),
             "equal_nan": bool(np.array_equal(compare, current, equal_nan=True)),
             "max_abs": float(np.nanmax(diff)) if not np.isnan(diff).all() else None,
             "different": int(np.count_nonzero(current != compare))}
    warmup_checks.append(check)
    if not args.diagnose_warmup and warmup_idx >= 2 and (not check["finite"] or not check["exact"]):
      raise RuntimeError(f"OpenPilot output changed across identical warmup inputs: {check}")

  if args.diagnose_warmup:
    manifest_sha256, manifest = captured_manifest(run)
    if args.save_output and reference is not None: np.save(args.save_output, reference)
    if args.save_manifest:
      with open(args.save_manifest, "w") as f: json.dump(manifest, f, sort_keys=True, separators=(",", ":"))
    print("A830_OPENPILOT_WARMUP=" + json.dumps({"schema": 1, "warmup_ms": warmup_ms, "checks": warmup_checks,
          "shape": list(reference.shape) if reference is not None else None, "kernels": len(manifest),
          "manifest_sha256": manifest_sha256}, sort_keys=True), flush=True)
    return

  perf, counter_error = None, None
  if args.counters:
    try: perf = dev.perf
    except (OSError, RuntimeError) as e: counter_error = f"{type(e).__name__}: {e}"

  total_ms:list[float] = []
  enqueue_ms:list[float] = []
  counter_runs:list[dict[str, int]] = []
  for _ in range(args.runs):
    dev.synchronize()
    if perf is not None: perf.begin()
    st = time.perf_counter()
    out = run(**inputs)
    queued = time.perf_counter()
    dev.synchronize()
    done = time.perf_counter()
    if perf is not None: counter_runs.append(perf.end())
    enqueue_ms.append((queued - st) * 1e3)
    total_ms.append((done - st) * 1e3)

  final = out.numpy()
  assert reference is not None
  if not np.array_equal(reference, final): raise RuntimeError("OpenPilot output changed after benchmark runs")
  if args.save_output: np.save(args.save_output, final)
  if args.save_manifest:
    manifest_sha256, manifest = captured_manifest(run)
    with open(args.save_manifest, "w") as f: json.dump(manifest, f, sort_keys=True, separators=(",", ":"))
  profile_path, profile_sha256, profile_comparison = None, None, None
  if args.profile:
    import gc
    from tinygrad.device import Buffer, Compiled, ProfileDeviceEvent, ProfileProgramEvent
    from tinygrad.helpers import cpu_events, profile_marker

    # HCQ graph timestamps are fixed at graph construction. Capture a separate instrumented graph after timing.
    @TinyJit(prune=True)
    def profile_run(**kwargs):
      return next(iter(runner({name:value.to(Device.DEFAULT) for name,value in kwargs.items()}).values())).cast(dtypes.float32)
    profile_run.cnt = 1  # skip the known-bad eager priming phase and start at capture
    with Context(PROFILE=1):
      profile_run(**inputs)
      dev.synchronize()
      profile_reference_out = profile_run(**inputs)
      dev.synchronize()
      profile_reference = profile_reference_out.numpy()
      # Discard capture/first-replay trace noise; retain device/program metadata before measuring one stable replay.
      Compiled.profile_events[:] = [x for x in Compiled.profile_events if isinstance(x, (ProfileDeviceEvent, ProfileProgramEvent))]
      Buffer.profile_events.clear()
      cpu_events.clear()
      profile_marker("openpilot_replay_start")
      profile_out = profile_run(**inputs)
      dev.synchronize()
      profile_marker("openpilot_replay_end")
      profile_np = profile_out.numpy()
      assert profile_run.captured is not None
      profile_run.captured.free_intermediates()
      del profile_run
      gc.collect()
      dev.synchronize()
    profile_path, profile_sha256 = temp("profile.pkl", append_user=True), hashlib.sha256(profile_np.tobytes()).hexdigest()
    if not np.isfinite(profile_reference).all() or not np.array_equal(profile_reference, profile_np):
      raise RuntimeError("instrumented OpenPilot graph is not exactly repeatable")
    diff = np.abs(profile_np.astype(np.float64) - reference.astype(np.float64))
    profile_comparison = {"exact": bool(np.array_equal(reference, profile_np)), "allclose_1e-3":
                          bool(np.allclose(reference, profile_np, rtol=1e-3, atol=1e-3)),
                          "max_abs": float(np.nanmax(diff)), "different": int(np.count_nonzero(profile_np != reference))}
    if not np.isfinite(profile_np).all() or not profile_comparison["allclose_1e-3"]:
      raise RuntimeError(f"profiled OpenPilot graph exceeds benchmark tolerance: {profile_comparison}")
  kernel_calls = [u for u in run.captured.linear.toposort(gate=lambda x: x.op is not Ops.PROGRAM)
                  if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM]
  result = {
    "schema": 1, "device": dev.device, "chip_id": f"{dev.chip_id:#x}", "mesa": "26.2.1",
    "model": os.path.basename(args.model), "model_bytes": os.path.getsize(args.model), "seed": args.seed,
    "flags": {k:os.environ.get(k) for k in ("IMAGE", "FLOAT16", "NOLOCALS", "QCOM_NO_LOCALS", "JIT_BATCH_SIZE", "OPENPILOT_HACKS", "QCOM_F16_MAD")},
    "kernels": len(kernel_calls), "warmup_ms": warmup_ms, "warmup_checks": warmup_checks,
    "enqueue_ms": enqueue_ms, "total_ms": total_ms,
    "latency": {"min_ms": min(total_ms), "median_ms": statistics.median(total_ms), "p90_ms": percentile(total_ms, 0.90),
                "max_ms": max(total_ms), "median_hz": 1000.0 / statistics.median(total_ms)},
    "output": {"shape": list(final.shape), "dtype": str(final.dtype), "finite": bool(np.isfinite(final).all()),
               "sha256": hashlib.sha256(final.tobytes()).hexdigest()},
    "counter_error": counter_error,
    "counters": counter_runs,
    "counter_metrics": [perf.metrics(x) for x in counter_runs] if perf is not None else [],
    "profile": {"path": profile_path, "output_sha256": profile_sha256,
                "vs_benchmark": profile_comparison,
                "note": "written on normal process exit; hardware timestamps intentionally exclude the timed TinyJit runs"},
  }
  print("A830_OPENPILOT_RESULT=" + json.dumps(result, sort_keys=True), flush=True)

if __name__ == "__main__": main()
