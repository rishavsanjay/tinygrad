"""A fresh process runs one pruned model capture and twenty changed-input replays."""
import argparse, hashlib, importlib, json, os
from pathlib import Path
import numpy as np
from tinygrad import TinyJit, Device, dtypes
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.helpers import FLOAT16, OPENPILOT_HACKS, JIT_BATCH_SIZE
from experiments.a830_openpilot import deterministic_inputs, captured_manifest

p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--seed', type=int, required=True)
p.add_argument('--output', required=True)
p.add_argument('--references', required=True)
p.add_argument('--reference-only', action='store_true')
p.add_argument('--no-prune', action='store_true', help='control: retain constant kernels in each replay')
a = p.parse_args()
runner = OnnxRunner(a.model)
inputs = [deterministic_inputs(runner, seed) for seed in (a.seed, a.seed+1)]
Device['QCOM'].synchronize()
if a.reference_only:
  results = [next(iter(runner(x).values())).cast(dtypes.float).numpy().copy() for x in inputs]
  np.savez(a.references, *results)
else:
  refs = np.load(a.references)
  @TinyJit(prune=not a.no_prune)
  def run(**kwargs): return next(iter(runner(kwargs).values())).cast(dtypes.float).realize()
  records = []
  def save_report():
    manifest_hash, manifest = captured_manifest(run) if run.captured is not None else (None, [])
    sources = {name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest() for name in
               ('tinygrad.runtime.ops_qcom', 'tinygrad.runtime.support.compiler_mesa', 'tinygrad.engine.jit')}
    Path(a.output+'.json').write_text(json.dumps({'seed':a.seed, 'prune':not a.no_prune, 'checks':records,
      'manifest':manifest_hash, 'kernels':manifest, 'sources':sources,
      'context':{'FLOAT16':FLOAT16.value, 'OPENPILOT_HACKS':OPENPILOT_HACKS.value, 'JIT_BATCH_SIZE':JIT_BATCH_SIZE.value},
      'environment':{k:os.environ.get(k) for k in ('DEV', 'IMAGE', 'NOLOCALS', 'QCOM_NO_LOCALS', 'IMAGE_DOT',
        'IMAGE_UPCAST_AMOUNT', 'QCOM_IMAGE_INVALIDATE', 'QCOM_DISPATCH_WFI', 'MESA_PATH')}}, indent=2))
    return manifest_hash
  for i in range(22):
    j = 0 if i < 2 else i%2
    out = run(**inputs[j]).numpy().copy()
    expected = refs[f'arr_{j}']
    exact = bool(np.array_equal(out, expected))
    records.append({'phase':i, 'input':j, 'exact':exact, 'finite':bool(np.isfinite(out).all()),
                    'max_abs':float(np.max(abs(out.astype(float)-expected.astype(float))))})
    if not exact:
      np.save(a.output+'.failure.npy', out)
      np.save(a.output+'.expected.npy', expected)
      save_report()
      raise AssertionError(records[-1])
  manifest_hash = save_report()
  print(f'PASS seed={a.seed} changed-input replays=20 manifest={manifest_hash}', flush=True)
