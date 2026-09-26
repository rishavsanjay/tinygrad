"""Model comparison with host-owned input snapshots and checked reference provenance."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.nn.onnx import OnnxRunner
from experiments.a830_openpilot import captured_manifest

def sha(data): return hashlib.sha256(data).hexdigest()

def main():
  p=argparse.ArgumentParser()
  p.add_argument('model')
  p.add_argument('--seed',type=int,default=42)
  p.add_argument('--references',required=True)
  p.add_argument('--output',required=True)
  p.add_argument('--reference-only',action='store_true')
  p.add_argument('--replay-despite-reference-mismatch',action='store_true',help='collect replay evidence and fail the reference gate afterward')
  p.add_argument('--replays',type=int,default=20)
  p.add_argument('--rtol',type=float,default=2e-3)
  p.add_argument('--atol',type=float,default=2e-3)
  a=p.parse_args()
  reference_data=Path(a.references if a.references.endswith('.npz') else a.references+'.npz')
  reference_meta=reference_data.with_suffix('.json')
  report={'model_sha256':sha(Path(a.model).read_bytes()),'device':Device.DEFAULT,'seed':a.seed,'checks':[],
          'flags':{k:os.getenv(k) for k in ('DEV','IMAGE','FLOAT16','OPENPILOT_HACKS','NOLOCALS','JIT_BATCH_SIZE')},
          'rtol':a.rtol,'atol':a.atol}
  if Device.DEFAULT.split(':')[0]=='ADRENO':
    from tinygrad.runtime.support.compiler_adreno import BUILD
    report['native_build']=BUILD
  try: runner=OnnxRunner(a.model)
  except BaseException as e:
    report['error']=f'{type(e).__name__}: {e}'
    Path(a.output+'.json').write_text(json.dumps(report,indent=2))
    raise
  hosts=[]
  for seed in (a.seed,a.seed+1):
    rng=np.random.default_rng(seed)
    host={}
    for name,spec in sorted(runner.graph_inputs.items()):
      shape=tuple(x if isinstance(x,int) else 1 for x in spec.shape)
      host[name]=rng.integers(0,256,shape,dtype=np.uint8) if spec.dtype==dtypes.uint8 else \
        rng.standard_normal(shape).astype(np.float32).astype(np.float16 if spec.dtype==dtypes.half else np.float32)
      host[name].flags.writeable=False
    hosts.append(host)
  inputs=[{name:Tensor(arr,device=Device.DEFAULT).realize() for name,arr in host.items()} for host in hosts]
  report['input_hashes']=[{name:sha(arr.tobytes()) for name,arr in host.items()} for host in hosts]
  def check_inputs(phase):
    for seed,(inp,host) in enumerate(zip(inputs,hosts)):
      for name,expected in host.items():
        actual=inp[name].numpy().copy()
        if actual.tobytes()!=expected.tobytes():
          np.savez(a.output+'.input-corruption.npz',expected=expected,actual=actual)
          raise AssertionError(f'input corruption at {phase}, seed {a.seed+seed}, {name}: {np.count_nonzero(actual!=expected)} elements')
  def evaluate(inp): return next(iter(runner(inp).values())).cast(dtypes.float).realize()
  eager=[]
  try:
    check_inputs('allocation')
    for seed,inp in enumerate(inputs):
      start=time.perf_counter()
      print('model_eager_start',a.seed+seed,flush=True)
      eager.append(evaluate(inp).numpy().copy())
      check_inputs(f'eager {a.seed+seed}')
      if not np.isfinite(eager[-1]).all(): raise AssertionError('non-finite eager model output')
      report['checks'].append({'phase':'reference','seed':a.seed+seed,'seconds':time.perf_counter()-start,'sha256':sha(eager[-1].tobytes())})
      print('model_eager_done',a.seed+seed,'input_bytes_unchanged',flush=True)
    if a.reference_only:
      np.savez(reference_data,*eager)
      reference_meta.write_text(json.dumps(report,indent=2))
      return
    provenance=json.loads(reference_meta.read_text())
    if provenance['model_sha256']!=report['model_sha256'] or provenance['input_hashes']!=report['input_hashes']:
      raise AssertionError('reference model/input provenance mismatch')
    report['reference_device']=provenance['device']
    report['independent_reference']=provenance['device']!=report['device']
    if not report['independent_reference']: raise AssertionError('reference must use an independent device/compiler path')
    refs=np.load(reference_data)
    reference_failures=[]
    for seed,actual in enumerate(eager):
      expected=refs[f'arr_{seed}']
      delta=abs(actual.astype(np.float64)-expected.astype(np.float64))
      comparison={'phase':'independent_reference','seed':a.seed+seed,'max_abs':float(delta.max()),
                  'mean_abs':float(delta.mean()),'not_close':int(np.count_nonzero(~np.isclose(actual,expected,rtol=a.rtol,atol=a.atol))),
                  'allclose':bool(np.allclose(actual,expected,rtol=a.rtol,atol=a.atol))}
      report['checks'].append(comparison)
      if not comparison['allclose']:
        np.savez(a.output+f'.output-mismatch-{a.seed+seed}.npz',expected=expected,actual=actual)
        reference_failures.append(comparison)
        if not a.replay_despite_reference_mismatch: raise AssertionError(comparison)
    run=TinyJit(prune=True)(evaluate)
    for phase in range(a.replays+2):
      seed=0 if phase<2 else phase%2
      actual=run(inputs[seed]).numpy().copy()
      check_inputs(f'JIT phase {phase}')
      exact=bool(np.array_equal(actual,eager[seed]))
      report['checks'].append({'phase':phase,'seed':a.seed+seed,'exact':exact,'sha256':sha(actual.tobytes())})
      if not exact:
        np.savez(a.output+'.jit-mismatch.npz',expected=eager[seed],actual=actual)
        raise AssertionError(report['checks'][-1])
      print('model_jit_phase',phase,'PASS',flush=True)
    report['manifest_sha256'],report['kernels']=captured_manifest(run)
    if reference_failures: raise AssertionError(f'independent reference mismatch: {reference_failures}')
  except BaseException as e:
    report['error']=f'{type(e).__name__}: {e}'
    raise
  finally:
    if Device.DEFAULT.split(':')[0]=='ADRENO':
      report['mesa_modules']=[m for m in sys.modules if m in ('tinygrad.renderer.nir','tinygrad.runtime.support.compiler_mesa',
                                                          'tinygrad.runtime.autogen.mesa')]
    Path(a.output+'.json').write_text(json.dumps(report,indent=2))
    # ONNX CPU-side operations can instantiate ops_cpu, which imports
    # renderer.nir. The Mesa compiler itself must remain absent.
    if 'tinygrad.runtime.support.compiler_mesa' in report.get('mesa_modules',[]):
      raise AssertionError('Mesa compiler imported during native model execution')

if __name__=='__main__': main()
