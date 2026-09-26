"""Compare Android Torch grouped Conv2D and optional KGSL execution with a scalar reference.

Run on the A830 phone, for example:
  PYTHONPATH=.:.deps python experiments/adreno_grouped_conv_reference.py --output results/torch-8.json --processes 10
  PYTHONPATH=.:.deps python experiments/adreno_grouped_conv_reference.py --output results/torch-1.json --processes 10 --torch-threads 1
  DEV=ADRENO PYTHONPATH=.:.deps python experiments/adreno_grouped_conv_reference.py --output results/native.json --device ADRENO
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

import numpy as np


def scalar_conv(x:np.ndarray, w:np.ndarray) -> np.ndarray:
  out = np.empty((4,35,3,3),dtype=np.float32)
  for batch in range(4):
    for oc in range(35):
      group = oc//7
      for oy in range(3):
        for ox in range(3):
          value = 0.0
          for ci in range(3):
            for ky in range(3):
              for kx in range(3): value += float(x[batch,group*3+ci,oy+ky,ox+kx])*float(w[oc,ci,ky,kx])
          out[batch,oc,oy,ox] = value
  return out


def check(actual:np.ndarray, reference:np.ndarray) -> dict:
  bad = ~np.isclose(actual,reference,atol=1e-6,rtol=1e-3)
  return {'bad_count':int(bad.sum()), 'max_error':float(np.max(np.abs(actual-reference))),
          'bad_blocks':[[batch,group,int(bad[batch,group*7:(group+1)*7].sum())]
                        for batch in range(4) for group in range(5) if bad[batch,group*7:(group+1)*7].any()]}


def worker(device:str|None, torch_threads:int|None) -> dict:
  import torch
  if torch_threads is not None: torch.set_num_threads(torch_threads)
  np.random.seed(0)
  x = np.random.uniform(-2,2,(4,15,5,5)).astype(np.float32)
  w = np.random.uniform(-2,2,(35,3,3,3)).astype(np.float32)
  reference = scalar_conv(x,w)
  result = {'torch_version':torch.__version__, 'torch_threads':torch.get_num_threads(),
            'input_sha256':[hashlib.sha256(a.tobytes()).hexdigest() for a in (x,w)],
            'torch':check(torch.nn.functional.conv2d(torch.from_numpy(x),torch.from_numpy(w),groups=5).numpy(),reference)}
  if device:
    from tinygrad import Tensor
    tx,tw = (Tensor(a,device=device).contiguous().realize() for a in (x,w))
    actual = tx.conv2d(tw,groups=5).contiguous().numpy()
    result['device'] = device
    result['gpu'] = check(actual,reference)
    result['inputs_unchanged'] = all(np.array_equal(t.numpy(),a) for t,a in ((tx,x),(tw,w)))
  return result


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--output',required=True)
  p.add_argument('--processes',type=int,default=10)
  p.add_argument('--device',choices=('ADRENO','QCOM'))
  p.add_argument('--torch-threads',type=int)
  p.add_argument('--worker',action='store_true')
  a = p.parse_args()
  if a.worker:
    print(json.dumps(worker(a.device,a.torch_threads)),flush=True)
    return
  if a.processes < 1 or a.torch_threads is not None and a.torch_threads < 1: raise ValueError('invalid process or thread count')
  output = Path(a.output)
  output.parent.mkdir(parents=True,exist_ok=True)
  report = {'device':a.device,'torch_threads':a.torch_threads,'processes_requested':a.processes,'processes':[]}
  for n in range(a.processes):
    env = os.environ.copy()
    env['CACHEDB'] = str(output.with_suffix(f'.process-{n}.db'))
    cmd = [sys.executable,__file__,'--output',str(output),'--worker']
    if a.device: cmd += ['--device',a.device]
    if a.torch_threads is not None: cmd += ['--torch-threads',str(a.torch_threads)]
    result = subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=90)
    if result.returncode: raise RuntimeError(f'process {n} failed: {result.stderr[-3000:]}')
    report['processes'].append(json.loads(next(line for line in reversed(result.stdout.splitlines()) if line.startswith('{'))))
    output.write_text(json.dumps(report,indent=2)+'\n')
  report['torch_failures'] = sum(p['torch']['bad_count']>0 for p in report['processes'])
  if a.device: report['gpu_failures'] = sum(p['gpu']['bad_count']>0 or not p['inputs_unchanged'] for p in report['processes'])
  output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k!='processes'}),flush=True)


if __name__ == '__main__': main()
