"""Create an independent ONNX Runtime CPU reference for A830 model qualification."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tinygrad import dtypes
from tinygrad.nn.onnx import OnnxRunner

def sha(data): return hashlib.sha256(data).hexdigest()

def main():
  p=argparse.ArgumentParser()
  p.add_argument('model')
  p.add_argument('--output',required=True)
  p.add_argument('--seed',type=int,default=42)
  a=p.parse_args()
  runner=OnnxRunner(a.model)
  session=ort.InferenceSession(a.model,providers=['CPUExecutionProvider'])
  report={'model_sha256':sha(Path(a.model).read_bytes()),'device':'ONNXRUNTIME:CPU','seed':a.seed,'checks':[],'input_hashes':[]}
  outputs=[]
  for seed in (a.seed,a.seed+1):
    rng=np.random.default_rng(seed)
    hosts={}
    for name,spec in sorted(runner.graph_inputs.items()):
      shape=tuple(x if isinstance(x,int) else 1 for x in spec.shape)
      hosts[name]=rng.integers(0,256,shape,dtype=np.uint8) if spec.dtype==dtypes.uint8 else \
        rng.standard_normal(shape).astype(np.float32).astype(np.float16 if spec.dtype==dtypes.half else np.float32)
    report['input_hashes'].append({name:sha(value.tobytes()) for name,value in hosts.items()})
    start=time.perf_counter()
    result=session.run(None,hosts)
    if len(result)!=1: raise ValueError(f'expected one model output, got {len(result)}')
    output=result[0].astype(np.float32)
    if not np.isfinite(output).all(): raise AssertionError(f'non-finite ONNX Runtime output for seed {seed}')
    outputs.append(output)
    report['checks'].append({'phase':'reference','seed':seed,'seconds':time.perf_counter()-start,'sha256':sha(output.tobytes())})
    print('onnxruntime_reference',seed,output.shape,report['checks'][-1]['sha256'],flush=True)
  np.savez(a.output,*outputs)
  Path(a.output+'.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
