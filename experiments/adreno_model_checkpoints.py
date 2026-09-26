"""Find the first ONNX node diverging across independently compiled A830 backends."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.nn.onnx import OnnxRunner
import tinygrad.nn.onnx as onnx

def sha(data): return hashlib.sha256(data).hexdigest()

def main():
  p=argparse.ArgumentParser()
  p.add_argument('model')
  p.add_argument('--output',required=True)
  p.add_argument('--seed',type=int,default=42)
  p.add_argument('--limits',type=int,nargs='+',required=True)
  p.add_argument('--save-arrays',action='store_true')
  a=p.parse_args()
  runner=OnnxRunner(a.model)
  rng=np.random.default_rng(a.seed)
  hosts={}
  for name,spec in sorted(runner.graph_inputs.items()):
    shape=tuple(x if isinstance(x,int) else 1 for x in spec.shape)
    hosts[name]=rng.integers(0,256,shape,dtype=np.uint8) if spec.dtype==dtypes.uint8 else \
      rng.standard_normal(shape).astype(np.float32).astype(np.float16 if spec.dtype==dtypes.half else np.float32)
  inputs={name:Tensor(arr,device=Device.DEFAULT).realize() for name,arr in hosts.items()}
  report={'device':Device.DEFAULT,'model_sha256':sha(Path(a.model).read_bytes()),'seed':a.seed,
          'input_hashes':{n:sha(v.tobytes()) for n,v in hosts.items()},
          'flags':{k:os.getenv(k) for k in ('DEV','IMAGE','FLOAT16','OPENPILOT_HACKS','NOLOCALS','JIT_BATCH_SIZE')},'nodes':{}}
  arrays={}
  try:
    for limit in a.limits:
      onnx.limit=limit
      out=runner(inputs)
      records=[]
      for name,value in out.items():
        if not isinstance(value,Tensor):
          records.append({'name':name,'type':type(value).__name__,'value':repr(value)[:200]})
          continue
        arr=value.numpy().copy()
        records.append({'name':name,'shape':arr.shape,'dtype':str(arr.dtype),'sha256':sha(arr.tobytes()),
                        'min':float(arr.min()),'max':float(arr.max()),'finite':bool(np.isfinite(arr).all())})
        if a.save_arrays: arrays[f'{limit}_{name}']=arr
      for name,host in hosts.items():
        if inputs[name].numpy().tobytes()!=host.tobytes(): raise AssertionError(f'input corruption at node {limit}: {name}')
      report['nodes'][str(limit)]={'op':runner.graph_nodes[limit].op,'outputs':records}
      print('checkpoint',Device.DEFAULT,limit,runner.graph_nodes[limit].op,[x.get('sha256') for x in records],flush=True)
  except BaseException as e:
    report['error']=f'{type(e).__name__}: {e}'
    raise
  finally:
    Path(a.output+'.json').write_text(json.dumps(report,indent=2))
    if a.save_arrays: np.savez(a.output+'.npz',**arrays)

if __name__=='__main__': main()
