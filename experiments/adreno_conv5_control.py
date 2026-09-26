"""Isolate the first model Conv's input and FP16 arithmetic effects on A830."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from tinygrad import Device, Tensor

def sha(array:np.ndarray) -> str: return hashlib.sha256(array.tobytes()).hexdigest()

def main():
  p=argparse.ArgumentParser()
  p.add_argument('--inputs',required=True,help='NPZ with cpu_div and native_div node-4 tensors')
  p.add_argument('--weights',required=True,help='NPZ with Conv node-5 w and b tensors')
  p.add_argument('--output',required=True)
  a=p.parse_args()
  inputs,weights=np.load(a.inputs),np.load(a.weights)
  report={'device':Device.DEFAULT,'inputs_sha256':{key:sha(inputs[key]) for key in ('cpu_div','native_div')},
          'weights_sha256':{key:sha(weights[key]) for key in ('w','b')},'outputs':{}}
  out={}
  for source in ('cpu_div','native_div'):
    x,w,b=Tensor(inputs[source]).realize(),Tensor(weights['w']).realize(),Tensor(weights['b']).realize()
    out[source]=x.conv2d(w,b,stride=(2,2),padding=(1,1,1,1)).numpy().copy()
    report['outputs'][source]={'sha256':sha(out[source]),'shape':list(out[source].shape),'dtype':str(out[source].dtype)}
  rng=np.random.default_rng(64)
  x=inputs['cpu_div'].ravel()[rng.integers(inputs['cpu_div'].size,size=131072)]
  w=weights['w'].ravel()[rng.integers(weights['w'].size,size=131072)]
  got=(Tensor(x)*Tensor(w)).numpy()
  ref=(x*w).astype(np.float16)
  report['fp16_product']={'sampled':got.size,'bitwise_bad':int(np.count_nonzero(got.view(np.uint16)!=ref.view(np.uint16)))}
  np.savez(a.output+'.npz',**out)
  Path(a.output+'.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)

if __name__=='__main__': main()
