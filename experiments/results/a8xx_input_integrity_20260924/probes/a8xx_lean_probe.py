import argparse
import ctypes
import sys
import numpy as np
from tinygrad import Device, TinyJit, dtypes
from tinygrad.nn.onnx import OnnxRunner
from experiments.a830_openpilot import deterministic_inputs

p=argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--reference',required=True)
p.add_argument('--output')
a=p.parse_args()
r=OnnxRunner(a.model)
inputs=[deterministic_inputs(r,s) for s in (42,43)]
Device['QCOM'].synchronize()
refs=np.load(a.reference)
@TinyJit(prune=True)
def run(**kwargs): return next(iter(r(kwargs).values())).cast(dtypes.float).realize()
for phase,j in enumerate((0,0,0,1)):
  out=run(**inputs[j]).numpy().copy()
  d=np.abs(out.astype(np.float32)-refs[f'arr_{j}'].astype(np.float32))
  print('phase',phase,'output_diff',int(np.count_nonzero(d)),'max',float(d.max()),flush=True)
corrupt=False
for seed,j in ((42,0),(43,1)):
  rng=np.random.default_rng(seed)
  for name,spec in sorted(r.graph_inputs.items()):
    shape=tuple(x if isinstance(x,int) else 1 for x in spec.shape)
    arr=rng.integers(0,256,shape,dtype=np.uint8) if spec.dtype is dtypes.uint8 else rng.standard_normal(shape).astype(np.float32)
    if name in ('big_img','img'):
      actual=inputs[j][name].numpy()
      count=int(np.count_nonzero(actual!=arr))
      print('input',seed,name,'diff',count,flush=True)
      if seed==43 and name=='big_img' and count:
        corrupt=True
        if a.output:
          scratch=Device['QCOM']._scratch.buf
          tail=min(2<<20,scratch.size)
          stack_tail=np.frombuffer(ctypes.string_at(int(scratch.va_addr)+scratch.size-tail,tail),dtype=np.uint8).copy()
          np.savez_compressed(a.output,expected=arr,actual=actual,scratch_tail=stack_tail)
          print('saved',a.output,'scratch',hex(int(scratch.va_addr)),scratch.size,flush=True)
if corrupt: sys.exit(7)
