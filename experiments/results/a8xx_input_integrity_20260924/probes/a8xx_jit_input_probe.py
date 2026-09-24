import argparse
import ctypes
import os
import numpy as np
from tinygrad import Device, TinyJit, dtypes
from tinygrad.nn.onnx import OnnxRunner
from experiments.a830_openpilot import deterministic_inputs
from tinygrad.runtime.ops_qcom import QCOMProgram, QCOMScratch, dcache_flush

guard_size = int(os.getenv('SCRATCH_GUARD', '0'))
guarded = []
if guard_size:
  original_scratch_init = QCOMScratch.__init__
  def guarded_scratch_init(self, dev, size):
    original_scratch_init(self, dev, size + guard_size)
    start = int(self.buf.va_addr) + size
    ctypes.memset(start, 0xA5, guard_size)
    dcache_flush().fxn(ctypes.c_uint64(start), guard_size // 64)
    guarded.append((self, size))
  QCOMScratch.__init__ = guarded_scratch_init

p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--reference', required=True)
a = p.parse_args()
r = OnnxRunner(a.model)
inputs = [deterministic_inputs(r, seed) for seed in (42, 43)]
Device['QCOM'].synchronize()
refs = np.load(a.reference)
for j in (0,1):
  for name in ('big_img','img'):
    b = inputs[j][name]._buffer()._buf
    print('address',j,name,hex(int(b.va_addr)),'size',b.size,flush=True)
target_buf = inputs[1]['big_img']._buffer()._buf
target_start, target_end = int(target_buf.va_addr), int(target_buf.va_addr) + target_buf.size
original_call = QCOMProgram.__call__
calls = []
writes = []
def traced_call(self, *bufs, **kwargs):
  calls.append(self.name)
  for slot in getattr(self, 'write_slots', ()):
    if slot < len(bufs):
      b=bufs[slot]
      if len(calls) <= 226:
        writes.append((self.name,slot,int(b.va_addr),b.size))
      if int(b.va_addr) < target_end and target_start < int(b.va_addr) + b.size:
        print('WRITE_OVERLAP',len(calls),self.name,slot,hex(int(b.va_addr)),b.size,flush=True)
  return original_call(self,*bufs,**kwargs)
QCOMProgram.__call__ = traced_call

@TinyJit(prune=True)
def run(**kwargs): return next(iter(r(kwargs).values())).cast(dtypes.float).realize()

for phase, j in enumerate((0, 0, 0, 1)):
  out = run(**inputs[j]).numpy().copy()
  want = refs[f'arr_{j}']
  d = np.abs(out.astype(np.float32)-want.astype(np.float32))
  print('phase',phase,'input',j,'output_diff',int(np.count_nonzero(d)),'max',float(d.max()),flush=True)
  print('program_calls',len(calls),flush=True)
  if phase == 0:
    dev=Device['QCOM']
    special_buffers = [('cmd',getattr(dev,'cmd_buf',None)),('args',getattr(dev,'kernargs_buf',None)),
                       ('stack',getattr(dev,'_stack',None)),('scratch',getattr(getattr(dev,'_scratch',None),'buf',None)),
                       ('border',getattr(dev,'border_color_buf',None))]
    for label,obj in special_buffers:
      if obj is not None: print('SPECIAL',label,hex(int(obj.va_addr)),obj.size,flush=True)
    nearest = sorted(writes, key=lambda x: max(0, x[2]-target_end, target_start-(x[2]+x[3])))[:12]
    for name,slot,start,size in nearest:
      print('NEAR_WRITE',name,slot,hex(start),size,'gap',max(0,start-target_end,target_start-(start+size)),flush=True)

for seed,j in ((42,0),(43,1)):
  rng = np.random.default_rng(seed)
  expected = {}
  for name,spec in sorted(r.graph_inputs.items()):
    shape = tuple(x if isinstance(x,int) else 1 for x in spec.shape)
    if spec.dtype is dtypes.uint8: arr=rng.integers(0,256,shape,dtype=np.uint8)
    else: arr=rng.standard_normal(shape).astype(np.float32)
    if name in ('big_img','img'): expected[name] = arr
  for name in ('big_img','img'):
    actual = inputs[j][name].numpy()
    print('input',seed,name,'different_bytes',int(np.count_nonzero(actual != expected[name])),flush=True)
for item,size in guarded:
  arr = np.frombuffer(ctypes.string_at(int(item.buf.va_addr)+size,guard_size),dtype=np.uint8)
  ix = np.flatnonzero(arr != 0xA5)
  print('GUARD',hex(int(item.buf.va_addr)),size,'changed',len(ix),'first',ix[:20].tolist(),'last',ix[-20:].tolist(),flush=True)
