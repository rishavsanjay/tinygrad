import argparse
import os
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.nn.onnx import OnnxRunner

p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--seed', type=int, default=43)
p.add_argument('--prefix-seed', type=int)
p.add_argument('--output', required=True)
a = p.parse_args()
r = OnnxRunner(a.model)

def inputs_for(seed):
  rng = np.random.default_rng(seed)
  ret, host = {}, {}
  for name, spec in sorted(r.graph_inputs.items()):
    shape = tuple(x if isinstance(x, int) else 1 for x in spec.shape)
    if spec.dtype is dtypes.uint8: arr = rng.integers(0, 256, shape, dtype=np.uint8)
    elif spec.dtype in (dtypes.float16, dtypes.float32): arr = rng.standard_normal(shape).astype(np.float32)
    else: raise RuntimeError(spec.dtype)
    host[name] = arr.copy()
    ret[name] = Tensor(arr, dtype=spec.dtype, device=Device.DEFAULT).realize()
  return ret, host

prefix = inputs_for(a.prefix_seed)[0] if a.prefix_seed is not None else None
target, host = inputs_for(a.seed)
arrays = {f'expected_{k}':v for k,v in host.items() if k in ('big_img','img')}
arrays.update({f'before_{k}':target[k].numpy().copy() for k in ('big_img','img')})
if prefix is not None:
  next(iter(r(prefix).values())).cast(dtypes.float).numpy()
arrays.update({f'after_prefix_{k}':target[k].numpy().copy() for k in ('big_img','img')})
final = next(iter(r(target).values())).cast(dtypes.float).numpy().copy()
arrays.update({f'after_target_{k}':target[k].numpy().copy() for k in ('big_img','img')})
if '_to_copy_1' in r.graph_values: arrays['node1'] = r.graph_values['_to_copy_1'].cast(dtypes.float).numpy().copy()
arrays['final'] = final
if os.getenv('SAVE_PROBE', '1') == '1': np.savez_compressed(a.output+'.npz', **arrays)
for k in ('big_img','img'):
  for stage in ('before','after_prefix','after_target'):
    d=arrays[f'{stage}_{k}'] != arrays[f'expected_{k}']
    print(k,stage,int(np.count_nonzero(d)),flush=True)
if 'node1' in arrays: print('node1 vs expected big_img',int(np.count_nonzero(arrays['node1'] != arrays['expected_big_img'])),flush=True)
print('saved',a.output,flush=True)
