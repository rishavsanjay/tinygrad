import argparse
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.nn.onnx import OnnxRunner
from experiments.a830_openpilot import deterministic_inputs
p=argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--corrupt',required=True)
p.add_argument('--reference',required=True)
a=p.parse_args()
r=OnnxRunner(a.model)
inputs=deterministic_inputs(r,43)
stored=np.load(a.corrupt)
bad=stored['after_prefix_big_img']
inputs['big_img']=Tensor(bad,dtype=dtypes.uint8,device=Device.DEFAULT).realize()
print('loaded_corrupt_bytes',int(np.count_nonzero(bad!=stored['expected_big_img'])),flush=True)
out=next(iter(r(inputs).values())).cast(dtypes.float).numpy().copy()
ref=np.load(a.reference)['arr_1']
d=np.abs(out.astype(np.float32)-ref.astype(np.float32))
print('corrupted_input_vs_saved_reference',int(np.count_nonzero(d)),float(d.max()),float(d.mean()),flush=True)
