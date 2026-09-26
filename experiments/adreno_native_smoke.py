"""Serial A830 native-compiler checkpoint, with independent host expected values."""
import sys
from tinygrad import Device, Tensor, TinyJit, dtypes

dev=Device['ADRENO']
print('native_device',hex(dev.chip_id),dev.renderer.target,flush=True)

@TinyJit
def affine(x): return (x*3+2).realize()

for phase in range(6):
  values=[float(i+phase) for i in range(257)]
  x=Tensor(values,device='ADRENO',dtype=dtypes.float).realize()
  result=affine(x).tolist()
  assert result==[v*3+2 for v in values],(phase,result[:20])
  assert x.tolist()==values,'native kernel changed its input'
  print('affine_phase',phase,'PASS',flush=True)
dev.synchronize()
forbidden=[m for m in sys.modules if m in ('tinygrad.renderer.nir','tinygrad.runtime.support.compiler_mesa','tinygrad.runtime.autogen.mesa')]
assert not forbidden,forbidden
print('native_no_mesa_modules PASS',flush=True)
