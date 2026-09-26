"""Inspect the float32 Cody-Waite steps used by tinygrad's sine decomposition."""
import json, math
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.decomp.transcendental import payne_hanek_reduction, cody_waite_reduction, sin_poly, sin_poly_large, sin_poly_small, xsin
from tinygrad.uop.ops import Ops, UOp

angles=Tensor([10.,100.,1000.,10000.,100000.,1000000.],dtype=dtypes.float)
x=angles+math.pi/2
q=(x*0.318309886183790671537767526745028724 + 0.5).cast(dtypes.int).cast(dtypes.float)
stages={'angle':angles,'sin_input':x,'quadrant':q}
d=q*-3.1414794921875+x
stages['step1']=d
d=q*-0.00011315941810607910156+d
stages['step2']=d
d=q*-1.9841872589410058936e-09+d
stages['step3']=d
d=q*-1.2154201256553420762e-10+d
stages['step4']=d
stages['cos']=angles.cos()
reductions={}
for name,fn in (('payne',payne_hanek_reduction),('cody',cody_waite_reduction)):
  r,quad=fn(x.uop)
  reductions[name]=(r,quad)
  stages[f'{name}_remainder']=Tensor._wrap_uop(r)
  stages[f'{name}_quadrant']=Tensor._wrap_uop(quad)
  if name=='payne':
    odd=(quad & 1).ne(0)
    stages['payne_odd']=Tensor._wrap_uop(odd)
    stages['payne_poly_input']=Tensor._wrap_uop(r+odd.where(r.const_like(math.pi/2),r.const_like(0)))
    stages['payne_poly']=Tensor._wrap_uop(sin_poly(r+odd.where(r.const_like(math.pi/2),r.const_like(0))))
    stages['payne_result']=Tensor._wrap_uop(sin_poly_large(r,quad))
stages['cody_result']=Tensor._wrap_uop(sin_poly_small(*reductions['cody']))
stages['under_30']=Tensor._wrap_uop(x.uop<x.uop.const_like(30))
stages['combined']=Tensor._wrap_uop((x.uop<x.uop.const_like(30)).where(sin_poly_small(*reductions['cody']),sin_poly_large(*reductions['payne'])))
stages['xsin']=Tensor._wrap_uop(xsin(x.uop))
cx=(math.pi/2)-angles
cx_sign=cx.ne(0).where((cx<0).where(-1,1),0)
stages['cos_input']=cx
stages['cos_sign']=cx_sign
stages['cos_abs']=cx*cx_sign
stages['xsin_cos_input']=Tensor._wrap_uop(xsin(cx.uop))
sub=UOp(Ops.SUB,src=(UOp.const(math.pi/2,dtypes.float),angles.uop))
stages['forced_sub']=Tensor._wrap_uop(sub)
stages['xsin_forced_sub']=Tensor._wrap_uop(xsin(sub))
print(json.dumps({'device':Device.DEFAULT,'stages':{name:value.numpy().tolist() for name,value in stages.items()}},indent=2))
