"""Reproduce batch-local Conv1D errors on the native A830 backend."""
import argparse, json
import numpy as np
import torch
from tinygrad import Device, Tensor

def main():
  p=argparse.ArgumentParser()
  p.add_argument('--output',required=True)
  p.add_argument('--seed',type=int,default=42)
  a=p.parse_args()
  rng=np.random.default_rng(a.seed)
  report={'device':Device.DEFAULT,'seed':a.seed,'cases':[]}
  arrays={}
  for cin,groups,h in ((1,1,2),(1,1,5),(3,1,5),(3,3,5)):
    x=rng.standard_normal((8,cin,11)).astype(np.float32)
    w=rng.standard_normal((6,cin//groups,h)).astype(np.float32)
    got=Tensor(x).conv2d(Tensor(w),groups=groups).numpy()
    ref=torch.nn.functional.conv1d(torch.from_numpy(x),torch.from_numpy(w),groups=groups).numpy()
    diff=np.abs(got-ref)
    key=f'c{cin}g{groups}h{h}'
    arrays.update({f'{key}_x':x,f'{key}_w':w,f'{key}_got':got,f'{key}_ref':ref})
    record={'case':key,'shape':list(got.shape),'bad_by_batch':np.count_nonzero(~np.isclose(got,ref,atol=1e-6,rtol=1e-3),axis=(1,2)).tolist(),
            'max_abs':float(diff.max()),'mean_abs':float(diff.mean())}
    print(record,flush=True)
    report['cases'].append(record)
  np.savez(a.output+'.npz',**arrays)
  with open(a.output+'.json','w') as f: json.dump(report,f,indent=2)

if __name__=='__main__': main()
