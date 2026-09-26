"""A830 image process/replay gates. Run with DEV=ADRENO IMAGE=2 and MESA_PATH=/nonexistent."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

def sha(a): return hashlib.sha256(a.tobytes()).hexdigest()

def check_once(seed:int):
  import numpy as np
  from tinygrad import Device, Tensor
  from tinygrad.codegen import to_program
  from tinygrad.runtime.support.compiler_adreno import AdrenoCompiler
  from tinygrad.uop.ops import Ops
  if Device.DEFAULT != 'ADRENO' or 'QCOM_IMAGE_PITCH_ALIGNMENT=16' not in Device['ADRENO'].arch:
    raise RuntimeError('requires native A830 with images enabled')
  probe=Tensor.empty(7,16,4,device='ADRENO')
  probe_calls=[c for c in (probe+1).contiguous().schedule_linear().src if c.op is Ops.CALL and c.src[0].op is Ops.SINK]
  resources=[AdrenoCompiler.unpack(to_program(c.src[0],Device['ADRENO'].renderer).to_elf().lib)[0] for c in probe_calls]
  if not any(r.num_uavs>=2 for r in resources): raise AssertionError('no native image load/store kernel was compiled')
  a=np.arange(7*16*4,dtype=np.float32).reshape(7,16,4)/8+seed/4
  x=Tensor(a,device='ADRENO').contiguous().realize()
  out=(x+1).contiguous()
  y=out.numpy()
  np.testing.assert_array_equal(y,a+1)
  np.testing.assert_array_equal(x.numpy(),a)
  return {'seed':seed,'input_sha256':sha(a),'output_sha256':sha(y)}

def replay(count:int):
  import numpy as np
  from tinygrad import Tensor, TinyJit
  @TinyJit
  def run(x): return (x+1).contiguous().realize()
  addresses=set()
  start=time.perf_counter()
  for i in range(count+2):
    a=np.arange(7*16*4,dtype=np.float32).reshape(7,16,4)/8+i/4
    x=Tensor(a,device='ADRENO').contiguous().realize()
    addresses.add(int(x.uop.buffer._buf.va_addr))
    np.testing.assert_array_equal(run(x).numpy(),a+1)
    np.testing.assert_array_equal(x.numpy(),a)
  if count and len(addresses)<2: raise AssertionError('replay did not rebind an image address')
  return {'replays':count,'unique_input_addresses':len(addresses),'seconds':time.perf_counter()-start}

def main():
  p=argparse.ArgumentParser()
  p.add_argument('--output',required=True)
  p.add_argument('--fresh-processes',type=int,default=100)
  p.add_argument('--replays',type=int,default=5000)
  p.add_argument('--worker',type=int)
  a=p.parse_args()
  if a.worker is not None:
    print(json.dumps(check_once(a.worker)),flush=True)
    return
  if a.fresh_processes<0 or a.replays<0: raise ValueError('gate counts must be nonnegative')
  report={'device':os.getenv('DEV'),'image':os.getenv('IMAGE'),'mesa_path':os.getenv('MESA_PATH'),
          'fresh_processes_requested':a.fresh_processes,'replays_requested':a.replays,'processes':[]}
  output=Path(a.output)
  output.parent.mkdir(parents=True,exist_ok=True)
  def save(): output.write_text(json.dumps(report,indent=2)+'\n')
  start=time.perf_counter()
  try:
    for i in range(a.fresh_processes):
      env=os.environ.copy()
      env['CACHEDB']=str(output.with_suffix(f'.process-{i}.db'))
      result=subprocess.run([sys.executable,__file__,'--output',str(output),'--worker',str(i)],env=env,
                            capture_output=True,text=True,timeout=45)
      if result.returncode: raise RuntimeError(f'fresh process {i} failed: {result.stderr[-3000:]}')
      report['processes'].append(json.loads(result.stdout.splitlines()[-1]))
      if (i+1)%10==0:
        save()
        print(f'fresh_processes {i+1}/{a.fresh_processes}',flush=True)
    report['replay']=replay(a.replays)
    report['passed']=True
  except BaseException as e:
    report['passed']=False
    report['error']=f'{type(e).__name__}: {e}'
    raise
  finally:
    report['seconds']=time.perf_counter()-start
    save()
    print(json.dumps({k:v for k,v in report.items() if k!='processes'}),flush=True)

if __name__=='__main__': main()
