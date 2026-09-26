"""Compile one controlled Conv with native ADRENO or QCOM:IR3 and save raw instruction disassembly."""
import argparse, contextlib, ctypes, hashlib, io, json
from pathlib import Path
import numpy as np
from tinygrad import Device, Tensor
from tinygrad.codegen import to_program
from tinygrad.uop.ops import Ops

def main():
  p=argparse.ArgumentParser()
  p.add_argument('--inputs',required=True,help='NPZ containing cpu_div and native_div tensors')
  p.add_argument('--weights',required=True,help='NPZ containing node-5 w and b tensors')
  p.add_argument('--output',required=True)
  a=p.parse_args()
  inp,wts=np.load(a.inputs),np.load(a.weights)
  x=Tensor(inp['cpu_div'],device=Device.DEFAULT).realize()
  w=Tensor(wts['w'],device=Device.DEFAULT).realize()
  b=Tensor(wts['b'],device=Device.DEFAULT).realize()
  out=x.conv2d(w,b,stride=(2,2),padding=(1,1,1,1))
  calls=[call for call in out.schedule_linear().src if call.op is Ops.CALL and call.src[0].op is Ops.SINK]
  report={'device':Device.DEFAULT,'input_sha256':hashlib.sha256(inp['cpu_div'].tobytes()).hexdigest(),
          'weight_sha256':hashlib.sha256(wts['w'].tobytes()).hexdigest(),'programs':[]}
  for i,call in enumerate(calls):
    program=to_program(call.src[0],Device[Device.DEFAULT].renderer)
    elf=program.to_elf()
    asm_path=Path(f'{a.output}-{i}.asm')
    if Device.DEFAULT.startswith('QCOM'):
      # Own the C FILE* directly. Termux fdsan rejects fdopen() on a Python-owned
      # descriptor, which the generic IR3 disassembly helper otherwise uses.
      from tinygrad.runtime.autogen import mesa, libc
      from tinygrad.runtime.support.compiler_mesa import IR3Compiler
      _,_,binary=IR3Compiler.unpack_lib(elf.lib)
      fp=libc.fopen(str(asm_path).encode(),b'w')
      if not fp: raise OSError(f'cannot open {asm_path}')
      try: mesa.ir3_isa_disasm(binary,len(binary),ctypes.cast(fp,ctypes.POINTER(mesa.struct__IO_FILE)),
                              mesa.struct_isa_decode_options(830,True,0,True))
      finally: libc.fclose(fp)
    else:
      output=io.StringIO()
      with contextlib.redirect_stdout(output): Device[Device.DEFAULT].compiler.disassemble(elf.lib)
      asm_path.write_text(output.getvalue())
    report['programs'].append({'name':elf.name,'artifact_sha256':hashlib.sha256(elf.lib).hexdigest(),
                               'artifact_bytes':len(elf.lib),'disassembly_lines':len(asm_path.read_text().splitlines()),
                               'signature':[(name,slot,str(dt),list(shape)) for name,slot,dt,shape in elf.signature]})
  Path(a.output+'.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)

if __name__=='__main__': main()
