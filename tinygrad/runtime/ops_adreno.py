from dataclasses import dataclass
from tinygrad.device import TinyELF
from tinygrad.renderer import Renderer
from tinygrad.renderer.adreno import AdrenoRenderer
from tinygrad.runtime.ops_qcom import QCOMDevice, QCOMProgram
from tinygrad.runtime.support.compiler_adreno import AdrenoCompiler

@dataclass(frozen=True)
class A830Properties:
  instr_cache_size:int = 127

@dataclass(frozen=True)
class A830Info:
  cs_shared_mem_size:int = 32768
  fibers_per_sp:int = 4096
  num_sp_cores:int = 6
  props:A830Properties = A830Properties()

class AdrenoProgram(QCOMProgram):
  def _decode_shader(self, obj:TinyELF):
    resources, signature, binary = AdrenoCompiler.unpack(obj.lib)
    expected = [[name,slot,dt.name,list(shape)] for name,slot,dt,shape in obj.signature]
    if signature != expected: raise ValueError('native Adreno artifact signature mismatch')
    if resources.arch != self.dev.renderer.target.arch: raise ValueError('native Adreno artifact device mismatch')
    return resources, b'', binary

class ADRENODevice(QCOMDevice):
  program_type = AdrenoProgram

  def _device_info(self, gpu_id:int):
    # Only the physical A830 KGSL revision is admitted by this initial contract.
    if self.chip_id != 0x44050001: raise RuntimeError(f'unsupported native Adreno chip {self.chip_id:#x}; expected A830 0x44050001')
    return 8, 830, A830Info()

  def _renderer_types(self) -> list[type[Renderer]]: return [AdrenoRenderer]
