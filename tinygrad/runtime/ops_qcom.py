from __future__ import annotations
import os, ctypes, functools, mmap, struct, array, math, sys, weakref, contextlib, re
from dataclasses import dataclass
assert sys.platform != 'win32'
from typing import Any, cast
from tinygrad.device import BufferSpec, Device
from tinygrad.runtime.support.hcq import HCQBuffer, HWQueue, HCQProgram, HCQCompiled, HCQAllocatorBase, HCQSignal, HCQArgsState, BumpAllocator
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.runtime.autogen import kgsl, mesa
from tinygrad.renderer.cstyle import QCOMCLRenderer
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.helpers import getenv, mv_address, to_mv, round_up, data64_le, ceildiv, prod, cpu_profile, lo32, suppress_finalizing, is_image_shape
from tinygrad.helpers import next_power2, flatten, PROFILE, IMAGE
from tinygrad.dtype import dtypes, DType
from tinygrad.runtime.support.system import System
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

BUFTYPE_BUF, BUFTYPE_TEX, BUFTYPE_IBO = 0, 1, 2

def _decode_chip_id(chip_id: int) -> tuple[int, int, str]:
  """Return (gpu_id_small, gen, arch_str) for a KGSL chip_id."""
  top = (chip_id >> 24) & 0xFF
  if top < 0x10:
    gpu_id_small = top * 100 + ((chip_id >> 16) & 0xFF) * 10
    return gpu_id_small, gpu_id_small // 100, f"a{gpu_id_small}"
  dev_id = mesa.struct_fd_dev_id(0, chip_id)
  name = mesa.fd_dev_name(ctypes.byref(dev_id))
  if name:
    s = ctypes.string_at(name).decode()
    if (m := re.search(r"(\d{3,})", s)):
      gpu_id_small = int(m.group(1))
      return gpu_id_small, gpu_id_small // 100, f"a{gpu_id_small}"
  raise RuntimeError(f"Unknown Adreno chip_id={chip_id:#x}")

@functools.cache
def dcache_flush():
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.codegen import to_program
  buf, n = UOp.param(0, dtypes.uint8, shape=(1,)), UOp.param(1, dtypes.int, shape=(1,), name="n", addrspace=None)
  i = UOp.range(n, 0, dtype=dtypes.int)
  flush = UOp(Ops.CUSTOM, src=(buf.index(i * 64),), arg='__asm__ volatile("dc cvac, %0" :: "r"({0}) : "memory");')
  sink = UOp.sink(flush.end(i), UOp(Ops.CUSTOM, arg='__asm__ volatile("dsb sy" ::: "memory");'), arg=KernelInfo(name="dcache_flush"))
  prg = to_program(UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())))), Device["CPU"].renderer)
  return Device["CPU"].runtime(prg.arg.function_name, prg.src[3].arg)

#Parse C-style defines: <regname>_<field_x>__SHIFT and <regname>_<field_y>__MASK from the adreno module into the following format:
# qreg.<regname>(<field_x>=..., <field_y>=..., ..., <field_n>=...)
def _qreg_exec(__reg, __val=0, **kwargs):
  for k, v in kwargs.items():
    reg_name = f"{__reg[4:]}_{k.removeprefix('_').upper()}"
    __val |= (getattr(mesa, reg_name) if v else 0) if type(v) is bool else (v << getattr(mesa, f'{reg_name}__SHIFT'))
  return __val
qreg: Any = type("QREG", (object,), {name[4:].lower(): functools.partial(_qreg_exec, name) for name in mesa.__dict__.keys() if name[:4] == 'REG_'})

def ctz(v): return (v & -v).bit_length() - 1

@dataclass(frozen=True)
class QCOMImageLayout:
  height: int
  width: int
  row_pitch: int
  array_pitch: int
  pitch_alignment: int
  @property
  def pitchalign(self): return ctz(self.pitch_alignment) - 6
  @property
  def size(self): return self.array_pitch

def qcom_image_layout(dtype:DType, shape:tuple[int, ...], row_pitch:int|None=None) -> QCOMImageLayout:
  """Linear RGBA layout matching fdl6_layout_image's A6xx+ explicit-layout requirements."""
  if dtype not in (dtypes.half, dtypes.float): raise ValueError(f"unsupported QCOM image dtype {dtype}")
  if not is_image_shape(shape): raise ValueError(f"QCOM images require HxWx4 shape, got {shape}")
  height, width, _ = shape
  if not (0 < height <= 0x7fff and 0 < width <= 0x7fff): raise ValueError(f"QCOM image dimensions out of range: {shape}")
  bytes_per_pixel = 4 * dtype.itemsize
  # Mesa fdl6_layout_image aligns linear rows to 16 pixels: 128B for RGBA16F and 256B for RGBA32F.
  pitch_alignment = 16 * bytes_per_pixel
  if row_pitch is None: row_pitch = round_up(width * bytes_per_pixel, pitch_alignment)
  if row_pitch < width * bytes_per_pixel or row_pitch % pitch_alignment:
    raise ValueError(f"invalid QCOM image row pitch {row_pitch} for {shape} {dtype}; alignment is {pitch_alignment}")
  # fdl6_layout_image pads the last linear level to four rows. ARRAY_SLICE_OFFSET is unused for a plain 2D view, but keep it Mesa-exact.
  array_pitch = row_pitch * round_up(height, 4)
  return QCOMImageLayout(height, width, row_pitch, array_pitch, pitch_alignment)

def qcom_image_descriptor(gen:int, dtype:DType, shape:tuple[int, ...], addr, storage:bool=False,
                          row_pitch:int|None=None) -> list:
  """Build one bindful sampled-texture or storage-image descriptor from Mesa's Gen6/Gen8 definitions."""
  layout = qcom_image_layout(dtype, shape, row_pitch)
  if isinstance(addr, int) and addr & (0x3f if gen >= 8 else 0x1f): raise ValueError(f"unaligned QCOM image address {addr:#x}")
  fmt = mesa.FMT6_32_32_32_32_FLOAT if dtype == dtypes.float else mesa.FMT6_16_16_16_16_FLOAT
  if gen in (6, 7):
    # A6xx/A7xx share the descriptor layout. Storage-image swizzles are ignored by image stores.
    desc = [(fmt << 22) if storage else 0x8 | (1 << 7) | (2 << 10) | (3 << 13) | (fmt << 22),
            layout.width | (layout.height << 15),
            layout.pitchalign | (layout.row_pitch << 7) | (mesa.A6XX_TEX_2D << 29),
            0, *data64_le(addr), 0x40000000, 13]
    return desc + [0] * (16 - len(desc))
  if gen != 8: raise ValueError(f"unsupported QCOM image descriptor generation {gen}")

  # Mesa a8xx_descriptors.xml / fdl/fd6_view.cc. Gen8 stores pitch in bits and moves RGBA swizzles to dword 3.
  tex_line_offset = layout.row_pitch * 8
  if tex_line_offset >= 1 << 24: raise ValueError(f"QCOM image row pitch is too large: {layout.row_pitch}")
  desc = [
    addr & 0xffffffc0,
    ((addr >> 32) & 0x1ffff) | (mesa.A6XX_TEX_2D << 17) | (1 << 20),
    layout.width | (layout.height << 15),
    fmt | (3 << 10) | (4 << 13) | (5 << 16) | (6 << 19),
    0, 0,
    tex_line_offset | (layout.pitchalign << 24),
    (layout.array_pitch >> 12) & 0x7fffff,
  ]
  return desc + [0] * (16 - len(desc))

def qcom_sampler_descriptor(gen:int) -> list[int]:
  """Nearest, unnormalized, clamp-to-zero-border sampler for the generation's 16-byte layout."""
  clamp = mesa.A6XX_TEX_CLAMP_TO_BORDER
  if gen in (6, 7):
    return [qreg.a6xx_tex_samp_0(wrap_s=clamp, wrap_t=clamp, wrap_r=clamp),
            qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0]
  if gen == 8:
    return [(clamp << 6) | (clamp << 9) | (clamp << 12), 1 << 31, 1, 0]
  raise ValueError(f"unsupported QCOM sampler generation {gen}")

def parity(val: int):
  for i in range(4,1,-1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int): return mesa.CP_TYPE7_PKT | cnt & 0x3FFF | parity(cnt) << 15 | (opcode & 0x7F) << 16 | parity(opcode) << 23

def pkt4_hdr(reg: int, cnt: int): return mesa.CP_TYPE4_PKT | cnt & 0x7F | parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | parity(reg) << 27

def _read_lib(lib, off) -> int: return struct.unpack("I", lib[off:off+4])[0]

class QCOMSignal(HCQSignal):
  def __init__(self, *args, **kwargs): super().__init__(*args, **{**kwargs, 'timestamp_divider': 19.2})

  def wait(self, value:int, timeout:int|None=None):
    # The timeline value write can become CPU-visible before all effects of the corresponding KGSL submission are visible.
    # Always wait for the kernel driver's completion timestamp instead of skipping _sleep when the value already compares true.
    if self.is_timeline and self.owner is not None and self.owner.last_cmd:
      kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.owner.fd, context_id=self.owner.ctx, timestamp=self.owner.last_cmd,
                                                   timeout=0xffffffff if timeout is None else timeout)
    return super().wait(value, timeout)

  def _sleep(self, time_spent_since_last_sleep_ms:int):
    # Sleep only for timeline signals. Do it immediately to free cpu.
    if self.is_timeline and self.owner is not None:
      kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.owner.fd, context_id=self.owner.ctx, timestamp=self.owner.last_cmd, timeout=0xffffffff)

class QCOMComputeQueue(HWQueue):
  def __init__(self, dev:QCOMDevice):
    self.dev = dev
    super().__init__()

  @suppress_finalizing
  def __del__(self):
    if self.binded_device is not None: self.binded_device.allocator.free(self.hw_page, self.hw_page.size, BufferSpec(cpu_access=True, nolru=True))

  def cmd(self, opcode: int, *vals: int): self.q(pkt7_hdr(opcode, len(vals)), *vals)

  def reg(self, reg: int, *vals: int): self.q(pkt4_hdr(reg, len(vals)), *vals)

  def _cache_flush(self, write_back=True, invalidate=False, sync=True, memsync=False):
    if self.dev.gen == 6:
      if write_back: self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS),
                              *data64_le(self.dev.dummy_addr), 0)
      if invalidate: self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_INVALIDATE))
    else:
      if write_back: self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write7_0(event=mesa.CACHE_FLUSH7, write_enabled=False))
      if invalidate: self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write7_0(event=mesa.CACHE_INVALIDATE7, write_enabled=False))
    if memsync: self.cmd(mesa.CP_WAIT_MEM_WRITES)
    if sync: self.cmd(mesa.CP_WAIT_FOR_IDLE)

  def memory_barrier(self):
    self._cache_flush(write_back=True, invalidate=True, sync=True, memsync=True)
    return self

  def signal(self, signal:QCOMSignal, value=0):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    if self.dev.gen == 6:
      self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS),
               *data64_le(signal.value_addr), lo32(value))
    else:
      self.cmd(mesa.CP_EVENT_WRITE,
               qreg.cp_event_write7_0(event=mesa.CACHE_FLUSH7, write_src=mesa.EV_WRITE_USER_32B,
                                      write_dst=mesa.EV_DST_RAM, write_enabled=True),
               *data64_le(signal.value_addr), lo32(value))
    self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    return self

  def timestamp(self, signal:QCOMSignal):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    reg = mesa.REG_A8XX_CP_ALWAYS_ON_COUNTER if self.dev.gen == 8 else mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER
    self.cmd(mesa.CP_REG_TO_MEM, qreg.cp_reg_to_mem_0(reg=reg, cnt=2, _64b=True),*data64_le(signal.timestamp_addr))
    return self

  def wait(self, signal:QCOMSignal, value=0):
    self.cmd(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),*data64_le(signal.value_addr),
             qreg.cp_wait_reg_mem_3(ref=value&0xFFFFFFFF), qreg.cp_wait_reg_mem_4(mask=0xFFFFFFFF), qreg.cp_wait_reg_mem_5(delay_loop_cycles=32))
    return self

  def _build_gpu_command(self, dev:QCOMDevice, hw_addr=None):
    to_mv((hw_page_addr:=hw_addr or dev.cmd_buf_allocator.alloc(len(self._q) * 4)), len(self._q) * 4).cast('I')[:] = array.array('I', self._q)
    obj = kgsl.struct_kgsl_command_object(gpuaddr=hw_page_addr, size=len(self._q) * 4, flags=kgsl.KGSL_CMDLIST_IB)
    submit_req = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), numcmds=1, context_id=dev.ctx,
                                              cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object))
    return submit_req, obj

  def bind(self, dev:QCOMDevice):
    self.binded_device = dev
    self.hw_page = dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True))
    self.submit_req, self.obj = self._build_gpu_command(self.binded_device, self.hw_page.va_addr)
    # From now on, the queue is on the device for faster submission.
    self._q = to_mv(self.obj.gpuaddr, len(self._q) * 4).cast("I")

  def _submit(self, dev:QCOMDevice):
    if self.binded_device == dev: submit_req, obj = self.submit_req, self.obj
    else: submit_req, obj = self._build_gpu_command(dev)
    dev.last_cmd = kgsl.IOCTL_KGSL_GPU_COMMAND(dev.fd, __payload=submit_req).timestamp

  def exec(self, prg:QCOMProgram, args_state:QCOMArgsState, global_size, local_size):
    self.bind_args_state(args_state)

    def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x
    global_size_mp = [cast_int(g*l) for g,l in zip(global_size, local_size)]

    # Image descriptors start at byte 2048, outside the constant range. Buffer kernels retain the original 1 KiB constant load.
    if prg.tex_cnt or prg.ibo_cnt:
      const_dwords, const_load_units, const_ram_mode = prg.constlen, ceildiv(prg.constlen, 4), prg.const_ram_mode
    else:
      const_dwords, const_load_units, const_ram_mode = 256, 256, mesa.CONSTLEN_256

    self.cmd(mesa.CP_SET_MARKER, (qreg.a8xx_cp_set_marker_0 if self.dev.gen == 8 else qreg.a6xx_cp_set_marker_0)(mode=mesa.RM6_COMPUTE))
    update_cntl = (qreg.a6xx_sp_update_cntl(cs_state=True, cs_uav=True) if self.dev.gen == 6 else
                   qreg.a7xx_sp_update_cntl(cs_state=True, cs_uav=True) if self.dev.gen == 7 else
                   qreg.a8xx_sp_update_cntl(vs_state=True, hs_state=True, ds_state=True, gs_state=True, fs_state=True, cs_state=True))
    self.reg(self.dev.reg_sp_update_cntl, update_cntl)
    self.reg(self.dev.reg_sp_update_cntl, 0x0)

    if self.dev.gen == 6:
      self.reg(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(0x80))
      self.reg(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(0x40))
      self.reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL if prg.NIR else mesa.ISAMMODE_CL,
                                                                  constant_demotion_enable=prg.NIR))
      self.reg(mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, qreg.a6xx_sp_perfctr_shader_mask(cs=True))
      self.reg(mesa.REG_A6XX_TPL1_MODE_CNTL, qreg.a6xx_tpl1_mode_cntl(isammode=mesa.ISAMMODE_GL if prg.NIR else mesa.ISAMMODE_CL))
      self.reg(mesa.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)
    else:
      self.reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL,
                                                                  constant_demotion_enable=True))
    self.cmd(mesa.CP_WAIT_FOR_IDLE)

    if self.dev.gen == 6:
      self.reg(self.dev.reg_sp_cs_ndrange_0,
               qreg.a6xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
               global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, 0xccc0cf,
               0xfc | qreg.a6xx_sp_cs_wge_cntl(threadsize=prg.threadsize),
               cast_int(global_size[0], ceil=True), cast_int(global_size[1], ceil=True), cast_int(global_size[2], ceil=True))
    else:
      tile_height = 3 if local_size[1] % 8 == 0 else 5 if local_size[1] % 4 == 0 else 9 if local_size[1] % 2 == 0 else 17
      self.reg(self.dev.reg_sp_cs_ndrange_0,
               qreg.a7xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
               global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0,
               qreg.a7xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=prg.threadsize,
                                        workgrouprastorderzfirsten=True, wgtilewidth=4, wgtileheight=tile_height),
               0xfc000000 | local_size[0], local_size[1], local_size[2],
               qreg.a7xx_sp_cs_ndrange_7(localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1))

    self.reg(mesa.REG_A6XX_SP_CS_CNTL_0,
             qreg.a6xx_sp_cs_cntl_0(threadsize=prg.threadsize, halfregfootprint=prg.hregs, fullregfootprint=prg.fregs,
                                   branchstack=prg.brnchstck, earlypreamble=prg.early_preamble, mergedregs=prg.mergedregs),
             qreg.a6xx_sp_cs_cntl_1(constantrammode=const_ram_mode, shared_size=prg.shared_size),
             0, prg.prg_offset, *data64_le(prg.lib_gpu.va_addr),
             qreg.a6xx_sp_cs_pvt_mem_param(memsizeperitem=prg.pvtmem_size_per_item), *data64_le(prg.dev._stack.va_addr),
             qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=prg.pvtmem_size_total, perwavememlayout=prg.pvtmem_per_wave))

    if self.dev.gen >= 7: self.reg(mesa.REG_A7XX_SP_CS_VGS_CNTL, 0)

    if prg.NIR and prg.wgsz != 0xfc: to_mv(int(args_state.buf.va_addr) + prg.wgsz * 4, 12)[:] = struct.pack("III", *local_size)
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER, num_unit=const_load_units),
             *data64_le(args_state.buf.va_addr))
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER, num_unit=round_up(prg.image_size, 128) // 128),
             *data64_le(prg.lib_gpu.va_addr))

    if self.dev.gen == 6:
      self.reg(self.dev.reg_sp_reg_prog_id_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc,
               qreg.a6xx_sp_cs_const_config(constlen=const_dwords, enabled=True))
    else:
      self.reg(self.dev.reg_sp_reg_prog_id_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc00)
      self.reg(self.dev.reg_sp_cs_const_config, qreg.a7xx_sp_cs_const_config(constlen=const_dwords, enabled=True))

    self.reg(mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, qreg.a6xx_sp_cs_pvt_mem_stack_offset(prg.hw_stack_offset))
    self.reg(mesa.REG_A6XX_SP_CS_INSTR_SIZE, qreg.a6xx_sp_cs_instr_size(prg.instrlen))

    if prg.samp_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_TEX, num_unit=args_state.prg.samp_cnt),
               *data64_le(args_state.buf.va_addr + args_state.prg.samp_off))
      self.reg(mesa.REG_A6XX_SP_CS_SAMPLER_BASE, *data64_le(args_state.buf.va_addr + args_state.prg.samp_off))
      self.reg(mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE, *data64_le(prg.dev.border_color_buf.va_addr))

    if prg.tex_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_TEX, num_unit=min(16, args_state.prg.tex_cnt)),
               *data64_le(args_state.buf.va_addr + args_state.prg.tex_off))
      self.reg(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, *data64_le(args_state.buf.va_addr + args_state.prg.tex_off))

    if prg.ibo_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST6_UAV, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_SHADER, num_unit=args_state.prg.ibo_cnt),
               *data64_le(args_state.buf.va_addr + args_state.prg.ibo_off))
      self.reg(self.dev.reg_sp_cs_uav_base, *data64_le(args_state.buf.va_addr + args_state.prg.ibo_off))

    # These are descriptor counts, despite their historical TSIZE/USIZE names. A8xx does not infer them from CP_LOAD_STATE6.
    self.reg(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(prg.tex_cnt))
    self.reg(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(prg.ibo_cnt))
    self.reg(mesa.REG_A6XX_SP_CS_CONFIG,
             qreg.a6xx_sp_cs_config(enabled=True, nsamp=args_state.prg.samp_cnt, ntex=args_state.prg.tex_cnt, nuav=args_state.prg.ibo_cnt))

    if self.dev.gen == 6:
      if prg.NIR:
        self.reg(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
                 qreg.a6xx_sp_cs_const_config_0(wgidconstid=prg.wgid, wgsizeconstid=prg.wgsz, wgoffsetconstid=0xfc, localidregid=prg.lid),
                 qreg.a6xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=mesa.THREAD64))
    else:
      self.reg(self.dev.reg_sp_ps_wave_cntl, qreg.a7xx_sp_ps_wave_cntl(threadsize=mesa.THREAD64))
      self.reg(self.dev.reg_sp_cs_wie_cntl_1,
               qreg.a7xx_sp_cs_wie_cntl_1(linearlocalidregid=0xfc, threadsize=prg.threadsize,
                                          workitemrastorder=mesa.WORKITEMRASTORDER_LINEAR))
      if prg.NIR:
        self.reg(mesa.REG_A6XX_SP_CS_WIE_CNTL_0,
                 qreg.a6xx_sp_cs_wie_cntl_0(wgidconstid=prg.wgid, wgsizeconstid=prg.wgsz, wgoffsetconstid=0xfc, localidregid=prg.lid))

    if prg.NIR:
      self.cmd(mesa.CP_EXEC_CS, 0,
               qreg.cp_exec_cs_1(ngroups_x=global_size[0]), qreg.cp_exec_cs_2(ngroups_y=global_size[1]), qreg.cp_exec_cs_3(_ngroups_z=global_size[2]))
    else: self.cmd(mesa.CP_RUN_OPENCL, 0)

    self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    return self

class QCOMArgsState(HCQArgsState):
  def __init__(self, buf:HCQBuffer, prg:QCOMProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    super().__init__(buf, prg, bufs, vals=vals)
    ctypes.memset(int(self.buf.va_addr), 0, prg.kernargs_alloc_size)

    ubos = [b for i,b in enumerate(bufs) for _,dt,shape in prg.buf_dtypes[i] if not is_image_shape(shape)]
    uavs = [(dt,shape,b) for i,b in enumerate(bufs) for _,dt,shape in prg.buf_dtypes[i] if is_image_shape(shape)]
    # NIR can reorder images to different texture slots
    ibos, texs = uavs[:prg.ibo_cnt], [uavs[prg.ibo_cnt + (prg.tex_to_image[i] if prg.NIR else i)] for i in range(prg.tex_cnt)]
    for cnst_val,cnst_off,cnst_sz in prg.consts_info:
      to_mv(cast(int, self.buf.va_addr) + cnst_off, cnst_sz)[:] = cnst_val.to_bytes(cnst_sz, byteorder='little')

    if prg.samp_cnt > 0: to_mv(int(self.buf.va_addr) + prg.samp_off, len(prg.samplers) * 4).cast('I')[:] = array.array('I', prg.samplers)
    if prg.NIR:
      self.bind_sints_to_buf(*[b.va_addr for b in ubos], buf=self.buf, fmt='Q', offset=prg.buf_off)
      self.bind_sints_to_buf(*vals, buf=self.buf, fmt='I', offset=prg.buf_off + len(ubos) * 8)
    else:
      for i, b in enumerate(ubos): self.bind_sints_to_buf(b.va_addr, buf=self.buf, fmt='Q', offset=prg.buf_offs[i])
      for i, v in enumerate(vals): self.bind_sints_to_buf(v, buf=self.buf, fmt='I', offset=prg.buf_offs[i+len(ubos)])

    def _tex(b, ibo=False):
      imgdt, shape, buf = b
      desc = qcom_image_descriptor(prg.dev.gen, imgdt, shape, buf.va_addr, storage=ibo,
                                   row_pitch=shape[1] * 4 * imgdt.itemsize)
      return desc

    self.bind_sints_to_buf(*flatten(map(_tex, texs)), buf=self.buf, fmt='I', offset=prg.tex_off)
    self.bind_sints_to_buf(*flatten(map(functools.partial(_tex, ibo=True), ibos)), buf=self.buf, fmt='I', offset=prg.ibo_off)

class QCOMProgram(HCQProgram):
  def __init__(self, dev: QCOMDevice, name: str, lib: bytes, buf_dtypes=[], **kwargs):
    self.dev: QCOMDevice = dev
    self.buf_dtypes, self.name, self.NIR = buf_dtypes, name, isinstance(dev.renderer, IR3Renderer)

    if self.NIR:
      from tinygrad.runtime.support.compiler_mesa import IR3Compiler
      v, cs, imm_vals, self.image = IR3Compiler.unpack_lib(lib)
      self.prg_offset, self.brnchstck, self.image_size = 0, round_up(v.branchstack, 2), v.info.size
      self.pvtmem, self.pvtmem_per_wave, self.shmem = v.pvtmem_size, v.pvtmem_per_wave, v.shared_size
      self.fibers_per_sp, self.num_sp_cores = dev.dev_info.fibers_per_sp, dev.dev_info.num_sp_cores
      self.early_preamble, self.mergedregs = v.early_preamble, v.mergedregs
      self.instrlen = v.instrlen
      self.threadsize = mesa.THREAD128 if v.info.double_threadsize else mesa.THREAD64
      self.constlen = round_up(v.constlen, 4)
      self.const_ram_mode = (mesa.CONSTLEN_512 if v.constlen > 256 else mesa.CONSTLEN_256 if v.constlen > 192 else
                             mesa.CONSTLEN_192 if v.constlen > 128 else mesa.CONSTLEN_128)
      self.wgsz = alloc.offset_vec4 * 4 + 8 if (alloc:=cs.allocs.consts[mesa.IR3_CONST_ALLOC_DRIVER_PARAMS]).size_vec4 else 0xfc

      self.wgid, self.lid = v.cs.work_group_id, v.cs.local_invocation_id # register ids
      self.buf_off, imm_off = cs.ubo_state.range[0].offset, cs.allocs.max_const_offset_vec4 * 16
      self.consts_info = [(struct.unpack_from("<I", imm_vals, i)[0], imm_off + i, 4) for i in range(0, len(imm_vals), 4)]

      # IR3 records sampled-image remapping separately from the total UAV count.
      self.samp_cnt, self.tex_cnt, self.ibo_cnt = (nt:=v.image_mapping.num_tex), nt, v.num_uavs - nt
      self.tex_to_image = v.image_mapping.tex_to_image[:]
      # IR3 emits one sampler slot for every sampled texture.
      self.samplers = qcom_sampler_descriptor(dev.gen) * self.samp_cnt

      self.tex_off, self.ibo_off, self.samp_off = 2048, 2048 + 0x40 * self.tex_cnt, 2048 + 0x40 * (self.tex_cnt + self.ibo_cnt)
      self.fregs, self.hregs = v.info.max_reg + 1, v.info.max_half_reg + 1
    else:
      self.early_preamble = self.mergedregs = False
      self.pvtmem_per_wave = False
      self.constlen, self.const_ram_mode = 1024 // 4, mesa.CONSTLEN_256
      self._parse_lib(lib)
      self.instrlen, self.threadsize = self.image_size // 4, mesa.THREAD64

    self.lib_gpu: HCQBuffer = self.dev.allocator.alloc(self.image_size, buf_spec:=BufferSpec(cpu_access=True, nolru=True))
    to_mv(self.lib_gpu.va_addr, self.image_size)[:] = self.image

    self.pvtmem_size_per_item: int = round_up(self.pvtmem, 512) >> 9
    if self.NIR:
      per_sp_size = round_up(self.pvtmem * self.fibers_per_sp, 1 << 12)
      self.pvtmem_size_total, self.hw_stack_offset = per_sp_size >> 12, per_sp_size >> 11
      stack_size = per_sp_size * self.num_sp_cores
    else:
      self.pvtmem_size_total = self.pvtmem_size_per_item * 128 * 2
      self.hw_stack_offset = round_up(next_power2(round_up(self.pvtmem, 512)) * 128 * 16, 0x1000)
      stack_size = self.hw_stack_offset * 4
    self.shared_size: int = max(1, (self.shmem - 1) // 1024)
    self.max_threads = min(1024, ((384 * 32) // (max(1, (self.fregs + round_up(self.hregs, 2) // 2)) * 128)) * 128)
    dev._ensure_stack_size(max(stack_size, 0x1000))

    kernargs_alloc_size = round_up(2048 + (self.tex_cnt + self.ibo_cnt) * 0x40 + len(self.samplers) * 4, 0x100)
    super().__init__(QCOMArgsState, self.dev, self.name, kernargs_alloc_size=kernargs_alloc_size)
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int|None, ...]=(), wait=False, **kw):
    if self.max_threads < prod(local_size): raise RuntimeError("Too many resources requested for launch")
    if any(g*l>mx for g,l,mx in zip(global_size, local_size, [65536, 65536, 65536])) and any(l>mx for l,mx in zip(local_size, [1024, 1024, 1024])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    return super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)

  def _parse_lib(self, lib):
    # Extract image binary
    self.image_size = _read_lib(lib, 0x100)
    self.image = lib[(image_offset:=_read_lib(lib, 0xc0)):image_offset+self.image_size]

    # Parse image descriptors
    image_desc_off = _read_lib(lib, 0x110)
    self.prg_offset, self.brnchstck = _read_lib(lib, image_desc_off+0xc4), _read_lib(lib, image_desc_off+0x108) // 2
    self.pvtmem, self.shmem = _read_lib(lib, image_desc_off+0xc8), _read_lib(lib, image_desc_off+0xd8)

    # Fill up constants and buffers info
    self.consts_info = []

    # Collect sampler info.
    self.samp_cnt = samp_cnt_in_file = _read_lib(lib, image_desc_off + 0xdc)
    assert self.samp_cnt <= 1, "Up to one sampler supported"
    if self.samp_cnt:
      self.samp_cnt += 1
      self.samplers = qcom_sampler_descriptor(self.dev.gen) + [0, 0, 0, 0]
    else: self.samplers = []

    # Collect kernel arguments (buffers) info.
    bdoff, binfos = round_up(image_desc_off + 0x158 + len(self.name), 4) + 8 * samp_cnt_in_file, []
    while bdoff + 32 <= len(lib):
      length, _, _, offset_words, _, _, _, typ = struct.unpack("8I", lib[bdoff:bdoff+32])
      if length == 0: break
      binfos.append((offset_words * 4, typ))
      bdoff += length
    self.buf_offs = [off for off,typ in binfos if typ not in {BUFTYPE_TEX, BUFTYPE_IBO}]

    # Setting correct offsets to textures/ibos.
    self.tex_cnt, self.ibo_cnt = sum(typ is BUFTYPE_TEX for _,typ in binfos), sum(typ is BUFTYPE_IBO for _,typ in binfos)
    self.ibo_off, self.tex_off, self.samp_off = 2048, 2048 + 0x40 * self.ibo_cnt, 2048 + 0x40 * self.tex_cnt + 0x40 * self.ibo_cnt

    if _read_lib(lib, 0xb0) != 0: # check if we have constants.
      cdoff = _read_lib(lib, 0xac)
      while cdoff + 40 <= image_offset:
        cnst, offset_words, _, is32 = struct.unpack("I", lib[cdoff:cdoff+4])[0], *struct.unpack("III", lib[cdoff+16:cdoff+28])
        self.consts_info.append((cnst, offset_words * (sz_bytes:=(2 << is32)), sz_bytes))
        cdoff += 40

    # Registers info
    reg_desc_off = _read_lib(lib, 0x34)
    self.fregs, self.hregs = _read_lib(lib, reg_desc_off + 0x14), _read_lib(lib, reg_desc_off + 0x18)

class QCOMAllocator(HCQAllocatorBase):
  def _alloc(self, size:int, opts:BufferSpec) -> HCQBuffer:
    # Command streams, kernargs, signals, and other cpu_access allocations are rewritten directly through their mmap.
    # Keep those mappings uncached. Ordinary tensor storage stays writeback cached and is maintained explicitly in copyin/copyout.
    return self.dev._gpu_map(opts.external_ptr, size) if opts.external_ptr else \
      self.dev._gpu_alloc(size, uncached=opts.uncached or opts.cpu_access)

  def _map(self, buf): return buf  # QCOM:IR3 and QCOM share the same GPU VA space

  def _sync_cache(self, buf:HCQBuffer, op:int):
    alloc, is_gpuobj = buf.meta
    if not is_gpuobj or alloc is None or \
       alloc.flags & kgsl.KGSL_CACHEMODE_MASK == flag("KGSL_CACHEMODE", kgsl.KGSL_CACHEMODE_UNCACHED): return
    kgsl.IOCTL_KGSL_GPUMEM_SYNC_CACHE(self.dev.fd, id=alloc.id, op=op | kgsl.KGSL_GPUMEM_CACHE_RANGE,
                                      offset=int(buf.va_addr) - int(buf.base.va_addr), length=buf.size)

  def _copyin(self, dest:HCQBuffer, src:memoryview):
    self.dev.synchronize()
    with cpu_profile(f"TINY -> {self.dev.device}", f"{self.dev.device}:COPY"): ctypes.memmove(dest.cpu_view().addr, mv_address(src), src.nbytes)
    self._sync_cache(dest, kgsl.KGSL_GPUMEM_CACHE_TO_GPU)

  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self.dev.synchronize()
    self._sync_cache(src, kgsl.KGSL_GPUMEM_CACHE_FROM_GPU)
    with cpu_profile(f"{self.dev.device} -> TINY", f"{self.dev.device}:COPY"): ctypes.memmove(mv_address(dest), src.cpu_view().addr, src.size)

  def _as_buffer(self, src:HCQBuffer) -> memoryview:
    self.dev.synchronize()
    self._sync_cache(src, kgsl.KGSL_GPUMEM_CACHE_FROM_GPU)
    return to_mv(src.cpu_view().addr, src.size)

  def _do_free(self, opaque, options:BufferSpec): self.dev._gpu_free(opaque)

def flag(nm, val): return (val << getattr(kgsl, f"{nm}_SHIFT")) & getattr(kgsl, f"{nm}_MASK")

class QCOMDevice(HCQCompiled):
  def _wait_for_command_arena(self):
    # Command allocation occurs after next_timeline(), so synchronize() would wait for the command that has not been submitted yet.
    # last_cmd is the KGSL timestamp of the most recent completed submission boundary and is exactly the lifetime protecting older PM4.
    if self.last_cmd:
      kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.fd, context_id=self.ctx, timestamp=self.last_cmd, timeout=0xffffffff)

  def __init__(self, device:str=""):
    self.fd = FileIOInterface('/dev/kgsl-3d0', os.O_RDWR)
    self.dummy_addr = int(self._gpu_alloc(0x1000).va_addr)

    flags = kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC \
      | flag("KGSL_CONTEXT_PRIORITY", getenv("QCOM_PRIORITY", 8)) | flag("KGSL_CONTEXT_PREEMPT_STYLE", kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN)
    self.ctx = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.fd, flags=flags).drawctxt_id

    self.cmd_buf = self._gpu_alloc(16 << 20, uncached=True)
    # PM4 remains live until KGSL completes the corresponding timeline. Do not overwrite a wrapped arena while older submissions can still read it.
    self.cmd_buf_allocator = BumpAllocator(size=self.cmd_buf.size, base=int(self.cmd_buf.va_addr), wrap=True,
                                           wrap_callback=self._wait_for_command_arena)

    self.border_color_buf = self._gpu_alloc(0x1000, fill_zeroes=True)

    self.last_cmd:int = 0

    # Set max power
    struct.pack_into('IIQQ', pwr:=memoryview(bytearray(0x18)), 0, 1, self.ctx, mv_address(_:=memoryview(array.array('I', [1]))), 4)
    kgsl.IOCTL_KGSL_SETPROPERTY(self.fd, type=kgsl.KGSL_PROP_PWR_CONSTRAINT, value=mv_address(pwr), sizebytes=pwr.nbytes)

    # Load info about qcom device
    info = kgsl.struct_kgsl_devinfo()
    kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.fd, type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    self.chip_id:int = info.chip_id
    self.gpu_id_small, self.gen, self.arch_str = _decode_chip_id(info.chip_id)
    self.gpu_id = (self.gen, self.gpu_id_small // 100, self.gpu_id_small % 100)
    dev_id = mesa.struct_fd_dev_id(self.gpu_id_small, self.chip_id)
    self.dev_info = mesa.fd_dev_info(ctypes.byref(dev_id))

    if self.gen == 6:
      self.reg_sp_update_cntl      = mesa.REG_A6XX_SP_UPDATE_CNTL
      self.reg_sp_reg_prog_id_0    = mesa.REG_A6XX_SP_REG_PROG_ID_0
      self.reg_sp_cs_const_config  = mesa.REG_A6XX_SP_CS_CONST_CONFIG
      self.reg_sp_cs_ndrange_0     = mesa.REG_A6XX_SP_CS_NDRANGE_0
      self.reg_sp_cs_uav_base      = mesa.REG_A6XX_SP_CS_UAV_BASE
      self.reg_sp_ps_wave_cntl     = mesa.REG_A6XX_SP_PS_WAVE_CNTL
      self.reg_sp_cs_wie_cntl_1    = mesa.REG_A6XX_SP_CS_WIE_CNTL_1
    else:
      self.reg_sp_update_cntl      = mesa.REG_A8XX_SP_UPDATE_CNTL if self.gen == 8 else mesa.REG_A7XX_SP_UPDATE_CNTL
      self.reg_sp_reg_prog_id_0    = mesa.REG_A7XX_SP_REG_PROG_ID_0
      self.reg_sp_cs_const_config  = mesa.REG_A7XX_SP_CS_CONST_CONFIG
      self.reg_sp_cs_ndrange_0     = mesa.REG_A7XX_SP_CS_NDRANGE_0
      self.reg_sp_cs_uav_base      = mesa.REG_A7XX_SP_CS_UAV_BASE
      self.reg_sp_ps_wave_cntl     = mesa.REG_A7XX_SP_PS_WAVE_CNTL
      self.reg_sp_cs_wie_cntl_1    = mesa.REG_A7XX_SP_CS_WIE_CNTL_1

    if self.gen not in (6, 7, 8): raise RuntimeError(f"Unsupported GPU: chip_id={info.chip_id:#x} (gen={self.gen})")

    if PROFILE and self.gen == 6:
      System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", value="4000000000", msg="Failed to disable suspend mode", expected="4294967276")

    arch = self.arch_str + (f",QCOM_IMAGE_PITCH_ALIGNMENT={16 if self.gen >= 8 else 64}" if IMAGE else "") + f",chip_id={self.chip_id:#x}"
    renderers = [QCOMCLRenderer, IR3Renderer] if self.gen == 6 else [IR3Renderer]
    super().__init__(device, QCOMAllocator(self), renderers, functools.partial(QCOMProgram, self), QCOMSignal,
                     functools.partial(QCOMComputeQueue, self), arch=arch)
    # Standalone kernargs share the same CPU-write/GPU-read lifetime rule. Graph kernargs are separately allocated and persistent.
    self.kernargs_offset_allocator = BumpAllocator(self.kernargs_buf.size, wrap=True, wrap_callback=self.synchronize)

  def _gpu_alloc(self, size:int, flags:int=0, uncached=False, fill_zeroes=False) -> HCQBuffer:
    flags |= flag("KGSL_MEMALIGN", alignment_hint:=12) | kgsl.KGSL_MEMFLAGS_USE_CPU_MAP
    flags |= flag("KGSL_CACHEMODE", kgsl.KGSL_CACHEMODE_UNCACHED if uncached else kgsl.KGSL_CACHEMODE_WRITEBACK)

    alloc = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.fd, size=(bosz:=round_up(size, 1<<alignment_hint)), flags=flags, mmapsize=bosz)
    va_addr = self.fd.mmap(0, bosz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)

    if fill_zeroes: ctypes.memset(va_addr, 0, size)
    return HCQBuffer(va_addr=va_addr, size=size, meta=(alloc, True), view=MMIOInterface(va_addr, size, fmt='B'), owner=self)

  def _gpu_map(self, ptr:int, size:int) -> HCQBuffer:
    ptr_aligned, size_aligned = (ptr & ~0xfff), round_up(size + (ptr & 0xfff), 0x1000)
    dcache_flush().fxn(ctypes.c_uint64(ptr_line_aligned:=ptr & ~63), ceildiv(ptr + size - ptr_line_aligned, 64))
    try:
      mi = kgsl.IOCTL_KGSL_MAP_USER_MEM(self.fd, hostptr=ptr_aligned, len=size_aligned, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
      return HCQBuffer(mi.gpuaddr + (ptr - ptr_aligned), size=size, meta=(mi, False), view=MMIOInterface(ptr, size, fmt='B'), owner=self)
    except OSError as e:
      if e.errno == 14: return HCQBuffer(va_addr=ptr, size=size, meta=(None, False), view=MMIOInterface(ptr, size, fmt='B'), owner=self)
      raise RuntimeError("Failed to map external pointer to GPU memory") from e

  def _gpu_free(self, mem:HCQBuffer):
    if mem.meta[0] is None: return # external (gpu) ptr
    if not mem.meta[1]: kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.fd, gpuaddr=mem.meta[0].gpuaddr) # external (cpu) ptr
    else:
      kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=mem.meta[0].id)
      FileIOInterface.munmap(mem.va_addr, mem.meta[0].mmapsize)

  def _ensure_stack_size(self, sz):
    if not hasattr(self, '_stack'): self._stack = self._gpu_alloc(sz)
    elif self._stack.size < sz:
      self.synchronize()
      self._gpu_free(self._stack)
      self._stack = self._gpu_alloc(sz)

  def _at_profile_finalize(self):
    super()._at_profile_finalize()
    if self.gen == 6:
      with contextlib.suppress(OSError, RuntimeError):
        System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", "10", "Failed to reenable suspend mode")
