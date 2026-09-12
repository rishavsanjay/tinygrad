from __future__ import annotations
import os, ctypes, functools, mmap, struct, array, math, sys, contextlib, errno, time
from dataclasses import dataclass
assert sys.platform != 'win32'
from typing import Any
from tinygrad.device import Compiled, BufferStorage, BufferSpec, Buffer, Device, Allocator, TinyELF, ProfileProgramEvent
from tinygrad.runtime.support.hcq2 import HWQueue, HCQ_RUNTIME_DEV, encode_submit, ccall, cfield, cstruct, patch, unwrap_view
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.runtime.autogen import kgsl, mesa, libc
from tinygrad.renderer.cstyle import QCOMCLRenderer
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.helpers import getenv, mv_address, round_up, ceildiv, prod, is_image_shape, data64_le
from tinygrad.helpers import next_power2, flatten, PROFILE, IMAGE, VIZ, ContextVar
from tinygrad.dtype import dtypes, DType, AddrSpace
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops
from tinygrad.runtime.support.system import System
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

BUFTYPE_BUF, BUFTYPE_TEX, BUFTYPE_IBO = 0, 1, 2
QCOM_PMC = ContextVar("QCOM_PMC", int(abs(VIZ.value) >= 2))
QCOM_PMC_RING = ContextVar("QCOM_PMC_RING", 1024)
QCOM_RETIREMENT_RING = 4096

def _qcom_identity(chip_id:int, gpu_id:int=0):
  if not gpu_id:
    if chip_id == 0x07002000: gpu_id = 702  # A702 has a 7xx marketing ID but uses A6xx.
    elif (major:=(chip_id >> 24) & 0xff) < 0x10: gpu_id = major * 100 + ((chip_id >> 16) & 0xff) * 10 + ((chip_id >> 8) & 0xff)
  if gpu_id // 100 == 6 or gpu_id == 702: return mesa.struct_fd_dev_id(gpu_id, chip_id), 6, None

  dev_id = mesa.struct_fd_dev_id(gpu_id, chip_id)
  raw_dev_info = mesa.fd_dev_info_raw(dev_id)
  if not raw_dev_info or raw_dev_info.contents.chip == 0:
    raise RuntimeError(f"tinymesa required or device unsupported: chip_id={chip_id:#x} gpu_id={gpu_id}")
  return dev_id, raw_dev_info.contents.chip, mesa.fd_dev_info(dev_id)

@functools.cache
def dcache_flush():
  from tinygrad.uop.ops import KernelInfo
  from tinygrad.codegen import to_program
  buf, n = UOp.param(0, dtypes.uint8, 1), UOp.param(1, dtypes.int, shape=(), name="n", addrspace=AddrSpace.ALU)
  i = UOp.range(n, 0, dtype=dtypes.int)
  flush = UOp(Ops.CUSTOM, src=(buf.index(i * 64),), arg=('__asm__ volatile("dc cvac, %0" :: "r"({0}) : "memory");', dtypes.void))
  sink = UOp.sink(flush.end(i), UOp(Ops.CUSTOM, arg=('__asm__ volatile("dsb sy" ::: "memory");', dtypes.void)),
                  arg=KernelInfo(name="dcache_flush"), tag=1)
  prg = to_program(sink, Device["CPU"].renderer)
  return Device["CPU"].runtime(prg.to_elf())

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
  height:int
  width:int
  row_pitch:int
  array_pitch:int
  pitch_alignment:int
  @property
  def pitchalign(self): return ctz(self.row_pitch) - 6
  @property
  def size(self): return self.array_pitch

def qcom_image_layout(dtype:DType, shape:tuple[int, ...], row_pitch:int|None=None) -> QCOMImageLayout:
  if dtype not in (dtypes.half, dtypes.float): raise ValueError(f"unsupported QCOM image dtype {dtype}")
  if not is_image_shape(shape): raise ValueError(f"QCOM images require HxWx4 shape, got {shape}")
  height, width, _ = shape
  if not (0 < height <= 0x7fff and 0 < width <= 0x7fff): raise ValueError(f"QCOM image dimensions out of range: {shape}")
  pitch_alignment = 16 * 4 * dtype.itemsize
  if row_pitch is None: row_pitch = round_up(width * 4 * dtype.itemsize, pitch_alignment)
  # Height-one images may use any 64-byte-aligned row; there is no following row whose address needs the preferred pitch alignment.
  row_alignment = 64 if height == 1 else pitch_alignment
  if row_pitch < width * 4 * dtype.itemsize or row_pitch % row_alignment:
    raise ValueError(f"invalid QCOM image row pitch {row_pitch} for {shape} {dtype}; alignment is {pitch_alignment}")
  return QCOMImageLayout(height, width, row_pitch, round_up(row_pitch * round_up(height, 4), 0x1000), pitch_alignment)

def qcom_image_descriptor(gen:int, dtype:DType, shape:tuple[int, ...], addr, storage:bool=False, row_pitch:int|None=None) -> list:
  layout = qcom_image_layout(dtype, shape, row_pitch)
  if isinstance(addr, int) and addr & (0x3f if gen == 8 else 0x1f): raise ValueError(f"unaligned QCOM image address {addr:#x}")
  fmt = mesa.FMT6_32_32_32_32_FLOAT if dtype == dtypes.float else mesa.FMT6_16_16_16_16_FLOAT
  if gen in (6, 7):
    desc = [(fmt << 22) if storage else 0x8 | (1 << 7) | (2 << 10) | (3 << 13) | (fmt << 22),
            layout.width | (layout.height << 15), layout.pitchalign | (layout.row_pitch << 7) | (mesa.A6XX_TEX_2D << 29),
            0, *data64_le(addr), 0x40000000, 13]
    return desc + [0] * (16 - len(desc))
  if gen != 8: raise ValueError(f"unsupported QCOM image descriptor generation {gen}")
  if layout.row_pitch * 8 >= 1 << 24: raise ValueError(f"QCOM image row pitch is too large: {layout.row_pitch}")
  addr_lo, addr_hi = addr & 0xffffffc0, ((addr >> 32) & 0x1ffff) | (mesa.A6XX_TEX_2D << 17) | (1 << 20)
  if isinstance(addr, UOp): addr_lo, addr_hi = addr_lo.cast(dtypes.uint32), addr_hi.cast(dtypes.uint32)
  desc = [addr_lo, addr_hi,
          layout.width | (layout.height << 15), fmt | (3 << 10) | (4 << 13) | (5 << 16) | (6 << 19), 0, 0,
          layout.row_pitch * 8 | (layout.pitchalign << 24), (layout.array_pitch >> 12) & 0x7fffff]
  return desc + [0] * (16 - len(desc))

def qcom_sampler_descriptor(gen:int) -> list[int]:
  clamp = mesa.A6XX_TEX_CLAMP_TO_BORDER
  if gen in (6, 7):
    return [qreg.a6xx_tex_samp_0(wrap_s=clamp, wrap_t=clamp, wrap_r=clamp),
            qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0]
  if gen == 8: return [(clamp << 6) | (clamp << 9) | (clamp << 12), 1 << 31, 1, 0]
  raise ValueError(f"unsupported QCOM sampler generation {gen}")

def qcom_validate_image_counts(sampled:int, total:int):
  if not 0 <= sampled <= min(total, 31) or total > 32:
    raise RuntimeError(f"IR3 image resource limit exceeded: {sampled} sampled, {total} total")

def qcom_last_local_size(global_size, local_size):
  return tuple((g - 1) % l + 1 for g,l in zip(global_size, local_size))

def parity(val: int):
  for i in range(4,1,-1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int): return mesa.CP_TYPE7_PKT | cnt & 0x3FFF | parity(cnt) << 15 | (opcode & 0x7F) << 16 | parity(opcode) << 23

def pkt4_hdr(reg: int, cnt: int): return mesa.CP_TYPE4_PKT | cnt & 0x7F | parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | parity(reg) << 27

def _read_lib(lib, off) -> int: return struct.unpack("I", lib[off:off+4])[0]

class QCOMComputeQueue(HWQueue):
  dev:QCOMDevice
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._timestamp_values:list[UOp] = []
    self._profiled:list[UOp] = []

  def _prof_buf(self, name:str) -> UOp:
    buf = getattr(self.dev, name)
    return UOp.placeholder((buf.size,), buf.dtype, 0, device=self.devs, volatile=True, tag=name)

  def _pmc_begin(self, lib:UOp) -> UOp|None:
    if not self.dev.pmc_enabled: return None
    log = self._prof_buf("qcom_pmc_log")
    slot = (log.index(0).load() + len(self._profiled)) % self.dev.pmc_slots
    self._profiled.append(log.index(1 + slot.cast(dtypes.int)).store(UOp.const(unwrap_view(lib)[0].arg.slot, dtypes.uint64)))
    base = self.dev.qcom_pmc_buf.get_buf(self.dev.device) + slot * self.dev.pmc_record_size
    self.dev.perf.emit_snapshot(self, base)
    return base

  def _pmc_end(self, base:UOp|None):
    if base is not None: self.dev.perf.emit_snapshot(self, base + self.dev.perf.slot_vals * 64)

  def _pmc_bump(self, submit:UOp) -> UOp:
    if not self._profiled: return submit
    log = self._prof_buf("qcom_pmc_log")
    return log.after(submit, *self._profiled).index(0).store(log.index(0).load() + len(self._profiled))

  def cmd(self, opcode:int, *vals): self.q(pkt7_hdr(opcode, sum(x.dtype.itemsize // 4 if isinstance(x, UOp) else 1 for x in vals)), *vals)

  def reg(self, reg:int, *vals): self.q(pkt4_hdr(reg, sum(x.dtype.itemsize // 4 if isinstance(x, UOp) else 1 for x in vals)), *vals)

  def _cache_flush(self, write_back=True, invalidate=False, sync=True, memsync=False):
    if self.dev.gen == 6:
      if write_back:
        dummy = UOp.placeholder((0x1000,), dtypes.uint8, 0, device=self.devs, tag="dummy")
        self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS), dummy.getaddr(self.devs), 0)
      if invalidate: self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_INVALIDATE))
    else:
      if write_back: self.cmd(mesa.CP_EVENT_WRITE7, qreg.cp_event_write7_0(event=mesa.CACHE_FLUSH7))
      if invalidate: self.cmd(mesa.CP_EVENT_WRITE7, qreg.cp_event_write7_0(event=mesa.CACHE_INVALIDATE7))
    if memsync: self.cmd(mesa.CP_WAIT_MEM_WRITES)
    if sync: self.cmd(mesa.CP_WAIT_FOR_IDLE)

  def memory_barrier(self): self._cache_flush(write_back=True, invalidate=True, sync=True, memsync=True)

  def signal(self, signal:UOp, value:UOp):
    if self.dev.gen == 8: self._timestamp_values.append(value)
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    if self.dev.gen == 6:
      self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS), signal.getaddr(self.devs), value.cast(dtypes.uint32))
      self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    else:
      self.cmd(mesa.CP_EVENT_WRITE7, qreg.cp_event_write7_0(event=mesa.CACHE_FLUSH_TS, write_src=mesa.EV_WRITE_USER_32B,
                                                            write_dst=mesa.EV_DST_RAM, write_enabled=True),
               signal.getaddr(self.devs), value.cast(dtypes.uint32))

  def timestamp(self, signal:UOp):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    reg = mesa.REG_A8XX_CP_ALWAYS_ON_COUNTER if self.dev.gen == 8 else mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER
    self.reg_to_mem(reg, signal.getaddr(self.devs))

  def reg_to_mem(self, reg:int, addr, count:int=2):
    self.cmd(mesa.CP_REG_TO_MEM, qreg.cp_reg_to_mem_0(reg=reg, cnt=count, _64b=True), addr)

  def wait(self, signal:UOp, value:UOp):
    self.cmd(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY), signal.getaddr(self.devs),
             value.cast(dtypes.uint32), qreg.cp_wait_reg_mem_4(mask=0xFFFFFFFF), qreg.cp_wait_reg_mem_5(delay_loop_cycles=32))

  def kernargs(self, call:UOp, prg:UOp, data:QCOMProgramData) -> UOp:
    bufs, vals = get_call_arg_uops(call), get_call_var_uops(call, prg)
    ubos = [bufs[slot] for _,slot,_,shape in data.signature if slot < len(bufs) and not is_image_shape(shape)]
    uavs = [(dt,shape,bufs[slot]) for _,slot,dt,shape in data.signature if slot < len(bufs) and is_image_shape(shape)]
    # NIR can reorder images to different texture slots
    ibos = uavs[:data.ibo_cnt]
    texs = [uavs[data.tex_to_image[i] if data.NIR else data.ibo_cnt + i] for i in range(data.tex_cnt)]

    # the words of the kernargs, as runs at their byte offsets
    runs:list[tuple[int, list]] = [(off, [UOp.const(val, dtypes.uint32 if sz == 4 else dtypes.uint16)]) for val,off,sz in data.consts_info]
    runs.append((data.samp_off, data.samplers))
    if data.NIR:
      runs.append((data.buf_off, [b.getaddr(self.devs) for b in ubos]))
      runs += [(data.buf_off + o, [v.ccast(dt)]) for v,(o,dt) in zip(vals, TinyELF.iter_sig(data.signature[len(bufs):], len(ubos)*8))]
      if data.wgsz != 0xfc: runs.append((data.wgsz * 4, list(prg.arg.local_size)))
    else:
      runs += [(data.buf_offs[i], [b.getaddr(self.devs)]) for i, b in enumerate(ubos)]
      runs += [(data.buf_offs[i+len(ubos)], [v.ccast(dt)]) for i,(v,(_,_,dt,_)) in enumerate(zip(vals, data.signature[len(bufs):]))]

    def _tex(b, ibo=False):
      imgdt, shape, buf = b
      if self.dev.gen == 8: self.require_alignment(buf, 64, "QCOM image")
      pitch = shape[1] * 4 * imgdt.itemsize
      return qcom_image_descriptor(self.dev.gen, imgdt, shape, buf.getaddr(self.devs), storage=ibo, row_pitch=pitch)
    runs += [(data.tex_off, flatten(map(_tex, texs))), (data.ibo_off, flatten(map(functools.partial(_tex, ibo=True), ibos)))]

    # laid out as a linear in the cmdbuf tail, like amd's kernargs: the runs in order, zero bytes between them and after the last
    out, end = [], 0
    for off, run in sorted([r for r in runs if r[1]], key=lambda r: r[0]) + [(data.kernargs_alloc_size, [])]:
      assert off >= end, f"kernargs run at {off} overlaps the one ending at {end}"
      if off > end: out.append(UOp(Ops.BINARY, arg=bytes(off - end)))
      out += (run:=[w if isinstance(w, UOp) else UOp.const(w, dtypes.uint32) for w in run])
      end = off + sum(w.dtype.itemsize for w in run)
    return UOp(Ops.LINEAR, src=tuple(out))

  def exec(self, call:UOp, prg:UOp):
    data, lib = qcom_build_program(self.dev, prg, self.devs)
    pmc_base = self._pmc_begin(lib)
    global_size, local_size = prg.arg.global_size, prg.arg.local_size
    threads = prod(local_size)
    if (self.dev.gen == 6 and data.max_threads < threads) or (self.dev.gen == 8 and threads > 1024):
      raise RuntimeError("Too many resources requested for launch")
    if self.dev.gen == 6 and any(g*l>mx for g,l,mx in zip(global_size, local_size, [65536, 65536, 65536])) \
      and any(l>mx for l,mx in zip(local_size, [1024, 1024, 1024])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")

    def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x
    global_size_mp = [cast_int(g*l) for g,l in zip(global_size, local_size)]
    if self.dev.gen == 8 and any(isinstance(x, int) and not 0 < x <= 0xffffffff for x in (*global_size_mp, *local_size)):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    group_counts = tuple(cast_int(g, ceil=True) for g in global_size)

    args_addr, lib_addr = self.kernargs(call, prg, data).getaddr(self.devs), lib.getaddr(self.devs)
    stack_addr = UOp.placeholder((data.stack_alloc_size,), dtypes.uint8, 0, device=self.devs).rtag("stack").getaddr(self.devs)

    if self.dev.gen == 6:
      self.cmd(mesa.CP_SET_MARKER, qreg.a6xx_cp_set_marker_0(mode=mesa.RM6_COMPUTE))
      self.reg(mesa.REG_A6XX_SP_UPDATE_CNTL, qreg.a6xx_sp_update_cntl(cs_state=True, cs_uav=True))
      self.reg(mesa.REG_A6XX_SP_UPDATE_CNTL, 0)
      self.reg(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(0x80))
      self.reg(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(0x40))
      self.reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL if data.NIR else mesa.ISAMMODE_CL,
                                                                  constant_demotion_enable=data.NIR))
      self.reg(mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, qreg.a6xx_sp_perfctr_shader_mask(cs=True))
      self.reg(mesa.REG_A6XX_TPL1_MODE_CNTL, qreg.a6xx_tpl1_mode_cntl(isammode=mesa.ISAMMODE_GL if data.NIR else mesa.ISAMMODE_CL))
      self.reg(mesa.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)
    else:
      self.cmd(mesa.CP_SET_MARKER, qreg.a8xx_cp_set_marker_0(mode=mesa.RM6_COMPUTE))
      self.reg(mesa.REG_A8XX_SP_UPDATE_CNTL,
               qreg.a8xx_sp_update_cntl(vs_state=True, hs_state=True, ds_state=True, gs_state=True, fs_state=True, cs_state=True))
      self.reg(mesa.REG_A8XX_SP_UPDATE_CNTL, 0)
      self.reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL, constant_demotion_enable=True))
    self.cmd(mesa.CP_WAIT_FOR_IDLE)

    if self.dev.gen == 8 and data.instrlen > self.dev.dev_info.props.instr_cache_size:
      self.reg(mesa.REG_A6XX_SP_PS_INSTR_SIZE, qreg.a6xx_sp_ps_instr_size(data.instrlen))
      self.cmd(mesa.CP_EVENT_WRITE7, qreg.cp_event_write7_0(event=mesa.DEBUG_LABEL))

    if self.dev.gen == 6:
      self.reg(mesa.REG_A6XX_SP_CS_NDRANGE_0,
               qreg.a6xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
               global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, 0xccc0cf,
               0xfc | qreg.a6xx_sp_cs_wge_cntl(threadsize=mesa.THREAD64), *group_counts)
    else:
      last_local_size = qcom_last_local_size(global_size_mp, local_size)
      local_y = local_size[1]
      tile_height = 1 + 16 // math.gcd(local_y, 8) if isinstance(local_y, int) else \
        (local_y % 8).ne(0).where((local_y % 4).ne(0).where((local_y % 2).ne(0).where(17, 9), 5), 3)
      self.reg(mesa.REG_A7XX_SP_CS_NDRANGE_0,
               qreg.a7xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
               global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0)
      self.reg(mesa.REG_A7XX_SP_CS_WGE_CNTL,
               qreg.a7xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=data.threadsize,
                                        workgrouprastorderzfirsten=True, wgtilewidth=4, wgtileheight=tile_height))
      self.reg(mesa.REG_A7XX_SP_CS_KERNEL_GROUP_X, *group_counts)
      self.reg(mesa.REG_A7XX_SP_CS_NDRANGE_7,
               qreg.a7xx_sp_cs_ndrange_7(localsizex=last_local_size[0] - 1, localsizey=last_local_size[1] - 1,
                                         localsizez=last_local_size[2] - 1))

    if self.dev.gen == 6:
      cs_cntl_0 = qreg.a6xx_sp_cs_cntl_0(threadsize=mesa.THREAD64, halfregfootprint=data.hregs, fullregfootprint=data.fregs,
                                         branchstack=data.brnchstck)
      pvt_mem_size_reg = qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=data.pvtmem_size_total)
    else:
      cs_cntl_0 = qreg.a6xx_sp_cs_cntl_0(threadsize=data.threadsize, halfregfootprint=data.hregs, fullregfootprint=data.fregs,
                                         branchstack=data.brnchstck, earlypreamble=data.early_preamble, mergedregs=data.mergedregs)
      pvt_mem_size_reg = qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=data.pvtmem_size_total, perwavememlayout=data.pvtmem_per_wave)
    self.reg(mesa.REG_A6XX_SP_CS_CNTL_0, cs_cntl_0,
             qreg.a6xx_sp_cs_cntl_1(constantrammode=data.const_ram_mode if self.dev.gen == 8 else mesa.CONSTLEN_256,
                                    shared_size=data.shared_size),
             0, data.prg_offset, lib_addr,
             qreg.a6xx_sp_cs_pvt_mem_param(memsizeperitem=data.pvtmem_size_per_item), stack_addr, pvt_mem_size_reg)

    if self.dev.gen == 8: self.reg(mesa.REG_A7XX_SP_CS_VGS_CNTL, 0)
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER,
                                                             num_unit=1024 // (16 if self.dev.gen == 8 else 4)), args_addr)
    shader_units = min(data.instrlen, self.dev.dev_info.props.instr_cache_size) if self.dev.gen == 8 else ceildiv(data.image_size, 128)
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER, num_unit=shader_units), lib_addr)

    if self.dev.gen == 6:
      self.reg(mesa.REG_A6XX_SP_REG_PROG_ID_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc,
               qreg.a6xx_sp_cs_const_config(constlen=1024 // 4, enabled=True))
    else:
      self.reg(mesa.REG_A7XX_SP_REG_PROG_ID_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc00)
      self.reg(mesa.REG_A7XX_SP_CS_CONST_CONFIG, qreg.a7xx_sp_cs_const_config(constlen=data.constlen_units, enabled=True))

    self.reg(mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, qreg.a6xx_sp_cs_pvt_mem_stack_offset(data.hw_stack_offset))
    self.reg(mesa.REG_A6XX_SP_CS_INSTR_SIZE,
             qreg.a6xx_sp_cs_instr_size(ceildiv(data.image_size, 128) if self.dev.gen == 6 else data.instrlen))

    if data.samp_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_TEX, num_unit=data.samp_cnt), args_addr + data.samp_off)
      self.reg(mesa.REG_A6XX_SP_CS_SAMPLER_BASE, args_addr + data.samp_off)
      self.reg(mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE,
               UOp.placeholder((0x1000,), dtypes.uint8, 0, device=self.devs, tag="border_color").getaddr(self.devs))

    if data.tex_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_TEX, num_unit=min(16, data.tex_cnt)), args_addr + data.tex_off)
      self.reg(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, args_addr + data.tex_off)

    if data.ibo_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST6_UAV, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_SHADER, num_unit=data.ibo_cnt), args_addr + data.ibo_off)
      self.reg(mesa.REG_A7XX_SP_CS_UAV_BASE if self.dev.gen == 8 else mesa.REG_A6XX_SP_CS_UAV_BASE, args_addr + data.ibo_off)

    if self.dev.gen == 8 and (data.tex_cnt or data.ibo_cnt):
      self.reg(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(data.tex_cnt))
      self.reg(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(data.ibo_cnt))

    self.reg(mesa.REG_A6XX_SP_CS_CONFIG,
             qreg.a6xx_sp_cs_config(enabled=True, nsamp=data.samp_cnt, ntex=data.tex_cnt, nuav=data.ibo_cnt))

    if self.dev.gen == 8:
      self.reg(mesa.REG_A7XX_SP_PS_WAVE_CNTL, qreg.a7xx_sp_ps_wave_cntl(threadsize=mesa.THREAD64))
      self.reg(mesa.REG_A6XX_SP_CS_WIE_CNTL_0,
               qreg.a6xx_sp_cs_wie_cntl_0(wgidconstid=data.wgid, wgsizeconstid=data.wgsz, wgoffsetconstid=0xfc, localidregid=data.lid))
      self.reg(mesa.REG_A7XX_SP_CS_WIE_CNTL_1,
               qreg.a7xx_sp_cs_wie_cntl_1(linearlocalidregid=0xfc, threadsize=data.threadsize,
                                          workitemrastorder=mesa.WORKITEMRASTORDER_LINEAR))
      self.reg(mesa.REG_A8XX_SP_CS_HYSTERESIS, 0)
    elif data.NIR:
      self.reg(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
               qreg.a6xx_sp_cs_const_config_0(wgidconstid=data.wgid, wgsizeconstid=data.wgsz, wgoffsetconstid=0xfc, localidregid=data.lid),
               qreg.a6xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=mesa.THREAD64))

    if data.NIR:
      self.cmd(mesa.CP_EXEC_CS, 0, qreg.cp_exec_cs_1(ngroups_x=group_counts[0]),
               qreg.cp_exec_cs_2(ngroups_y=group_counts[1]), qreg.cp_exec_cs_3(_ngroups_z=group_counts[2]))
    else: self.cmd(mesa.CP_RUN_OPENCL, 0)

    self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    self._pmc_end(pmc_base)

  def submit(self, cmdbuf:UOp) -> UOp:
    ib, ib_off = unwrap_view(cmdbuf)
    fd, ctxid = [UOp.variable(n, 0, 2**31 - 1, dtypes.int32, param=True) for n in ("kgsl_fd", "kgsl_ctx")]
    obj = cstruct(kgsl.struct_kgsl_command_object, gpuaddr=ib.getaddr(self.devs) + ib_off, size=cmdbuf.max_numel(), flags=kgsl.KGSL_CMDLIST_IB)
    req = cstruct(kgsl.struct_kgsl_gpu_command, cmdlist=obj.getaddr(HCQ_RUNTIME_DEV.value), cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object),
                  numcmds=1, context_id=ctxid)
    ret = UOp.placeholder((1,), dtypes.int32, device=self.devs, volatile=True, tag="submit_ret")

    idir, base, nr, struct_t = kgsl.IOCTL_KGSL_GPU_COMMAND.args
    ioctl_cmd = (idir << 30) | (ctypes.sizeof(struct_t) << 16) | (base << 8) | nr
    last = ret.index(0).store(ccall(libc.dll.ioctl, fd, UOp.const(ioctl_cmd, dtypes.uint32), req.after(cmdbuf).index(0)))
    if self.dev.gen == 8:
      # Keep retirement metadata separate from HCQ signals. Writing a signal's adjacent word makes HCQ alias ordering move an earlier GPU wait
      # after the CPU timeline bump, which self-deadlocks the queue. Each record is [signal value, exact KGSL submission timestamp].
      timestamp = cfield(req.after(last), kgsl.struct_kgsl_gpu_command, "timestamp").cast(dtypes.uint64)
      log = UOp.placeholder((QCOM_RETIREMENT_RING * 2,), dtypes.uint64, device=self.devs, volatile=True, tag="qcom_retirement_log")
      for value in dict.fromkeys(self._timestamp_values):
        slot = value.cast(dtypes.int) % QCOM_RETIREMENT_RING
        last = log.index(slot * 2 + 1).store(timestamp)
        last = log.index(slot * 2).store(value.cast(dtypes.uint64)).after(last)
    return self._pmc_bump(last)

class QCOMProgramData:
  def __init__(self, dev:QCOMDevice, obj:TinyELF):
    self.signature, self.name, self.NIR = obj.signature, obj.name, isinstance(dev.renderer, IR3Renderer)

    if self.NIR:
      from tinygrad.runtime.support.compiler_mesa import IR3Compiler
      v, cs, imm_vals, self.image = IR3Compiler.unpack_lib(obj.lib)
      self.prg_offset = 0
      self.brnchstck = round_up(min(v.branchstack, 64), 2) if dev.gen == 8 else v.branchstack
      self.image_size, self.pvtmem, self.shmem = v.info.size, v.pvtmem_size, v.shared_size
      if dev.gen == 8:
        self.pvtmem_per_wave, self.early_preamble, self.mergedregs = v.pvtmem_per_wave, v.early_preamble, v.mergedregs
        self.instrlen, self.threadsize = v.instrlen, mesa.THREAD128 if v.info.double_threadsize else mesa.THREAD64
        self.constlen_units = round_up(v.constlen, 4) // 4
        self.const_ram_mode = mesa.CONSTLEN_512 if v.constlen > 256 else mesa.CONSTLEN_256 if v.constlen > 192 else \
          mesa.CONSTLEN_192 if v.constlen > 128 else mesa.CONSTLEN_128
      self.wgsz = alloc.offset_vec4 * 4 + 8 if (alloc:=cs.allocs.consts[mesa.IR3_CONST_ALLOC_DRIVER_PARAMS]).size_vec4 else 0xfc

      self.wgid, self.lid = v.cs.work_group_id, v.cs.local_invocation_id # register ids
      self.buf_off, imm_off = cs.ubo_state.range[0].offset, cs.allocs.max_const_offset_vec4 * 16
      self.consts_info = [(struct.unpack_from("<I", imm_vals, i)[0], imm_off + i, 4) for i in range(0, len(imm_vals), 4)]

      # see https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_shader.h#L525
      # and https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L5389
      self.samp_cnt, self.tex_cnt, self.ibo_cnt = (nt:=v.image_mapping.num_tex), nt, v.num_uavs
      self.tex_to_image = v.image_mapping.tex_to_image[:self.tex_cnt]
      qcom_validate_image_counts(self.tex_cnt, self.ibo_cnt)
      if any(i >= self.ibo_cnt for i in self.tex_to_image):
        raise RuntimeError(f"IR3 texture mapping outside image table: {self.tex_to_image} for {self.ibo_cnt} images")
      # IR3 outputs a sampler for every texture (https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L1714)
      self.samplers = qcom_sampler_descriptor(dev.gen) * self.samp_cnt

      self.tex_off, self.ibo_off, self.samp_off = 2048, 2048 + 0x40 * self.tex_cnt, 2048 + 0x40 * (self.tex_cnt + self.ibo_cnt)
      self.fregs, self.hregs = v.info.max_reg + 1, v.info.max_half_reg + 1
    else: self._parse_lib(obj.lib)

    if dev.gen == 6:
      self.pvtmem_size_per_item = round_up(self.pvtmem, 512) >> 9
      self.pvtmem_size_total = self.pvtmem_size_per_item * 128 * 2
      self.hw_stack_offset = round_up(next_power2(round_up(self.pvtmem, 512)) * 128 * 16, 0x1000)
      self.stack_alloc_size = self.hw_stack_offset * 4
      self.shared_size = max(1, (self.shmem - 1) // 1024)
      self.max_threads = min(1024, ((384 * 32) // (max(1, (self.fregs + round_up(self.hregs, 2) // 2)) * 128)) * 128)
    else:
      pvtmem_per_fiber = round_up(self.pvtmem, 512)
      pvtmem_per_sp = round_up(pvtmem_per_fiber * dev.dev_info.fibers_per_sp, 0x1000)
      self.pvtmem_size_per_item = pvtmem_per_fiber >> 9
      self.pvtmem_size_total = pvtmem_per_sp >> 12
      self.hw_stack_offset = pvtmem_per_sp >> 11
      self.stack_alloc_size = max(pvtmem_per_sp * dev.dev_info.num_sp_cores, 0x1000)
      if not 0 <= self.shmem <= 32 << 10: raise RuntimeError(f"Invalid shared-memory size {self.shmem}")
      self.shared_size = max(1, ceildiv(self.shmem, 1 << 10) - 1)
    self.kernargs_alloc_size = round_up(2048 + (self.tex_cnt + self.ibo_cnt) * 0x40 + len(self.samplers) * 4, 0x100)

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
      self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                       qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0, 0, 0, 0, 0]
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
        cnst = struct.unpack("I", lib[cdoff:cdoff+4])[0]
        offset_words, _, is32 = struct.unpack("III", lib[cdoff+16:cdoff+28])
        self.consts_info.append((cnst, offset_words * (sz_bytes:=(2 << is32)), sz_bytes))
        cdoff += 40

    # Registers info
    reg_desc_off = _read_lib(lib, 0x34)
    self.fregs, self.hregs = _read_lib(lib, reg_desc_off + 0x14), _read_lib(lib, reg_desc_off + 0x18)

_qcom_program_cache:dict[tuple[bytes, tuple[str, ...]], tuple[QCOMProgramData, UOp]] = {}
_qcom_program_prof:dict[UOp, tuple[str, bytes, bytes|None]] = {}
def qcom_build_program(dev:QCOMDevice, prg:UOp, devs:tuple[str, ...]) -> tuple[QCOMProgramData, UOp]:
  if (cached:=_qcom_program_cache.get(key:=(prg.src[3].arg, devs))) is None:
    data = QCOMProgramData(dev, obj:=prg.to_elf())
    image = bytes(data.image).ljust(round_up(len(data.image), 4), b"\x00")
    buf = UOp.placeholder((len(image),), dtypes.uint8, next(UOp.unique_num), device=devs).rtag("program")
    if PROFILE: _qcom_program_prof[buf] = (data.name, obj.lib, obj.profile_key)
    cached = _qcom_program_cache[key] = (data, patch(buf, [], image))
  return cached

class QCOMAllocator(Allocator['QCOMDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    return self.dev._gpu_map(options.external_ptr, size) if options.external_ptr else \
      self.dev._gpu_alloc(size, uncached=options.uncached or self.dev.gen == 8)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    self.dev._gpu_free(storage)
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

def flag(nm, val): return (val << getattr(kgsl, f"{nm}_SHIFT")) & getattr(kgsl, f"{nm}_MASK")

class QCOMDevice(Compiled):
  timestamp_divider = 19.2
  pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_qcom_compute", name="submit"), lambda ctx, submit: encode_submit(QCOMComputeQueue(ctx, submit))),
  ])

  @property
  def has_copy_queue(self) -> bool: return False

  def __init__(self, device:str=""):
    self.fd = FileIOInterface('/dev/kgsl-3d0', os.O_RDWR)

    flags = kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC \
      | flag("KGSL_CONTEXT_PRIORITY", getenv("QCOM_PRIORITY", 8)) | flag("KGSL_CONTEXT_PREEMPT_STYLE", kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN)
    self.ctx = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.fd, flags=flags).drawctxt_id
    self._stack:Buffer|None = None # private-memory stack

    # Set max power
    struct.pack_into('IIQQ', pwr:=memoryview(bytearray(0x18)), 0, 1, self.ctx, mv_address(_:=memoryview(array.array('I', [1]))), 4)
    kgsl.IOCTL_KGSL_SETPROPERTY(self.fd, type=kgsl.KGSL_PROP_PWR_CONSTRAINT, value=mv_address(pwr), sizebytes=pwr.nbytes)

    # Load info about qcom device
    info = kgsl.struct_kgsl_devinfo()
    kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.fd, type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    self.chip_id = info.chip_id
    dev_id, self.gen, self.dev_info = _qcom_identity(self.chip_id, info.gpu_id)
    gpu_id = dev_id.gpu_id or self.gen * 100
    self.gpu_id = (gpu_id // 100, (gpu_id // 10) % 10, gpu_id % 10)
    if self.gen not in (6, 8): raise RuntimeError(f"Unsupported GPU: chip_id={info.chip_id:#x}")

    if PROFILE and self.gen == 6:
      System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", value="4000000000", msg="Failed to disable suspend mode", expected="4294967276")

    arch = f"a{gpu_id}{',IMAGE_PITCH_ALIGNMENT=64' if self.gen == 6 and IMAGE else ''}"
    if self.gen == 8 and IMAGE: arch += ",IMAGE_PITCH_ALIGNMENT=16"
    if self.gen == 8: arch += f",chip_id={self.chip_id:#x}"
    renderers = [QCOMCLRenderer, IR3Renderer] if self.gen == 6 else [IR3Renderer]
    super().__init__(device, QCOMAllocator(self), renderers, None, arch=arch)

    self.pmc_enabled = bool(PROFILE > 0 and self.gen == 8 and QCOM_PMC.value > 0)
    if self.pmc_enabled:
      self.pmc_slots, self.pmc_read, self._pmc_last_ao = QCOM_PMC_RING.value, 0, None
      self.pm_bufferize = PatternMatcher([
        (UPat(Ops.PARAM, tag="program", name="b"), lambda ctx, b: ctx.program_buffer(b)),
        (UPat(Ops.PARAM, tag="qcom_pmc_log"), lambda ctx: ctx.qcom_pmc_log),
      ]) + self.pm_bufferize

    self.var_vals = {"kgsl_fd": self.fd.fd, "kgsl_ctx": self.ctx}
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="stack", name="b"), lambda ctx, b: ctx._ensure_stack_size(b.max_numel())),
      (UPat(Ops.PARAM, tag="dummy"), lambda ctx: ctx.dummy),
      (UPat(Ops.PARAM, tag="border_color"), lambda ctx: ctx.border_color),
      (UPat(Ops.PARAM, tag="qcom_retirement_log"), lambda ctx: ctx.qcom_retirement_log),
    ]) + self.pm_bufferize

  @functools.cached_property
  def dummy(self) -> Buffer: return Buffer(self.device, 0x1000, dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True) # cache flush target

  @functools.cached_property
  def border_color(self) -> Buffer: # zeros: the samplers clamp to a black border
    return Buffer(self.device, 0x1000, dtypes.uint8, options=BufferSpec(nolru=True), initial_value=bytes(0x1000))

  @functools.cached_property
  def perf(self):
    from tinygrad.runtime.support.qcom_profile import QCOMPerfCounters
    return QCOMPerfCounters(self)

  @property
  def pmc_record_size(self) -> int: return self.perf.slot_vals * 64 * 2

  @functools.cached_property
  def qcom_pmc_log(self) -> Buffer:
    return Buffer(self.device, 1 + self.pmc_slots, dtypes.uint64,
                  options=BufferSpec(host=True, uncached=True, cpu_access=True, nolru=True),
                  initial_value=bytes((1 + self.pmc_slots) * 8))

  @functools.cached_property
  def qcom_pmc_buf(self) -> Buffer:
    return Buffer(self.device, self.pmc_record_size * self.pmc_slots, dtypes.uint8,
                  options=BufferSpec(host=True, uncached=True, cpu_access=True, nolru=True),
                  initial_value=bytes(self.pmc_record_size * self.pmc_slots))

  @functools.cached_property
  def qcom_retirement_log(self) -> Buffer:
    return Buffer(self.device, QCOM_RETIREMENT_RING * 2, dtypes.uint64,
                  options=BufferSpec(host=True, uncached=True, cpu_access=True, nolru=True),
                  initial_value=bytes(QCOM_RETIREMENT_RING * 16))

  def program_buffer(self, b:UOp) -> Buffer:
    if b not in self.prog_bufs:
      buf = self.prog_bufs[b] = Buffer(self.device, b.max_numel(), b.dtype,
                                       options=BufferSpec(cpu_access=True, nolru=True), preallocate=True)
      if PROFILE:
        name, lib, key = _qcom_program_prof[b]
        Compiled.profile_events.append(ProfileProgramEvent(self.device, name, lib, buf._buf, b.arg.slot, key))
    return self.prog_bufs[b]

  def _gpu_alloc(self, size:int, flags:int=0, uncached=False, fill_zeroes=False) -> BufferStorage:
    flags |= flag("KGSL_MEMALIGN", alignment_hint:=12) | kgsl.KGSL_MEMFLAGS_USE_CPU_MAP
    if uncached: flags |= flag("KGSL_CACHEMODE", kgsl.KGSL_CACHEMODE_UNCACHED)

    alloc = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.fd, size=(bosz:=round_up(size, 1<<alignment_hint)), flags=flags, mmapsize=bosz)
    va_addr = self.fd.mmap(0, bosz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)

    if fill_zeroes: ctypes.memset(va_addr, 0, size)
    return BufferStorage(va_addr, (alloc, True), MMIOInterface(va_addr, size, fmt='B'))

  def _gpu_map(self, ptr:int, size:int) -> BufferStorage:
    ptr_aligned, size_aligned = (ptr & ~0xfff), round_up(size + (ptr & 0xfff), 0x1000)
    dcache_flush().fxn(ctypes.c_uint64(ptr_line_aligned:=ptr & ~63), ceildiv(ptr + size - ptr_line_aligned, 64))
    try:
      mi = kgsl.IOCTL_KGSL_MAP_USER_MEM(self.fd, hostptr=ptr_aligned, len=size_aligned, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
      return BufferStorage(mi.gpuaddr + (ptr - ptr_aligned), (mi, False), MMIOInterface(ptr, size, fmt='B'))
    except OSError as e:
      if e.errno == 14: return BufferStorage(ptr, (None, False), MMIOInterface(ptr, size, fmt='B'))
      raise RuntimeError("Failed to map external pointer to GPU memory") from e

  def _gpu_free(self, storage:BufferStorage):
    if storage.meta[0] is None: return # external (gpu) ptr
    if not storage.meta[1]: kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.fd, gpuaddr=storage.meta[0].gpuaddr) # external (cpu) ptr
    else:
      kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=storage.meta[0].id)
      FileIOInterface.munmap(storage.buf, storage.meta[0].mmapsize)

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    ts = 0
    if self.gen == 8 and (log_buf:=self.__dict__.get("qcom_retirement_log")) is not None:
      log, slot = log_buf.host.view(fmt='Q'), value % QCOM_RETIREMENT_RING
      if log[slot * 2] != value: raise RuntimeError(f"Missing KGSL retirement timestamp for QCOM signal value {value}")
      ts = log[slot * 2 + 1]
    if self.gen == 8 and ts:
      timeout_ms = self.wait_timeout_ms if timeout is None else timeout
      deadline = time.monotonic() + timeout_ms / 1000
      while True:
        remaining_ms = max(0, math.ceil((deadline - time.monotonic()) * 1000))
        try:
          kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.fd, context_id=self.ctx, timestamp=ts, timeout=remaining_ms)
          break
        except OSError as e:
          if e.errno in (errno.EINTR, errno.EAGAIN, errno.EDEADLK) and remaining_ms: continue
          if e.errno not in (errno.ETIMEDOUT, errno.EDEADLK): raise
          raise RuntimeError(f"Wait timeout: {timeout_ms} ms! KGSL submission {ts} did not retire") from e
    elif sig[0] < value:
      ts = kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID(self.fd, context_id=self.ctx, type=kgsl.KGSL_TIMESTAMP_QUEUED).timestamp
      with contextlib.suppress(OSError, RuntimeError):
        kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.fd, context_id=self.ctx, timestamp=ts, timeout=int(timeout or self.wait_timeout_ms))
    super()._wait_signal(sig, value, timeout)

  def _ensure_stack_size(self, sz:int) -> Buffer: # one stack for the device, grown to the deepest program's private memory
    if self._stack is None or self._stack.nbytes < sz:
      if self._stack is not None: self.synchronize()
      self._stack = Buffer(self.device, sz, dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True)
    return self._stack

  def _at_profile_finalize(self):
    if self.pmc_enabled:
      self.synchronize()
      self.pmc_enabled = False
      try: super()._at_profile_finalize()
      finally: self.pmc_enabled = True
      self.pmc_read = self.qcom_pmc_log.host.view(fmt='Q')[0]
    else: super()._at_profile_finalize()
    if self.gen == 6:
      with contextlib.suppress(RuntimeError): System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", "10", "Failed to reenable suspend mode")

  def collect_prof(self):
    if self.pmc_enabled:
      from tinygrad.runtime.support.qcom_profile import QCOMPMCSample, QCOMProfilePMCEvent
      log = self.qcom_pmc_log.host.view(fmt='Q')
      if (lost:=log[0] - self.pmc_read - self.pmc_slots) > 0:
        print(f"{self.device}: Warning: {lost} kernel profiles were overwritten; raise QCOM_PMC_RING")
      cpu = self.qcom_pmc_buf.host
      for k in range(max(self.pmc_read, log[0] - self.pmc_slots), log[0]):
        slot, tag = k % self.pmc_slots, log[1 + k % self.pmc_slots]
        deltas, self._pmc_last_ao = self.perf.fetch_record(cpu, slot * self.pmc_record_size, self._pmc_last_ao)
        sched = [QCOMPMCSample(name, off=8*i) for i,name in enumerate(self.perf.names)]
        blob = struct.pack(f"<{len(self.perf.names)}Q", *(deltas[n] for n in self.perf.names))
        Compiled.profile_events.append(QCOMProfilePMCEvent(self.device, tag, sched, blob, k))
      self.pmc_read = log[0]
    super().collect_prof()
