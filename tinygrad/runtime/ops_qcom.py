from __future__ import annotations
import os, ctypes, functools, mmap, struct, array, math, sys, weakref, contextlib, time, threading, errno
from dataclasses import dataclass
assert sys.platform != 'win32'
from typing import Any, cast
from tinygrad.device import BufferSpec, Device, TinyELF
from tinygrad.runtime.support.hcq import HCQBuffer, HWQueue, HCQProgram, HCQCompiled, HCQAllocatorBase, HCQSignal, HCQArgsState, BumpAllocator
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.runtime.autogen import kgsl, mesa
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import QCOMCLRenderer
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.helpers import getenv, mv_address, to_mv, round_up, data64_le, ceildiv, prod, cpu_profile, lo32, suppress_finalizing, is_image_shape
from tinygrad.helpers import next_power2, flatten, PROFILE, IMAGE, ContextVar, VIZ
from tinygrad.dtype import dtypes, AddrSpace, DType
from tinygrad.runtime.support.system import System
from tinygrad.uop.ops import sym_infer
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

BUFTYPE_BUF, BUFTYPE_TEX, BUFTYPE_IBO = 0, 1, 2
QCOM_PMC = ContextVar("QCOM_PMC", abs(VIZ.value)>=2)
# Ring capacity for per-kernel PMC records. Graph capture bakes one record slot per kernel
# position; the openpilot graph has ~310 kernels, so 1024 leaves ample headroom.
QCOM_PMC_RING = 1024

@functools.cache
def dcache_flush():
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.codegen import to_program
  buf, n = UOp.param(0, dtypes.uint8, shape=(1,)), UOp.param(1, dtypes.int, shape=(), name="n", addrspace=AddrSpace.ALU)
  i = UOp.range(n, 0, dtype=dtypes.int)
  flush = UOp(Ops.CUSTOM, src=(buf.index(i * 64),), arg=('__asm__ volatile("dc cvac, %0" :: "r"({0}) : "memory");', dtypes.void))
  sink = UOp.sink(flush.end(i), UOp(Ops.CUSTOM, arg=('__asm__ volatile("dsb sy" ::: "memory");', dtypes.void)), arg=KernelInfo(name="dcache_flush"))
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
  def pitchalign(self): return ctz(self.pitch_alignment) - 6
  @property
  def size(self): return self.array_pitch

def qcom_image_layout(dtype:DType, shape:tuple[int, ...], row_pitch:int|None=None) -> QCOMImageLayout:
  if dtype not in (dtypes.half, dtypes.float): raise ValueError(f"unsupported QCOM image dtype {dtype}")
  if not is_image_shape(shape): raise ValueError(f"QCOM images require HxWx4 shape, got {shape}")
  height, width, _ = shape
  if not (0 < height <= 0x7fff and 0 < width <= 0x7fff): raise ValueError(f"QCOM image dimensions out of range: {shape}")
  pitch_alignment = 16 * 4 * dtype.itemsize
  if row_pitch is None: row_pitch = round_up(width * 4 * dtype.itemsize, pitch_alignment)
  if row_pitch < width * 4 * dtype.itemsize or row_pitch % pitch_alignment:
    raise ValueError(f"invalid QCOM image row pitch {row_pitch} for {shape} {dtype}; alignment is {pitch_alignment}")
  # Mesa linear explicit layouts require 16-pixel rows and a four-row tail.
  # https://gitlab.freedesktop.org/mesa/mesa/-/blob/6dfbc555b4128ee51139c5f78c5aba2594c9701b/src/freedreno/fdl/fd6_layout.c#L180-221
  # https://gitlab.freedesktop.org/mesa/mesa/-/blob/6dfbc555b4128ee51139c5f78c5aba2594c9701b/src/freedreno/fdl/fd6_layout.c#L277-288
  # A8xx ARRAY_SLICE_OFFSET is encoded in 4 KiB units. Mesa rounds the complete level size before using it as the layer stride.
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
  # A8xx moves address, dimensions, format/swizzles, bit pitch, and array pitch into a distinct memobj layout.
  # https://gitlab.freedesktop.org/mesa/mesa/-/blob/6dfbc555b4128ee51139c5f78c5aba2594c9701b/src/freedreno/fdl/fd6_view.cc#L350-408
  desc = [addr & 0xffffffc0, ((addr >> 32) & 0x1ffff) | (mesa.A6XX_TEX_2D << 17) | (1 << 20),
          layout.width | (layout.height << 15), fmt | (3 << 10) | (4 << 13) | (5 << 16) | (6 << 19), 0, 0,
          layout.row_pitch * 8 | (layout.pitchalign << 24), (layout.array_pitch >> 12) & 0x7fffff]
  return desc + [0] * (16 - len(desc))

def qcom_sampler_descriptor(gen:int) -> list[int]:
  clamp = mesa.A6XX_TEX_CLAMP_TO_BORDER
  if gen in (6, 7):
    return [qreg.a6xx_tex_samp_0(wrap_s=clamp, wrap_t=clamp, wrap_r=clamp),
            qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0]
  if gen == 8:
    # Nearest is zero; A8xx relocates wrap/unnormalized fields and provides the zero-border fast path.
    # https://gitlab.freedesktop.org/mesa/mesa/-/blob/6dfbc555b4128ee51139c5f78c5aba2594c9701b/src/freedreno/vulkan/tu_sampler.cc#L90-112
    return [(clamp << 6) | (clamp << 9) | (clamp << 12), 1 << 31, 1, 0]
  raise ValueError(f"unsupported QCOM sampler generation {gen}")

def qcom_validate_image_counts(sampled:int, total:int):
  if not 0 <= sampled <= min(total, 31) or total > 32:
    raise RuntimeError(f"IR3 image resource limit exceeded: {sampled} sampled, {total} total")

def qcom_last_local_size(global_size, local_size) -> tuple[Any, Any, Any]:
  return ((global_size[0] - 1) % local_size[0] + 1, (global_size[1] - 1) % local_size[1] + 1,
          (global_size[2] - 1) % local_size[2] + 1)

def qcom_validate_launch(global_size, local_size):
  if len(global_size) != 3 or len(local_size) != 3 or any(type(l) is not int or not 0 < l <= 1024 for l in local_size):
    raise ValueError(f"Invalid QCOM launch {global_size=} {local_size=}")
  if prod(local_size) > 1024 or any(not isinstance(g, (int, float)) or not math.isfinite(g) or
      not 0 < g <= 0xffffffff or not 0 < g*l <= 0xffffffff or int(g*l) != g*l for g,l in zip(global_size, local_size)):
    raise ValueError(f"Invalid QCOM launch {global_size=} {local_size=}; limit 1024 threads and 32-bit extents")

def parity(val: int):
  for i in range(4,1,-1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int): return mesa.CP_TYPE7_PKT | cnt & 0x3FFF | parity(cnt) << 15 | (opcode & 0x7F) << 16 | parity(opcode) << 23

def pkt4_hdr(reg: int, cnt: int): return mesa.CP_TYPE4_PKT | cnt & 0x7F | parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | parity(reg) << 27

def _read_lib(lib, off) -> int: return struct.unpack("I", lib[off:off+4])[0]

class _QCOMSubmission:
  timestamp:int|None
  def __init__(self, dev:QCOMDevice): self.dev, self.timestamp = dev, None

class QCOMScratch:
  def __init__(self, dev:QCOMDevice, size:int):
    self.buf = dev._gpu_alloc(size)
    # Programs and captured queues retain this owner when a later shader grows the device scratch pool.
    weakref.finalize(self, dev.allocator.free, self.buf, size, BufferSpec(nolru=True))

class QCOMRingAllocator(BumpAllocator):
  def __init__(self, dev:QCOMDevice, size:int, base:int=0):
    super().__init__(size, base)
    self.dev = dev

  def alloc(self, size:int, alignment:int=1) -> int:
    if size > self.size: raise ValueError(f"QCOM ring request {size} exceeds capacity {self.size}")
    if round_up(self.ptr, alignment) + size > self.size and self.dev._last_submit_timestamp is not None:
      # The caller may already have reserved the next timeline value, which is not submitted yet.
      kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.dev.fd, context_id=self.dev.ctx,
        timestamp=self.dev._last_submit_timestamp, timeout=getenv("HCQDEV_WAIT_TIMEOUT_MS", 30000))
    return super().alloc(size, alignment)

class QCOMSignal(HCQSignal):
  _command_timestamps:dict[int, list[_QCOMSubmission]]
  _retirement_lock:threading.Lock
  _retired_value:int|None

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **{**kwargs, 'timestamp_divider': 19.2})
    if self.owner is not None and self.owner.gen == 8:
      self._command_timestamps = {}
      self._retirement_lock = threading.Lock()
      self._retired_value = None
      if isinstance(self.value_addr, int):
        self.owner._retirement_signals[self.value_addr] = self
        self.owner._submitted_retirement_signals.add(self)

  @property
  def value(self) -> int: return self.base_buf.cpu_view().view(0, 8, 'Q')[0]

  @value.setter
  def value(self, new_value:int):
    self.base_buf.cpu_view().view(0, 8, 'Q')[0] = new_value
    if hasattr(self, '_retirement_lock'):
      with self._retirement_lock: self._retired_value = None

  def _retirement_record(self, value:int, current_value:int) -> tuple[int, _QCOMSubmission]|None:
    with self._retirement_lock:
      # Once RAM satisfies an HCQ >= wait, only its current value can be its GPU producer.
      # Before then, select the first mapped value that can satisfy it.
      if current_value >= value:
        if (records:=self._command_timestamps.get(current_value)): return current_value, records[0]
      elif (satisfying_values:=[v for v,rs in self._command_timestamps.items() if rs and v >= value]):
        signal_value = min(satisfying_values)
        return signal_value, self._command_timestamps[signal_value][0]
    return None

  def _retired(self, record:_QCOMSubmission):
    assert record.timestamp is not None
    signals = list(record.dev._submitted_retirement_signals)
    if self not in signals: signals.append(self)
    for signal in signals:
      retired_values = []
      with signal._retirement_lock:
        for value,records in list(signal._command_timestamps.items()):
          kept = [r for r in records if r.dev is not record.dev or r.timestamp is None or
                  ((record.timestamp - r.timestamp) & 0xffffffff) >= (1 << 31)]
          if len(kept) != len(records): retired_values.append(value)
          if kept: signal._command_timestamps[value] = kept
          else: signal._command_timestamps.pop(value)
        if retired_values:
          signal._retired_value = max(retired_values) if signal._retired_value is None else max(signal._retired_value, *retired_values)

  def wait(self, value:int, timeout:int|None=None):
    if self.owner is None or self.owner.gen == 6: return super().wait(value, timeout)
    timeout = getenv("HCQDEV_WAIT_TIMEOUT_MS", 30000) if timeout is None else timeout
    if timeout < 0: raise ValueError("QCOM wait timeout must be nonnegative")
    stall_start = time.perf_counter()
    while True:
      current_value = self.value
      now = time.perf_counter()
      with self._retirement_lock: retired_value = self._retired_value
      if current_value >= value and retired_value is not None and retired_value >= value: return
      retirement = self._retirement_record(value, current_value)
      if current_value >= value and retirement is None: return
      remaining_ms = max(0, math.ceil(timeout - (now - stall_start) * 1000))
      if remaining_ms == 0 and timeout != 0: break
      if retirement is None or retirement[1].timestamp is None:
        if remaining_ms == 0: break
        self._sleep(math.floor((now - stall_start) * 1000))
        continue
      _, record = retirement
      try:
        kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(record.dev.fd, context_id=record.dev.ctx, timestamp=record.timestamp, timeout=remaining_ms)
      except OSError as e:
        # Retry transient failures within the original deadline; a deadlocked context is a fault.
        if e.errno == errno.EDEADLK: raise RuntimeError(f"QCOM context {record.dev.ctx} fault at timestamp {record.timestamp}") from e
        if e.errno in (errno.EINTR, errno.EAGAIN, errno.ETIMEDOUT):
          if timeout == 0: break
          continue
        raise
      self._retired(record)
      while True:
        current_value, now = self.value, time.perf_counter()
        if current_value >= value: return
        if math.ceil(timeout - (now - stall_start) * 1000) <= 0: break
        self._sleep(math.floor((now - stall_start) * 1000))
      break
    current_value = self.value
    raise RuntimeError(f"Wait timeout: {timeout} ms! (the signal is not set to {value}, but {current_value})")

  def _sleep(self, time_spent_since_last_sleep_ms:int):
    if self.owner is not None and self.owner.gen == 6 and self.is_timeline:
      kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.owner.fd, context_id=self.owner.ctx, timestamp=self.owner.last_cmd, timeout=0xffffffff)
    elif self.owner is not None and self.owner.gen == 8:
      time.sleep(0.001)

class QCOMComputeQueue(HWQueue):
  def __init__(self, dev:QCOMDevice):
    self.dev = dev
    self._launches:list[tuple[tuple, tuple]] = []
    self._arg_states:list[QCOMArgsState] = []
    if dev.gen == 8: self._signals:list[tuple[QCOMSignal, Any]] = []
    super().__init__()

  @suppress_finalizing
  def __del__(self):
    if self.binded_device is not None: self.binded_device.allocator.free(self.hw_page, self.hw_page.size, BufferSpec(cpu_access=True, nolru=True))

  def cmd(self, opcode: int, *vals: int): self.q(pkt7_hdr(opcode, len(vals)), *vals)

  def reg(self, reg: int, *vals: int): self.q(pkt4_hdr(reg, len(vals)), *vals)

  def _cache_flush(self, write_back=True, invalidate=False, sync=True, memsync=False):
    if self.dev.gen == 6:
      if write_back: self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS), *data64_le(self.dev.dummy_addr), 0)
      if invalidate: self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_INVALIDATE))
    else:
      if write_back: self.cmd(mesa.CP_EVENT_WRITE7, qreg.cp_event_write7_0(event=mesa.CACHE_FLUSH7))
      if invalidate: self.cmd(mesa.CP_EVENT_WRITE7, qreg.cp_event_write7_0(event=mesa.CACHE_INVALIDATE7))
    if memsync: self.cmd(mesa.CP_WAIT_MEM_WRITES)
    if sync: self.cmd(mesa.CP_WAIT_FOR_IDLE)

  def memory_barrier(self):
    self._cache_flush(write_back=True, invalidate=True, sync=True, memsync=True)
    return self

  def _dispatch_wait(self):
    # Keep the historical behavior by default. A8xx can opt out to test whether
    # CP_EXEC_CS snapshots enough state to safely queue consecutive dispatches.
    if self.dev.gen != 8 or getenv("QCOM_DISPATCH_WFI", 1): self.cmd(mesa.CP_WAIT_FOR_IDLE)

  def _pmc_begin(self, prg, global_size, local_size):
    dev = self.dev
    assert dev.perf is not None, "QCOM PMC requires reserved gen8 counters"
    self._pmc_off = dev._pmc_alloc_record()
    dev.perf.emit_snapshot(self, int(dev._pmc_buf.va_addr) + self._pmc_off)
    self._pmc_prg, self._pmc_gsize, self._pmc_lsize = prg, tuple(global_size), tuple(local_size)

  def _pmc_end(self):
    dev = self.dev
    assert dev.perf is not None
    dev.perf.emit_snapshot(self, int(dev._pmc_buf.va_addr) + self._pmc_off + dev.perf.slot_vals * 64)
    dev._pmc_pending.append((self._pmc_off, self._pmc_prg, self._pmc_gsize, self._pmc_lsize))

  def signal(self, signal:QCOMSignal, value=0):
    if self.dev.gen == 8: self._signals.append((signal, value))
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    if self.dev.gen == 6:
      self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS), *data64_le(signal.value_addr), lo32(value))
    else:
      self.cmd(mesa.CP_EVENT_WRITE7, qreg.cp_event_write7_0(event=mesa.CACHE_FLUSH_TS, write_src=mesa.EV_WRITE_USER_32B,
                                                            write_dst=mesa.EV_DST_RAM, write_enabled=True),
               *data64_le(signal.value_addr), lo32(value))
    if self.dev.gen == 6: self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
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
    if dev.gen == 8:
      for args in getattr(self, '_arg_states', ()):
        if all(isinstance(b.va_addr, int) for b in args.bufs): args.validate_resources({})
    if self.binded_device == dev: submit_req = self.submit_req
    else: submit_req, _ = self._build_gpu_command(dev)
    submissions:list[tuple[QCOMSignal, int, _QCOMSubmission]] = []
    try:
      if dev.gen == 8:
        resolved_signals = getattr(self, '_resolved_signals', None)
        if resolved_signals is None: resolved_signals = self._signals
        # The GPU may publish RAM completion before GPU_COMMAND returns its timestamp.
        for signal,value in resolved_signals:
          if isinstance(value, int):
            record = _QCOMSubmission(dev)
            dev._submitted_retirement_signals.add(signal)
            with signal._retirement_lock: signal._command_timestamps.setdefault(value, []).append(record)
            submissions.append((signal, value, record))
      timestamp = kgsl.IOCTL_KGSL_GPU_COMMAND(dev.fd, __payload=submit_req).timestamp
      dev._last_submit_timestamp = timestamp
      if dev.gen == 6: dev.last_cmd = timestamp
      else:
        for signal,value,record in submissions:
          with signal._retirement_lock:
            if record in signal._command_timestamps.get(value, []): record.timestamp = timestamp
    except BaseException:
      for signal,value,record in submissions:
        with signal._retirement_lock:
          if records:=signal._command_timestamps.get(value):
            records[:] = [r for r in records if r is not record]
            if not records: signal._command_timestamps.pop(value)
      raise
    finally:
      self.__dict__.pop('_resolved_signals', None)

  def _apply_var_vals(self, var_vals:dict[str, int]):
    if self.dev.gen == 6: return super()._apply_var_vals(var_vals)
    for gs,ls in getattr(self, '_launches', ()):
      qcom_validate_launch(tuple(sym_infer(g, var_vals) if not isinstance(g, float) else g for g in gs),
                           tuple(sym_infer(l, var_vals) for l in ls))
    for args in getattr(self, '_arg_states', ()): args.validate_resources(var_vals)
    resolved_signals = []
    for signal,value in self._signals:
      if isinstance(signal.value_addr, int): resolved_signal = signal
      else:
        if (physical_signal:=self.dev._retirement_signals.get(sym_infer(signal.value_addr, var_vals))) is None:
          raise RuntimeError("QCOM signal address did not resolve to an owned physical signal")
        resolved_signal = physical_signal
      resolved_signals.append((resolved_signal, sym_infer(value, var_vals)))
    self._resolved_signals = resolved_signals
    try: super()._apply_var_vals(var_vals)
    except BaseException:
      self.__dict__.pop('_resolved_signals', None)
      raise

  def exec(self, prg:QCOMProgram, args_state:QCOMArgsState, global_size, local_size):
    if self.dev.gen == 8:
      self._launches.append((tuple(global_size), tuple(local_size)))
      self._arg_states.append(args_state)
      if all(isinstance(v, (int, float)) for v in (*global_size, *local_size)): qcom_validate_launch(global_size, local_size)
    self.bind_args_state(args_state)

    def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x
    global_size_mp = [cast_int(g*l) for g,l in zip(global_size, local_size)]
    group_counts = tuple(cast_int(g, ceil=True) for g in global_size) if self.dev.gen == 8 else ()

    if self.dev.gen == 6:
      self.cmd(mesa.CP_SET_MARKER, qreg.a6xx_cp_set_marker_0(mode=mesa.RM6_COMPUTE))
      self.reg(mesa.REG_A6XX_SP_UPDATE_CNTL, qreg.a6xx_sp_update_cntl(cs_state=True, cs_uav=True))
      self.reg(mesa.REG_A6XX_SP_UPDATE_CNTL, 0)
      self.reg(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(0x80)) # is this right? mesa uses 1
      self.reg(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(0x40)) # mesa also uses 1
      self.reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL if prg.NIR else mesa.ISAMMODE_CL,
                                                                  constant_demotion_enable=prg.NIR))
      self.reg(mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, qreg.a6xx_sp_perfctr_shader_mask(cs=True))
      self.reg(mesa.REG_A6XX_TPL1_MODE_CNTL, qreg.a6xx_tpl1_mode_cntl(isammode=mesa.ISAMMODE_GL if prg.NIR else mesa.ISAMMODE_CL))
      self.reg(mesa.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)
    else:
      self.cmd(mesa.CP_SET_MARKER, qreg.a8xx_cp_set_marker_0(mode=mesa.RM6_COMPUTE))
      self.reg(mesa.REG_A8XX_SP_UPDATE_CNTL,
               qreg.a8xx_sp_update_cntl(vs_state=True, hs_state=True, ds_state=True, gs_state=True, fs_state=True, cs_state=True))
      self.reg(mesa.REG_A8XX_SP_UPDATE_CNTL, 0)
      self.reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL, constant_demotion_enable=True))
    self._dispatch_wait()

    # Oversized compute shaders need both instruction-length contexts to match.
    # https://gitlab.freedesktop.org/mesa/mesa/-/blob/6dfbc555b4128ee51139c5f78c5aba2594c9701b/src/freedreno/vulkan/tu_cmd_buffer.cc#L9148
    if self.dev.gen == 8 and prg.instrlen > self.dev.dev_info.props.instr_cache_size:
      self.reg(mesa.REG_A6XX_SP_PS_INSTR_SIZE, qreg.a6xx_sp_ps_instr_size(prg.instrlen))
      self.cmd(mesa.CP_EVENT_WRITE7, qreg.cp_event_write7_0(event=mesa.DEBUG_LABEL))

    if self.dev.gen == 6:
      self.reg(mesa.REG_A6XX_SP_CS_NDRANGE_0,
               qreg.a6xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
               global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, 0xccc0cf,
               0xfc | qreg.a6xx_sp_cs_wge_cntl(threadsize=mesa.THREAD64),
               cast_int(global_size[0], ceil=True), cast_int(global_size[1], ceil=True), cast_int(global_size[2], ceil=True))
    else:
      last_local_size = qcom_last_local_size(global_size_mp, local_size)
      # A8xx reuses the A7xx compute layouts; tile height follows local Y divisibility.
      # https://gitlab.freedesktop.org/mesa/mesa/-/blob/6dfbc555b4128ee51139c5f78c5aba2594c9701b/src/freedreno/computerator/a6xx.cc#L218
      if isinstance(local_y:=local_size[1], int): tile_height = 1 + 16 // math.gcd(local_y, 8)
      else: tile_height = (local_y % 8).ne(0).where((local_y % 4).ne(0).where((local_y % 2).ne(0).where(17, 9), 5), 3)
      self.reg(mesa.REG_A7XX_SP_CS_NDRANGE_0,
               qreg.a7xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
               global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0)
      self.reg(mesa.REG_A7XX_SP_CS_WGE_CNTL,
               qreg.a7xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=prg.threadsize,
                                        workgrouprastorderzfirsten=True, wgtilewidth=4, wgtileheight=tile_height))
      self.reg(mesa.REG_A7XX_SP_CS_KERNEL_GROUP_X, *group_counts)
      self.reg(mesa.REG_A7XX_SP_CS_NDRANGE_7,
               qreg.a7xx_sp_cs_ndrange_7(localsizex=last_local_size[0] - 1, localsizey=last_local_size[1] - 1,
                                         localsizez=last_local_size[2] - 1))

    if self.dev.gen == 6:
      cs_cntl_0 = qreg.a6xx_sp_cs_cntl_0(threadsize=mesa.THREAD64, halfregfootprint=prg.hregs, fullregfootprint=prg.fregs,
                                         branchstack=prg.brnchstck)
      pvt_mem_size_reg = qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=prg.pvtmem_size_total)
    else:
      # Bit 22 (UNK22 here, COMPUTERRMODEEN in Mesa 26.2) requests round-robin wave scheduling.
      # Mesa normally enables it when the shader asks for occupancy-bounded workgroup fairness
      # (or via its debug override). A8xx has round_robin_errata=False, so the <=8-wave register
      # footprint workaround used on affected later A6xx/A7xx parts does not apply to A830.
      # This env knob is deliberately an unconditional bit-level experiment: "no effect" only
      # characterizes the workloads tested with it, not forward-progress-sensitive algorithms.
      rr = prg.round_robin_mode or getenv("QCOM_COMPUTE_RR", 0)
      cs_cntl_0 = qreg.a6xx_sp_cs_cntl_0(threadsize=prg.threadsize, halfregfootprint=prg.hregs, fullregfootprint=prg.fregs,
                                         branchstack=prg.brnchstck, earlypreamble=prg.early_preamble, mergedregs=prg.mergedregs,
                                         **({'unk22': True} if rr else {}))
      pvt_mem_size_reg = qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=prg.pvtmem_size_total, perwavememlayout=prg.pvtmem_per_wave)
    self.reg(mesa.REG_A6XX_SP_CS_CNTL_0, cs_cntl_0,
             qreg.a6xx_sp_cs_cntl_1(constantrammode=prg.const_ram_mode if self.dev.gen == 8 else mesa.CONSTLEN_256,
                                    shared_size=prg.shared_size),
             0, prg.prg_offset, *data64_le(prg.lib_gpu.va_addr),
             qreg.a6xx_sp_cs_pvt_mem_param(memsizeperitem=prg.pvtmem_size_per_item),
             *data64_le(prg.scratch.buf.va_addr if self.dev.gen == 8 else prg.dev._stack.va_addr), pvt_mem_size_reg)

    if self.dev.gen == 8: self.reg(mesa.REG_A7XX_SP_CS_VGS_CNTL, 0)

    if prg.NIR and prg.wgsz != 0xfc: to_mv(int(args_state.buf.va_addr) + prg.wgsz * 4, 12)[:] = struct.pack("III", *local_size)
    # A8xx constant-state loads count vec4s; A6xx loads count dwords.
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER,
                                                             num_unit=1024 // (16 if self.dev.gen == 8 else 4)),
             *data64_le(args_state.buf.va_addr))
    # Preload only the portion of the shader that fits in the instruction cache.
    # https://gitlab.freedesktop.org/mesa/mesa/-/blob/6dfbc555b4128ee51139c5f78c5aba2594c9701b/src/freedreno/computerator/a6xx.cc#L279
    shader_units = min(prg.instrlen, self.dev.dev_info.props.instr_cache_size) if self.dev.gen == 8 else round_up(prg.image_size, 128) // 128
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER, num_unit=shader_units),
             *data64_le(prg.lib_gpu.va_addr))

    if self.dev.gen == 6:
      self.reg(mesa.REG_A6XX_SP_REG_PROG_ID_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc,
               qreg.a6xx_sp_cs_const_config(constlen=1024 // 4, enabled=True))
    else:
      self.reg(mesa.REG_A7XX_SP_REG_PROG_ID_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc00)
      self.reg(mesa.REG_A7XX_SP_CS_CONST_CONFIG, qreg.a7xx_sp_cs_const_config(constlen=prg.constlen_units, enabled=True))

    self.reg(mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, qreg.a6xx_sp_cs_pvt_mem_stack_offset(prg.hw_stack_offset))
    # image_size is in bytes, but INSTR_SIZE is measured in units of instruction groups (16 instructions, 8 bytes each)
    # https://elixir.bootlin.com/mesa/mesa-26.1.5/source/src/freedreno/ir3/ir3_shader.h#L719-L723
    self.reg(mesa.REG_A6XX_SP_CS_INSTR_SIZE, qreg.a6xx_sp_cs_instr_size(ceildiv(prg.image_size, 128) if self.dev.gen == 6 else prg.instrlen))

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
      self.reg(mesa.REG_A7XX_SP_CS_UAV_BASE if self.dev.gen == 8 else mesa.REG_A6XX_SP_CS_UAV_BASE,
               *data64_le(args_state.buf.va_addr + args_state.prg.ibo_off))

    if self.dev.gen == 8 and (prg.tex_cnt or prg.ibo_cnt):
      self.reg(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(prg.tex_cnt))
      self.reg(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(prg.ibo_cnt))

    self.reg(mesa.REG_A6XX_SP_CS_CONFIG,
             qreg.a6xx_sp_cs_config(enabled=True, nsamp=args_state.prg.samp_cnt, ntex=args_state.prg.tex_cnt, nuav=args_state.prg.ibo_cnt))

    # https://gitlab.freedesktop.org/mesa/mesa/-/blob/6dfbc555b4128ee51139c5f78c5aba2594c9701b/src/freedreno/vulkan/tu_shader.cc#L1962-1975
    if self.dev.gen == 8:
      self.reg(mesa.REG_A7XX_SP_PS_WAVE_CNTL, qreg.a7xx_sp_ps_wave_cntl(threadsize=mesa.THREAD64))
      self.reg(mesa.REG_A6XX_SP_CS_WIE_CNTL_0,
               qreg.a6xx_sp_cs_wie_cntl_0(wgidconstid=prg.wgid, wgsizeconstid=prg.wgsz, wgoffsetconstid=0xfc, localidregid=prg.lid))
      self.reg(mesa.REG_A7XX_SP_CS_WIE_CNTL_1,
               qreg.a7xx_sp_cs_wie_cntl_1(linearlocalidregid=0xfc, threadsize=prg.threadsize,
                                          workitemrastorder=mesa.WORKITEMRASTORDER_LINEAR))
      self.reg(mesa.REG_A8XX_SP_CS_HYSTERESIS, 0)
    elif prg.NIR:
      self.reg(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
               qreg.a6xx_sp_cs_const_config_0(wgidconstid=prg.wgid, wgsizeconstid=prg.wgsz, wgoffsetconstid=0xfc, localidregid=prg.lid),
               qreg.a6xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=mesa.THREAD64))

    # Per-kernel PMC snapshots bracket CP_EXEC_CS in the same submission so attribution is
    # exact without extra KGSL round-trips. The snapshots add WFI stalls around the kernel;
    # measured durations therefore include profiling overhead (like every PMC mode).
    if self.dev.pmc_enabled: self._pmc_begin(prg, global_size, local_size)
    if prg.NIR and self.dev.gen == 8:
      self.cmd(mesa.CP_EXEC_CS, 0, qreg.cp_exec_cs_1(ngroups_x=group_counts[0]),
               qreg.cp_exec_cs_2(ngroups_y=group_counts[1]), qreg.cp_exec_cs_3(_ngroups_z=group_counts[2]))
    elif prg.NIR:
      self.cmd(mesa.CP_EXEC_CS, 0,
               qreg.cp_exec_cs_1(ngroups_x=global_size[0]), qreg.cp_exec_cs_2(ngroups_y=global_size[1]), qreg.cp_exec_cs_3(_ngroups_z=global_size[2]))
    else: self.cmd(mesa.CP_RUN_OPENCL, 0)
    if self.dev.pmc_enabled: self._pmc_end()

    # This writeback is load-bearing on A830: removing it corrupts dependent
    # dispatch chains even when CP_WAIT_FOR_IDLE remains enabled.
    self._cache_flush(write_back=True, invalidate=bool(self.dev.gen == 8 and getenv("QCOM_IMAGE_INVALIDATE", 0) and
                      (prg.tex_cnt or prg.ibo_cnt)), sync=False, memsync=False)
    return self

class QCOMArgsState(HCQArgsState):
  def validate_resources(self, var_vals):
    if not self.prg.NIR: return
    images = [(slot, shape, dt) for _,slot,dt,shape in self.prg.signature if slot < len(self.bufs) and is_image_shape(shape)]
    for slot,shape,dt in images:
      addr = sym_infer(self.bufs[slot].va_addr, var_vals)
      qcom_image_descriptor(self.prg.dev.gen, dt, shape, addr, row_pitch=shape[1]*4*dt.itemsize)
    # Distinct NIR parameters may alias at runtime. A sampled view of a writable allocation is not coherent.
    for i in self.prg.tex_to_image:
      sampled = self.bufs[images[i][0]]
      start = sym_infer(sampled.va_addr, var_vals)
      for slot in self.prg.write_slots:
        written = self.bufs[slot]
        addr = sym_infer(written.va_addr, var_vals)
        if start < addr + written.size and addr < start + sampled.size:
          raise ValueError("QCOM sampled image aliases a writable argument; use buffer lowering")

  def __init__(self, buf:HCQBuffer, prg:QCOMProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    super().__init__(buf, prg, bufs, vals=vals)
    ctypes.memset(int(self.buf.va_addr), 0, prg.kernargs_alloc_size)

    ubos = [bufs[slot] for _,slot,_,shape in prg.signature if slot < len(bufs) and not is_image_shape(shape)]
    images = [(dt,shape,bufs[slot]) for _,slot,dt,shape in prg.signature if slot < len(bufs) and is_image_shape(shape)]
    if prg.NIR:
      if len(images) != prg.ibo_cnt: raise RuntimeError(f"IR3 image count mismatch: signature has {len(images)}, compiler requires {prg.ibo_cnt}")
      # NIR image indices are image-parameter ordinals. Every image gets a UAV entry; sampled descriptors are compacted in IR3 order.
      ibos, texs = images, [images[prg.tex_to_image[i]] for i in range(prg.tex_cnt)]
    else:
      # The legacy OpenCL object records storage arguments first and sampled arguments second.
      ibos, texs = images[:prg.ibo_cnt], images[prg.ibo_cnt:prg.ibo_cnt + prg.tex_cnt]
    for cnst_val,cnst_off,cnst_sz in prg.consts_info:
      to_mv(cast(int, self.buf.va_addr) + cnst_off, cnst_sz)[:] = cnst_val.to_bytes(cnst_sz, byteorder='little')

    if prg.samp_cnt > 0: to_mv(int(self.buf.va_addr) + prg.samp_off, len(prg.samplers) * 4).cast('I')[:] = array.array('I', prg.samplers)
    if prg.NIR:
      signature = {slot:dt for _,slot,dt,_ in prg.signature}
      for slot, offset, size in prg.param_layout:
        if slot < len(bufs): value, fmt, expected_size = bufs[slot].va_addr, 'Q', 8
        else:
          if slot - len(bufs) >= len(vals): raise ValueError(f"Missing IR3 scalar argument {slot}")
          value, fmt, expected_size = vals[slot - len(bufs)], cast(str, signature[slot].fmt), signature[slot].itemsize
        if size != expected_size: raise ValueError(f"IR3 argument {slot} width mismatch")
        self.bind_sints_to_buf(value, buf=self.buf, fmt=fmt, offset=prg.buf_off + offset)
    else:
      for i, b in enumerate(ubos): self.bind_sints_to_buf(b.va_addr, buf=self.buf, fmt='Q', offset=prg.buf_offs[i])
      for i,(v,(_,_,dt,_)) in enumerate(zip(vals, prg.signature[len(bufs):])):
        self.bind_sints_to_buf(v, buf=self.buf, fmt=dt.fmt, offset=prg.buf_offs[i+len(ubos)])

    def _tex(b, ibo=False):
      imgdt, shape, buf = b
      pitch = shape[1] * 4 * imgdt.itemsize
      layout = qcom_image_layout(imgdt, shape, pitch)
      # Base allocations are page-rounded by KGSL. Views may use that tail only when it really remains after their offset.
      if isinstance(buf.va_addr, int):
        owned = isinstance(buf.base.meta, tuple) and len(buf.base.meta) > 1 and buf.base.meta[1] is True
        allocation_size = round_up(buf.base.size, 0x1000) if owned else buf.base.size
        available = allocation_size - (int(buf.va_addr) - int(buf.base.va_addr))
      else:
        # HCQGraph virtual buffers preserve the logical size of page-rounded QCOM allocations until replay binds their concrete addresses.
        available = round_up(buf.size, 0x1000)
      if layout.size > available:
        raise ValueError(f"QCOM image layout needs {layout.size} bytes, only {available} remain in the allocation")
      if prg.dev.gen == 8: return qcom_image_descriptor(8, imgdt, shape, buf.va_addr, storage=ibo, row_pitch=pitch)
      fmt = mesa.FMT6_32_32_32_32_FLOAT if imgdt.itemsize == 4 else mesa.FMT6_16_16_16_16_FLOAT
      return [qreg.a6xx_tex_const_0(fmt=fmt) if ibo else qreg.a6xx_tex_const_0(0x8, swiz_x=0, swiz_y=1, swiz_z=2, swiz_w=3, fmt=fmt),
              qreg.a6xx_tex_const_1(width=shape[1], height=shape[0]),
              qreg.a6xx_tex_const_2(type=mesa.A6XX_TEX_2D, pitch=pitch, pitchalign=ctz(pitch)-6), 0, *data64_le(buf.va_addr),
              qreg.a6xx_tex_const_6(plane_pitch=0x400000), qreg.a6xx_tex_const_7(13), 0, 0, 0, 0, 0, 0, 0, 0]

    self.bind_sints_to_buf(*flatten(map(_tex, texs)), buf=self.buf, fmt='I', offset=prg.tex_off)
    self.bind_sints_to_buf(*flatten(map(functools.partial(_tex, ibo=True), ibos)), buf=self.buf, fmt='I', offset=prg.ibo_off)

class QCOMProgram(HCQProgram['QCOMDevice']):
  scratch:QCOMScratch
  def __init__(self, dev: QCOMDevice, obj: TinyELF):
    self.dev: QCOMDevice = dev
    self.signature, self.name, self.NIR = obj.signature, obj.name, isinstance(dev.renderer, IR3Renderer)

    if self.NIR:
      from tinygrad.runtime.support.compiler_mesa import IR3Compiler
      v, imm_vals, self.image = IR3Compiler.unpack_lib(obj.lib)
      if v.arch != dev.renderer.target.arch: raise ValueError(f"IR3 shader target {v.arch} does not match {dev.renderer.target.arch}")
      if v.build != cast(IR3Compiler, dev.renderer.compiler).build: raise ValueError("IR3 shader Mesa build mismatch; recompile shader")
      self.param_layout, self.round_robin_mode, self.write_slots = v.params, v.round_robin_mode, v.writes
      slots = {slot for _,slot,_,_ in self.signature}
      if any(slot not in slots for slot,_,_ in v.params) or any(slot not in slots for slot in v.writes):
        raise ValueError("IR3 metadata refers to an argument outside the program signature")
      self.prg_offset, self.brnchstck, self.image_size = 0, round_up(v.branchstack, 2) if dev.gen == 8 else v.branchstack, len(self.image)
      self.pvtmem, self.shmem = v.pvtmem_size, v.shared_size
      if dev.gen == 8:
        self.pvtmem_per_wave, self.early_preamble, self.mergedregs = v.pvtmem_per_wave, v.early_preamble, v.mergedregs
        self.instrlen, self.threadsize = v.instrlen, mesa.THREAD128 if v.double_threadsize else mesa.THREAD64
        # Select the smallest constant RAM mode that contains the shader constants.
        # https://gitlab.freedesktop.org/mesa/mesa/-/blob/6dfbc555b4128ee51139c5f78c5aba2594c9701b/src/freedreno/computerator/a6xx.cc#L191
        self.constlen_units = round_up(v.constlen, 4) // 4
        self.const_ram_mode = mesa.CONSTLEN_512 if v.constlen > 256 else mesa.CONSTLEN_256 if v.constlen > 192 else \
          mesa.CONSTLEN_192 if v.constlen > 128 else mesa.CONSTLEN_128
      self.wgsz = v.wgsz

      self.wgid, self.lid = v.wgid, v.lid
      self.buf_off, imm_off = v.buf_off, v.imm_off
      self.consts_info = [(struct.unpack_from("<I", imm_vals, i)[0], imm_off + i, 4) for i in range(0, len(imm_vals), 4)]

      # see https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_shader.h#L525
      # and https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L5389
      self.samp_cnt, self.tex_cnt, self.ibo_cnt = (nt:=len(v.tex_to_image)), nt, v.num_uavs
      qcom_validate_image_counts(self.samp_cnt, self.ibo_cnt)
      self.tex_to_image = v.tex_to_image
      if any(i >= self.ibo_cnt for i in self.tex_to_image):
        raise RuntimeError(f"IR3 texture mapping outside image table: {self.tex_to_image} for {self.ibo_cnt} images")
      # IR3 outputs a sampler for every texture (https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L1714)
      self.samplers = (qcom_sampler_descriptor(8) if dev.gen == 8 else
        [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
         qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0]) * self.samp_cnt

      self.tex_off, self.ibo_off, self.samp_off = 2048, 2048 + 0x40 * self.tex_cnt, 2048 + 0x40 * (self.tex_cnt + self.ibo_cnt)
      self.fregs, self.hregs = v.fregs, v.hregs
    else: self._parse_lib(obj.lib)

    self.lib_gpu: HCQBuffer = self.dev.allocator.alloc(self.image_size, buf_spec:=BufferSpec(cpu_access=True, nolru=True))
    to_mv(self.lib_gpu.va_addr, self.image_size)[:] = self.image

    if dev.gen == 8 and not 0 <= self.shmem <= dev.dev_info.cs_shared_mem_size:
      raise RuntimeError(f"Invalid shared-memory size {self.shmem}, device limit {dev.dev_info.cs_shared_mem_size}")
    self.shared_size = max(1, (self.shmem - 1) // 1024)
    if dev.gen == 6:
      self.pvtmem_size_per_item = round_up(self.pvtmem, 512) >> 9
      self.pvtmem_size_total = self.pvtmem_size_per_item * 128 * 2
      self.hw_stack_offset = round_up(next_power2(round_up(self.pvtmem, 512)) * 128 * 16, 0x1000)
      self.max_threads = min(1024, ((384 * 32) // (max(1, (self.fregs + round_up(self.hregs, 2) // 2)) * 128)) * 128)
      dev._ensure_stack_size(self.hw_stack_offset * 4)
    else:
      # Private memory is allocated per SP across all resident fibers.
      # https://gitlab.freedesktop.org/mesa/mesa/-/blob/6dfbc555b4128ee51139c5f78c5aba2594c9701b/src/freedreno/computerator/a6xx.cc#L261
      pvtmem_per_fiber = round_up(self.pvtmem, 512)
      pvtmem_per_sp = round_up(pvtmem_per_fiber * dev.dev_info.fibers_per_sp, 0x1000)
      self.pvtmem_size_per_item = pvtmem_per_fiber >> 9
      self.pvtmem_size_total = pvtmem_per_sp >> 12
      self.hw_stack_offset = pvtmem_per_sp >> 11
      dev._ensure_stack_size(max(pvtmem_per_sp * dev.dev_info.num_sp_cores, 0x1000))
      self.scratch = dev._scratch

    kernargs_alloc_size = round_up(2048 + (self.tex_cnt + self.ibo_cnt) * 0x40 + len(self.samplers) * 4, 0x100)
    super().__init__(QCOMArgsState, self.dev, obj, kernargs_alloc_size=kernargs_alloc_size)
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int|None, ...]=(), wait=False, timeout:int|None=None):
    if self.dev.gen == 8: qcom_validate_launch(global_size, local_size)
    threads = prod(local_size)
    if (self.dev.gen == 6 and self.max_threads < threads) or (self.dev.gen == 8 and threads > 1024):
      raise RuntimeError("Too many resources requested for launch")
    if self.dev.gen == 6 and any(g*l>mx for g,l,mx in zip(global_size, local_size, [65536, 65536, 65536])) \
      and any(l>mx for l,mx in zip(local_size, [1024, 1024, 1024])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    if self.dev.gen == 8 and (any(not 0 < l <= 1024 for l in local_size) or
                              any(not 0 < g or math.ceil(g) > 0xffffffff or g*l > 0xffffffff for g,l in zip(global_size, local_size))):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    n_pmc = len(getattr(self.dev, "_pmc_pending", ()))
    if getenv("DEBUG_DISPATCH"):
      print(f"[{self.dev.timeline_value:3d}] launch {self.name} g={global_size} l={local_size} "
            f"tex={self.tex_cnt} ibo={self.ibo_cnt} samp={self.samp_cnt} sig={self.dev.timeline_signal.value}", flush=True)
    try:
      ret = super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait, timeout=timeout)
    except BaseException:
      # A failed submission must not leave a stale record behind: its snapshots may never have
      # executed, and a later fetch would poll the ALWAYS_ON slot until the 5s deadline.
      if hasattr(self.dev, "_pmc_pending"): del self.dev._pmc_pending[n_pmc:]
      raise
    if self.dev.pmc_enabled:
      # The end snapshot's ALWAYS_ON poll below guarantees the GPU executed this kernel's
      # snapshots before host copyout (profiling-only extra synchronization).
      self.dev.synchronize()
      self.dev._pmc_emit(self.dev._pmc_pending[n_pmc:], self.dev.prof_exec_counter)
      del self.dev._pmc_pending[n_pmc:]
    return ret

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

class QCOMAllocator(HCQAllocatorBase):
  def alloc(self, size:int, options:BufferSpec|None=None):
    # LRU free does not call _free, so cache reuse must retire prior GPU users explicitly.
    if self.dev.gen == 8 and self.cache[(size, options)]: self.dev.synchronize()
    return super().alloc(size, options)

  def _alloc(self, size:int, opts:BufferSpec) -> HCQBuffer:
    uncached = opts.uncached or (self.dev.gen == 8 and opts.cpu_access)
    return self.dev._gpu_map(opts.external_ptr, size) if opts.external_ptr else \
      self.dev._gpu_alloc(size, uncached=uncached, fill_zeroes=bool(getenv("QCOM_ZERO_ALLOC", 0)))

  def _copyin(self, dest:HCQBuffer, src:memoryview):
    self.dev.synchronize()
    with cpu_profile(f"TINY -> {self.dev.device}", f"{self.dev.device}:COPY"):
      ctypes.memmove(dst_addr:=dest.cpu_view().addr, mv_address(src), src.nbytes)
      if self.dev.gen == 8 and getenv("QCOM_COPYIN_DCACHE", 0):
        dcache_flush().fxn(ctypes.c_uint64(line_addr:=dst_addr & ~63), ceildiv(dst_addr + src.nbytes - line_addr, 64))

  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self.dev.synchronize()
    with cpu_profile(f"{self.dev.device} -> TINY", f"{self.dev.device}:COPY"): ctypes.memmove(mv_address(dest), src.cpu_view().addr, src.size)

  def _as_buffer(self, src:HCQBuffer) -> memoryview: return to_mv(src.cpu_view().addr, src.size)

  def _do_free(self, opaque, options:BufferSpec): self.dev._gpu_free(opaque)

def flag(nm, val): return (val << getattr(kgsl, f"{nm}_SHIFT")) & getattr(kgsl, f"{nm}_MASK")

class QCOMDevice(HCQCompiled):
  _stack: HCQBuffer
  _scratch: QCOMScratch
  def __init__(self, device:str=""):
    self.fd = FileIOInterface('/dev/kgsl-3d0', os.O_RDWR)
    info = kgsl.struct_kgsl_devinfo()
    kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.fd, type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    self.chip_id = info.chip_id
    gpu_id = info.gpu_id
    if not gpu_id:
      if self.chip_id == 0x07002000: gpu_id = 702  # A702 has a 7xx marketing ID but uses A6xx.
      elif (major:=(self.chip_id >> 24) & 0xff) < 0x10:
        gpu_id = major * 100 + ((self.chip_id >> 16) & 0xff) * 10 + ((self.chip_id >> 8) & 0xff)
    dev_id = mesa.struct_fd_dev_id(gpu_id, self.chip_id)
    self.dev_info: Any
    if gpu_id // 100 == 6 or gpu_id == 702: self.gen, self.dev_info = 6, None
    else:
      try: raw_dev_info = mesa.fd_dev_info_raw(dev_id)
      except AttributeError as e:
        raise RuntimeError(f"tinymesa required or device unsupported: chip_id={self.chip_id:#x} gpu_id={gpu_id}") from e
      if not raw_dev_info or raw_dev_info.contents.chip == 0:
        raise RuntimeError(f"tinymesa required or device unsupported: chip_id={self.chip_id:#x} gpu_id={gpu_id}")
      self.gen, self.dev_info = raw_dev_info.contents.chip, mesa.fd_dev_info(dev_id)
    gpu_id = dev_id.gpu_id or self.gen * 100
    if self.gen not in (6, 8): raise RuntimeError(f"Unsupported GPU: chip_id={self.chip_id:#x}")
    if self.gen == 6: self.dummy_addr = int(self._gpu_alloc(0x1000).va_addr)

    flags = kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC \
      | flag("KGSL_CONTEXT_PRIORITY", getenv("QCOM_PRIORITY", 8)) | flag("KGSL_CONTEXT_PREEMPT_STYLE", kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN)
    self.ctx = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.fd, flags=flags).drawctxt_id

    self.cmd_buf = self._gpu_alloc(16 << 20, uncached=self.gen == 8)
    self._last_submit_timestamp:int|None = None
    self.cmd_buf_allocator = QCOMRingAllocator(self, self.cmd_buf.size, int(self.cmd_buf.va_addr)) if self.gen == 8 else \
      BumpAllocator(size=self.cmd_buf.size, base=int(self.cmd_buf.va_addr), wrap=True)

    self.border_color_buf = self._gpu_alloc(0x1000, uncached=self.gen == 8, fill_zeroes=True)

    if self.gen == 6: self.last_cmd = 0
    else:
      self._retirement_signals:weakref.WeakValueDictionary[int, QCOMSignal] = weakref.WeakValueDictionary()
      self._submitted_retirement_signals:weakref.WeakSet[QCOMSignal] = weakref.WeakSet()

    # Set max power
    struct.pack_into('IIQQ', pwr:=memoryview(bytearray(0x18)), 0, 1, self.ctx, mv_address(_:=memoryview(array.array('I', [1]))), 4)
    kgsl.IOCTL_KGSL_SETPROPERTY(self.fd, type=kgsl.KGSL_PROP_PWR_CONSTRAINT, value=mv_address(pwr), sizebytes=pwr.nbytes)

    if PROFILE and self.gen == 6:
      System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", value="4000000000", msg="Failed to disable suspend mode", expected="4294967276")

    # Per-kernel PMC snapshots (PROFILE=1 QCOM_PMC=1, or VIZ>=2 which defaults QCOM_PMC on).
    # Counter reservation itself stays lazy on first profiled kernel via dev.perf.
    self.pmc_enabled:bool = bool(PROFILE > 0 and self.gen == 8 and QCOM_PMC.value > 0)

    arch = f"a{gpu_id}{',IMAGE_PITCH_ALIGNMENT=64' if self.gen == 6 and IMAGE else ''}"
    if self.gen == 8 and IMAGE: arch += ",QCOM_IMAGE_PITCH_ALIGNMENT=16"
    if self.gen == 8: arch += f",chip_id={self.chip_id:#x}"
    renderers:list[type[Renderer]] = [QCOMCLRenderer, IR3Renderer] if self.gen == 6 else [IR3Renderer]
    super().__init__(device, QCOMAllocator(self), renderers, QCOMProgram, QCOMSignal, functools.partial(QCOMComputeQueue, self), arch=arch)
    if self.gen == 8: self.kernargs_offset_allocator = QCOMRingAllocator(self, self.kernargs_buf.size)

  def _gpu_alloc(self, size:int, flags:int=0, uncached=False, fill_zeroes=False) -> HCQBuffer:
    flags |= flag("KGSL_MEMALIGN", alignment_hint:=12) | kgsl.KGSL_MEMFLAGS_USE_CPU_MAP
    if uncached: flags |= flag("KGSL_CACHEMODE", kgsl.KGSL_CACHEMODE_UNCACHED)
    elif self.gen == 8: flags |= flag("KGSL_CACHEMODE", kgsl.KGSL_CACHEMODE_WRITEBACK) | kgsl.KGSL_MEMFLAGS_IOCOHERENT

    alloc = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.fd, size=(bosz:=round_up(size, 1<<alignment_hint)), flags=flags, mmapsize=bosz)
    if flags & kgsl.KGSL_MEMFLAGS_IOCOHERENT and not alloc.flags & kgsl.KGSL_MEMFLAGS_IOCOHERENT:
      kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=alloc.id)
      return self._gpu_alloc(size, flags & ~(kgsl.KGSL_CACHEMODE_MASK | kgsl.KGSL_MEMFLAGS_IOCOHERENT), uncached=True, fill_zeroes=fill_zeroes)
    try: va_addr = self.fd.mmap(0, bosz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)
    except BaseException:
      kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=alloc.id)
      raise

    # The hardware image layout may legally address the page-rounded row/array tail beyond a buffer's logical bytes.
    if fill_zeroes: ctypes.memset(va_addr, 0, bosz)
    return HCQBuffer(va_addr=va_addr, size=size, meta=(alloc, True), view=MMIOInterface(va_addr, size, fmt='B'), owner=self)

  def _gpu_map(self, ptr:int, size:int) -> HCQBuffer:
    ptr_aligned, size_aligned = (ptr & ~0xfff), round_up(size + (ptr & 0xfff), 0x1000)
    dcache_flush().fxn(ctypes.c_uint64(ptr_line_aligned:=ptr & ~63), ceildiv(ptr + size - ptr_line_aligned, 64))
    try:
      mi = kgsl.IOCTL_KGSL_MAP_USER_MEM(self.fd, hostptr=ptr_aligned, len=size_aligned, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
      return HCQBuffer(mi.gpuaddr + (ptr - ptr_aligned), size=size, meta=(mi, False), view=MMIOInterface(ptr, size, fmt='B'), owner=self)
    except OSError as e:
      # Preserve the legacy A6xx import path; Gen8 requires a successful mapping.
      if self.gen == 6 and e.errno == errno.EFAULT:
        return HCQBuffer(va_addr=ptr, size=size, meta=(None, False), view=MMIOInterface(ptr, size, fmt='B'), owner=self)
      raise RuntimeError("Failed to map external pointer to GPU memory") from e

  def _gpu_free(self, mem:HCQBuffer):
    if mem.meta[0] is None: return # external (gpu) ptr
    if not mem.meta[1]: kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.fd, gpuaddr=mem.meta[0].gpuaddr) # external (cpu) ptr
    else:
      kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=mem.meta[0].id)
      FileIOInterface.munmap(mem.va_addr, mem.meta[0].mmapsize)

  def _ensure_stack_size(self, sz):
    if self.gen == 8:
      if not hasattr(self, '_scratch') or self._scratch.buf.size < sz: self._scratch = QCOMScratch(self, sz)
      self._stack = self._scratch.buf
      return
    if not hasattr(self, '_stack'): self._stack = self._gpu_alloc(sz)
    elif self._stack.size < sz:
      self.synchronize()
      self._gpu_free(self._stack)
      self._stack = self._gpu_alloc(sz)

  def _wrap_timeline_signal(self):
    super()._wrap_timeline_signal()
    if self.gen == 8:
      for signal in (self.timeline_signal, self._shadow_timeline_signal):
        with signal._retirement_lock: signal._command_timestamps.clear()

  @property
  def perf(self):
    # gen8 counters are kernel-managed; reservation happens lazily on first access.
    if not hasattr(self, '_perf'):
      from tinygrad.runtime.support.qcom_profile import QCOMPerfCounters
      self._perf = QCOMPerfCounters(self) if self.gen == 8 else None
    return self._perf

  def _pmc_alloc_record(self) -> int:
    # Byte offset of a fresh begin/end record slot pair in the PMC ring buffer. Records are
    # consumed in order (eager __call__ immediately, graph captures at profile finalize).
    perf = self.perf
    assert perf is not None, "QCOM PMC requires reserved gen8 counters"
    if not hasattr(self, '_pmc_buf'):
      self._pmc_buf = self._gpu_alloc(perf.slot_vals * 64 * 2 * QCOM_PMC_RING, uncached=True, fill_zeroes=True)
      self._pmc_next, self._pmc_last_ao = 0, None
      self._pmc_pending:list[tuple[int, Any, tuple, tuple]] = []
    if self._pmc_next >= QCOM_PMC_RING: raise RuntimeError(f"exceeded {QCOM_PMC_RING} profiled kernel records")
    off = self._pmc_next * perf.slot_vals * 64 * 2
    self._pmc_next += 1
    return off

  @staticmethod
  def _pmc_info(prg, gsize, lsize) -> dict[str, Any]:
    info:dict[str, Any] = {"global_size": str(tuple(gsize)), "local_size": str(tuple(lsize))}
    for k in ("threadsize", "fregs", "hregs", "shared_size", "instrlen", "constlen_units", "image_size"):
      try:
        if (v:=getattr(prg, k, None)) is not None: info[k] = int(v)
      except (TypeError, ValueError): info[k] = str(getattr(prg, k))
    return info

  def _pmc_emit(self, entries, exec_tag:int):
    if not entries: return
    from tinygrad.device import Compiled
    from tinygrad.runtime.support.qcom_profile import QCOMPMCSample, QCOMProfilePMCEvent
    perf = self.perf
    assert perf is not None
    cpu = self._pmc_buf.cpu_view()
    for off, prg, gsize, lsize in entries:
      deltas, self._pmc_last_ao = perf.fetch_record(cpu, off, self._pmc_last_ao)
      sched = [QCOMPMCSample(name, off=8*i) for i, name in enumerate(perf.names)]
      blob = struct.pack(f"<{len(perf.names)}Q", *(deltas[n] for n in perf.names))
      Compiled.profile_events.append(QCOMProfilePMCEvent(self.device, prg.prof_prg_counter, sched, blob, exec_tag,
                                                         self._pmc_info(prg, gsize, lsize)))

  def _at_profile_finalize(self):
    # Flush graph-captured PMC records (their snapshot IBs replayed with baked slot addresses;
    # slots hold the last replay's values). Eager records were already emitted per __call__.
    if getattr(self, "_pmc_pending", None):
      self.synchronize()
      self._pmc_emit(self._pmc_pending, self.prof_exec_counter)
      self._pmc_pending.clear()
    super()._at_profile_finalize()
    if self.gen == 6:
      with contextlib.suppress(RuntimeError): System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", "10", "Failed to reenable suspend mode")
