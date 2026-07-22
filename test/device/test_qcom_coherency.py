"""Focused QCOM CPU/GPU visibility and HCQ replay regressions.

Run on Android with:
  PYTHONPATH=. LIBC_PATH=/system/lib64/libc.so DEV=QCOM:IR3 \
    python -m pytest test/device/test_qcom_coherency.py -x -q
"""
import struct, unittest
from unittest import mock
import numpy as np

from tinygrad import Device, Tensor, TinyJit, Variable, dtypes
from tinygrad.device import Buffer, BufferSpec
from tinygrad.helpers import DEV
from tinygrad.runtime.support.hcq import HCQBuffer, HCQCompiled
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.uop.ops import Ops


QCOM_DEVICE = DEV.device == "QCOM"


def _assert_hcq_graph(test: unittest.TestCase, jit: TinyJit):
  test.assertIsNotNone(jit.captured)
  assert jit.captured is not None
  test.assertTrue(any(call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "graph" for call in jit.captured.linear.src))


@unittest.skipUnless(QCOM_DEVICE and issubclass(type(Device[Device.DEFAULT]), HCQCompiled), "QCOM HCQ device required")
class TestQCOMCoherency(unittest.TestCase):
  def test_terminal_gpu_write_visible_after_synchronize(self):
    output = Tensor.zeros(1 << 18, dtype=dtypes.int32, device=Device.DEFAULT).contiguous().realize()
    for value in (0x1234567, 0x2345678, 0x3456789, 0x456789A):
      output.assign(Tensor.full(output.shape, value, dtype=dtypes.int32, device=Device.DEFAULT)).realize()
      Device[Device.DEFAULT].synchronize()
      got = output.numpy()
      self.assertTrue(np.all(got == value), f"terminal output retained stale data instead of {value:#x}")

  def test_scalar_kernarg_changes_on_graph_replay(self):
    x = Tensor.zeros(1, dtype=dtypes.int32, device=Device.DEFAULT).contiguous().realize()
    scalar = Variable("qcom_scalar", 0, 0xffff)

    @TinyJit
    def add_scalar(inp: Tensor, val):
      intermediate = (inp + val).contiguous().realize()
      return (intermediate + 0).contiguous().realize()

    got = [int(add_scalar(x, scalar.bind(val)).item()) for val in (11, 22, 33, 44, 55)]
    self.assertEqual(got, [11, 22, 33, 44, 55])
    _assert_hcq_graph(self, add_scalar)

  def test_input_buffer_address_changes_on_graph_replay(self):
    @TinyJit
    def transform(inp: Tensor):
      intermediate = (inp * 3).contiguous().realize()
      return (intermediate + 1).contiguous().realize()

    got, addrs = [], []
    for val in (7, 19, 31, 43, 55):
      inp = Tensor([val], dtype=dtypes.int32, device=Device.DEFAULT).contiguous().realize()
      addrs.append(int(inp.uop.buffer._buf.va_addr))
      got.append(int(transform(inp).item()))
    self.assertEqual(got, [22, 58, 94, 130, 166])
    self.assertGreater(len(set(addrs)), 1)
    _assert_hcq_graph(self, transform)

  def test_same_address_cpu_payload_changes(self):
    inp = Tensor.zeros(4, dtype=dtypes.int32, device=Device.DEFAULT).contiguous().realize()

    @TinyJit
    def transform(x: Tensor):
      intermediate = (x * 5).contiguous().realize()
      return (intermediate + 2).contiguous().realize()

    got = []
    for val in (3, 17, 29, 41, 53):
      raw = memoryview(bytearray(struct.pack("4i", *([val] * 4))))
      inp.uop.buffer.copy_from(Buffer("PYTHON", 4, dtypes.int32, opaque=raw))
      got.append(transform(inp).numpy().tolist())
    self.assertEqual(got, [[val * 5 + 2] * 4 for val in (3, 17, 29, 41, 53)])
    _assert_hcq_graph(self, transform)

  def test_gpu_raw_between_graph_kernels(self):
    @TinyJit
    def two_kernels(inp: Tensor):
      intermediate = (inp + 7).contiguous().realize()
      return (intermediate * 9).contiguous().realize()

    got = []
    for val in (2, 13, 24, 35, 46):
      inp = Tensor.full((4096,), val, dtype=dtypes.int32, device=Device.DEFAULT).contiguous().realize()
      got.append(two_kernels(inp).numpy())
    for val, out in zip((2, 13, 24, 35, 46), got): np.testing.assert_equal(out, np.full(4096, (val + 7) * 9, dtype=np.int32))
    _assert_hcq_graph(self, two_kernels)

  def test_symbolic_static_store_changes_position_and_payload(self):
    cache = Tensor.zeros(8, dtype=dtypes.int32, device=Device.DEFAULT).contiguous().realize()
    pos = Variable("qcom_cache_pos", 0, 7)

    @TinyJit
    def store_and_read(value: Tensor, start_pos):
      view = cache[start_pos:start_pos+1]
      assigned = Tensor(cache.uop.after(view.uop.store(value.uop)))
      return (assigned + 0).contiguous().realize()

    for i, val in enumerate((101, 202, 303, 404, 505)):
      value = Tensor([val], dtype=dtypes.int32, device=Device.DEFAULT).contiguous().realize()
      got = store_and_read(value, pos.bind(i)).numpy()
      expected = np.zeros(8, dtype=np.int32)
      expected[:i+1] = np.array((101, 202, 303, 404, 505)[:i+1], dtype=np.int32)
      np.testing.assert_equal(got, expected)
    _assert_hcq_graph(self, store_and_read)

class TestQCOMAllocationOptions(unittest.TestCase):
  def test_uncached_option_is_forwarded(self):
    from tinygrad.runtime.ops_qcom import QCOMAllocator

    class FakeDevice:
      def _gpu_alloc(self, size, **kwargs): return size, kwargs

    allocator = object.__new__(QCOMAllocator)
    allocator.dev = FakeDevice()
    self.assertEqual(allocator._alloc(4096, BufferSpec(uncached=True)), (4096, {"uncached": True}))
    self.assertEqual(allocator._alloc(4096, BufferSpec(cpu_access=True)), (4096, {"uncached": True}))

  def test_writecombine_cache_sync_uses_buffer_range(self):
    from tinygrad.runtime.autogen import kgsl
    from tinygrad.runtime.ops_qcom import QCOMAllocator

    alloc = type("FakeAllocation", (), {"id": 7,
                                         "flags": kgsl.KGSL_CACHEMODE_WRITEBACK << kgsl.KGSL_CACHEMODE_SHIFT})()
    base = HCQBuffer(0x1000, 0x1000, meta=(alloc, True))
    buf = base.offset(0x180, 0x200)
    allocator = object.__new__(QCOMAllocator)
    allocator.dev = type("FakeDevice", (), {"fd": object()})()
    with mock.patch.object(kgsl, "IOCTL_KGSL_GPUMEM_SYNC_CACHE") as sync:
      allocator._sync_cache(buf, kgsl.KGSL_GPUMEM_CACHE_TO_GPU)
    sync.assert_called_once_with(allocator.dev.fd, id=7,
                                 op=kgsl.KGSL_GPUMEM_CACHE_TO_GPU | kgsl.KGSL_GPUMEM_CACHE_RANGE,
                                 offset=0x180, length=0x200)

  def test_bump_allocator_calls_back_before_wrap(self):
    events = []
    allocator = BumpAllocator(16, wrap_callback=lambda: events.append(allocator.ptr))
    self.assertEqual(allocator.alloc(12), 0)
    self.assertEqual(allocator.alloc(8), 0)
    self.assertEqual(events, [12])
    self.assertEqual(allocator.wrap_count, 1)
    self.assertEqual(allocator.total_allocated, 20)

  def test_command_arena_wait_uses_last_submitted_kgsl_timestamp(self):
    from tinygrad.runtime.autogen import kgsl
    from tinygrad.runtime.ops_qcom import QCOMDevice

    fake_device = type("FakeQCOMDevice", (), {"last_cmd": 123, "fd": object(), "ctx": 456})()
    with mock.patch.object(kgsl, "IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID") as wait:
      QCOMDevice._wait_for_command_arena(fake_device)
    wait.assert_called_once_with(fake_device.fd, context_id=456, timestamp=123, timeout=0xffffffff)


if __name__ == "__main__": unittest.main()
