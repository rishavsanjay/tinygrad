import unittest
from tinygrad import Tensor, Device, TinyJit, Variable, dtypes, GlobalCounters
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import Ops
from test.helpers import assert_kernel_count

class TestSetitemInto(unittest.TestCase):
  def test_setitem_into_unrealized(self):
    GlobalCounters.reset()
    t = Tensor.arange(4, dtype=dtypes.int32).reshape(2, 2)
    assert_kernel_count(0)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(t.tolist(), [[0, 1], [5, 5]])

  def test_setitem_into_unrealized_sliced_compute(self):
    # base computation contains SHRINK from prior slicing (like QR decomposition pattern)
    GlobalCounters.reset()
    a = Tensor.arange(8, dtype=dtypes.int32).reshape(2, 4)
    w = a[0] + a[1]  # unrealized ADD with SHRINK in graph: [4, 6, 8, 10]
    assert_kernel_count(0)
    w[1] = 99
    assert_kernel_count(0)
    w.realize()
    assert_kernel_count(0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(w.tolist(), [4, 99, 8, 10])

  def test_setitem_into_empty(self):
    GlobalCounters.reset()
    t = Tensor.empty(4, dtype=dtypes.int32)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 4)
    t[1].realize()
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(t[1].item(), 5)

  def test_setitem_into_empty_alu(self):
    GlobalCounters.reset()
    t = Tensor.empty(4, dtype=dtypes.int32) + 1
    assert_kernel_count(0)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertLessEqual(GlobalCounters.global_mem, 32)
    t[1].realize()
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(t[1].item(), 5)

  def test_setitem_into_tensor(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.int32).realize()
    GlobalCounters.reset()
    t[1] = 5
    assert_kernel_count(0)
    t[1].realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 4)
    t.realize()
    assert_kernel_count(1)
    self.assertListEqual(t.tolist(), [1, 5, 3, 4])

  def test_setitem_into_tensor_alu(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.int32).realize() + 1
    GlobalCounters.reset()
    t[1] = 5
    assert_kernel_count(0)
    t[1].realize()
    assert_kernel_count(1)
    self.assertLessEqual(GlobalCounters.global_mem, 32)
    t[1].realize()
    t.realize()
    assert_kernel_count(1)
    self.assertListEqual(t.tolist(), [2, 5, 4, 5])

  def test_setitem_into_const(self):
    GlobalCounters.reset()
    t = Tensor.ones(4, dtype=dtypes.int32, buffer=False)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(t.tolist(), [1, 5, 1, 1])

  def test_setitem_into_const_alu(self):
    GlobalCounters.reset()
    t = Tensor.ones(4, dtype=dtypes.int32, buffer=False) + 1
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(t.tolist(), [2, 5, 2, 2])

  def test_setitem_into_arange(self):
    # NOTE: arange has no real buffer, but assigning to it is fine
    GlobalCounters.reset()
    other = Tensor.arange(4, dtype=dtypes.int32)
    t = Tensor.arange(4, dtype=dtypes.int32)
    self.assertIs(other.uop, t.uop)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(0)
    self.assertListEqual(t.tolist(), [0, 5, 2, 3])

  def test_setitem_slice_const(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[20:50] = 3
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 30*4)  # 30 elements written

  def test_setitem_slice_tensor(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    v = Tensor.zeros(30, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[20:50] = v
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 30*4*2)  # 30 read + 30 written

  def test_setitem_full(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[:] = 3
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 100*4)  # full buffer written

  @unittest.skipUnless(Device.DEFAULT != "CPU", "source must be on another device")
  def test_setitem_slice_assign_from_other_device(self):
    a = Tensor.ones(20, device="CPU")
    b = Tensor.arange(20).float().clone()
    Tensor.realize(a, b)
    GlobalCounters.reset()
    a[10:12].assign(b[13:15].to(a.device)).realize()
    assert_kernel_count(2 if Device.DEFAULT.startswith(("PYTHON", "NPY", "DISK", "CL", "WEBGPU", "NULL")) else 1)
    self.assertListEqual(a.tolist(), [1.0]*10 + [13.0, 14.0] + [1.0]*8)

  def test_cross_device_slice_copy_targets_destination_view(self):
    dst = Tensor(list(range(8)), dtype=dtypes.int32, device="CPU:1").realize()
    src = Tensor(list(range(10, 18)), dtype=dtypes.int32, device="CPU").realize()
    linear = dst[6:8].assign(src[3:5].to(dst.device)).schedule_linear()
    self.assertEqual(len(linear.src), 1)
    call = linear.src[0]
    self.assertIs(call.src[0].op, Ops.COPY)
    self.assertEqual((call.src[1].contiguous_view_offset(), call.src[2].contiguous_view_offset()), (6, 3))
    self.assertEqual(call.src[1].max_numel(), 2)
    run_linear(linear)
    self.assertListEqual(dst.tolist(), [0, 1, 2, 3, 4, 5, 13, 14])

  def test_cross_device_contiguous_reshape_uses_single_copy(self):
    dst = Tensor(list(range(30)), device="CPU").realize()
    src = Tensor(list(range(100, 110)), device="CPU:1").reshape(2, 5).realize()
    linear = dst.reshape(6, 5)[2:4, :].assign(src.to(dst.device)).schedule_linear()
    self.assertEqual([x.src[0].op for x in linear.src], [Ops.COPY])
    run_linear(linear)
    self.assertListEqual(dst.tolist(), list(range(10)) + list(range(100, 110)) + list(range(20, 30)))

  def test_cross_device_noncontiguous_destination_falls_back(self):
    dst = Tensor(list(range(12)), device="CPU").realize()
    src = Tensor(list(range(100, 104)), device="CPU:1").realize()
    linear = dst[1:9:2].assign(src.to(dst.device)).schedule_linear()
    self.assertGreaterEqual(len(linear.src), 2)
    run_linear(linear)
    self.assertListEqual(dst.tolist(), [0, 100, 2, 101, 4, 102, 6, 103, 8, 9, 10, 11])

  def test_cross_device_noncontiguous_source_falls_back(self):
    dst = Tensor.zeros(2, dtype=dtypes.int32, device="CPU").realize()
    src = Tensor(list(range(8)), dtype=dtypes.int32, device="CPU:1").reshape(2, 4).T[1]
    linear = dst.assign(src.to(dst.device)).schedule_linear()
    self.assertGreaterEqual(len(linear.src), 2)
    run_linear(linear)
    self.assertListEqual(dst.tolist(), [1, 5])

  def test_cross_device_shared_copy_falls_back(self):
    dst = Tensor.zeros(6, dtype=dtypes.int32, device="CPU").realize()
    rhs = Tensor([7, 8], device="CPU:1").realize().to(dst.device)
    dst[0:2].assign(rhs)
    linear = dst[4:6].assign(rhs).schedule_linear()
    self.assertGreaterEqual(len(linear.src), 2)
    run_linear(linear)
    self.assertListEqual(dst.tolist(), [7, 8, 0, 0, 7, 8])
    self.assertListEqual(rhs.tolist(), [7, 8])

  def test_symbolic_length_cross_device_slice_does_not_overcopy(self):
    @TinyJit
    def write(dst:Tensor, src:Tensor, v):
      dst.shrink(((7, 7+v),)).assign(src.shrink(((9, 9+v),)).to(dst.device)).realize()
    for i, value in enumerate((1, 4, 2)):
      before = [-100-i]*20
      values = list(range(100*i, 100*i+24))
      dst = Tensor(before, dtype=dtypes.int32, device="CPU").realize()
      src = Tensor(values, dtype=dtypes.int32, device="CPU:1").realize()
      write(dst, src, Variable("slice_len", 1, 4).bind(value))
      expected = before[:]
      expected[7:7+value] = values[9:9+value]
      self.assertListEqual(dst.tolist(), expected)

if __name__ == '__main__':
  unittest.main()