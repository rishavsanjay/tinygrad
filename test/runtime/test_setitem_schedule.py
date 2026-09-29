import unittest
from tinygrad import Tensor, Device, dtypes, GlobalCounters
from tinygrad.engine.realize import run_linear
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

  @unittest.skipUnless(Device.DEFAULT != "CPU", "source must be on another device")
  def test_setitem_slice_assign_from_other_device(self):
    # CL and WEBGPU cannot use buffer views, so they also materialize the source slice.
    a = Tensor.ones(20, device="CPU")
    b = Tensor.arange(20).float().clone()
    Tensor.realize(a, b)
    GlobalCounters.reset()
    a[10:12].assign(b[13:15].to(a.device)).realize()
    assert_kernel_count(2 if Device.DEFAULT in {"CL", "WEBGPU"} else 1)
    self.assertListEqual(a.tolist(), [1.0]*10 + [13.0, 14.0] + [1.0]*8)

  def test_cross_device_copy_contiguous_view(self):
    dst = Tensor.full((4, 4), -1, dtype=dtypes.int32).contiguous().realize()
    src = Tensor.arange(12, dtype=dtypes.int32).clone("CPU:1" if Device.DEFAULT == "CPU" else "CPU").realize()
    view = dst.reshape(4, 1, 4)[1:3].permute(1, 0, 2).reshape(2, 4)
    linear = view.assign(src[2:10].reshape(2, 4).to(dst.device)).schedule_linear()
    self.assertEqual(len(linear.src), 2 if Device.DEFAULT in {"CL", "WEBGPU"} else 1)
    run_linear(linear)
    self.assertListEqual(dst.tolist(), [[-1]*4, [2, 3, 4, 5], [6, 7, 8, 9], [-1]*4])

  def test_cross_device_copy_noncontiguous_view(self):
    a = Tensor.zeros(4, 4).contiguous().realize()
    b = Tensor.arange(4, dtype=dtypes.float32).reshape(2, 2).clone("CPU:1" if Device.DEFAULT == "CPU" else "CPU").realize()
    linear = a[1:3, 1:3].assign(b.to(a.device)).schedule_linear()
    self.assertEqual(len(linear.src), 2)
    run_linear(linear)
    self.assertListEqual(a.tolist(), [[0.0]*4, [0.0, 0.0, 1.0, 0.0], [0.0, 2.0, 3.0, 0.0], [0.0]*4])

if __name__ == '__main__':
  unittest.main()
