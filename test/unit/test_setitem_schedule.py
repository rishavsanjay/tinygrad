import unittest
from tinygrad import Tensor, Device, TinyJit, dtypes, GlobalCounters
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
    assert_kernel_count(2 if Device.DEFAULT.startswith(("PYTHON", "NPY", "DISK", "CL", "WEBGPU")) else 1)
    self.assertListEqual(a.tolist(), [1.0]*10 + [13.0, 14.0] + [1.0]*8)

  def test_cross_device_slice_copy_targets_destination_view(self):
    a = Tensor(list(range(30)), device="CPU").realize()
    b = Tensor(list(range(30)), device="CPU:1").realize()
    out = a[10:12].assign(b[13:15].to(a.device))
    linear = out.schedule_linear()
    self.assertEqual(len(linear.src), 1)
    call = linear.src[0]
    self.assertIs(call.src[0].op, Ops.COPY)
    self.assertEqual(len(call.src), 3)
    self.assertEqual(call.src[1].contiguous_view_offset(), 10)
    self.assertEqual(call.src[1].max_numel(), 2)
    self.assertTrue(call.src[1].storage_base.is_realized and call.src[2].storage_base.is_realized)
    run_linear(linear)
    self.assertListEqual(a.tolist(), list(range(10)) + [13, 14] + list(range(12, 30)))

  def test_cross_device_contiguous_slice_compositions(self):
    cases = [
      ("plain", 60, lambda x:x[10:15], (5,), 10, 5),
      ("reshape_rows", 30, lambda x:x.reshape(6, 5)[2:4, :], (2, 5), 10, 10),
      ("nested_reshape", 60, lambda x:x.reshape(4, 5, 3)[1:3].reshape(30), (30,), 15, 30),
      ("singleton_permute", 60, lambda x:x.reshape(12, 1, 5)[2:3].permute(1, 0, 2).reshape(5), (5,), 10, 5),
    ]
    for name, base_size, make_view, src_shape, offset, size in cases:
      with self.subTest(name=name):
        a = Tensor(list(range(base_size)), device="CPU").realize()
        b = Tensor(list(range(100, 100+size)), device="CPU:1").reshape(src_shape).realize()
        out = make_view(a).assign(b.to(a.device))
        linear = out.schedule_linear()
        self.assertEqual(len(linear.src), 1)
        self.assertIs(linear.src[0].src[0].op, Ops.COPY)
        run_linear(linear)
        expected = list(range(base_size))
        expected[offset:offset+size] = list(range(100, 100+size))
        self.assertListEqual(a.flatten().tolist(), expected)

  def test_cross_device_noncontiguous_slices_fall_back(self):
    cases = [
      ("transpose", 16, lambda x:x.reshape(4, 4).T[1], (4,), (1, 5, 9, 13)),
      ("stride", 12, lambda x:x[1:9:2], (4,), (1, 3, 5, 7)),
      ("row_gaps", 20, lambda x:x.reshape(4, 5)[:, 1:3], (4, 2), (1, 2, 6, 7, 11, 12, 16, 17)),
      ("flip", 10, lambda x:x[2:6].flip(0), (4,), (5, 4, 3, 2)),
    ]
    for name, base_size, make_view, src_shape, indices in cases:
      with self.subTest(name=name):
        a = Tensor(list(range(base_size)), device="CPU").realize()
        b = Tensor(list(range(100, 100+len(indices))), device="CPU:1").reshape(src_shape).realize()
        out = make_view(a).assign(b.to(a.device))
        linear = out.schedule_linear()
        self.assertGreaterEqual(len(linear.src), 2)
        run_linear(linear)
        expected = list(range(base_size))
        for idx, val in zip(indices, range(100, 100+len(indices))): expected[idx] = val
        self.assertListEqual(a.flatten().tolist(), expected)

  def test_cross_device_noncontiguous_source_falls_back(self):
    a = Tensor.zeros(2, dtype=dtypes.int32, device="CPU").realize()
    b = Tensor(list(range(8)), dtype=dtypes.int32, device="CPU:1").reshape(2, 4).T[1]
    out = a.assign(b.to(a.device))
    linear = out.schedule_linear()
    self.assertGreaterEqual(len(linear.src), 2)
    run_linear(linear)
    self.assertListEqual(a.tolist(), [1, 5])

  def test_cross_device_multiple_slice_copies(self):
    a = Tensor(list(range(8)), device="CPU").realize()
    b = Tensor(list(range(10, 18)), device="CPU:1").realize()
    a[0:4].assign(b[:4].to(a.device))
    out = a[2:6].assign(b[4:].to(a.device))
    linear = out.schedule_linear()
    self.assertEqual(len(linear.src), 2)
    self.assertTrue(all(call.src[0].op is Ops.COPY for call in linear.src))
    run_linear(linear)
    self.assertListEqual(a.tolist(), [10, 11, 14, 15, 16, 17, 6, 7])

  def test_cross_device_shared_copy_falls_back(self):
    a = Tensor.zeros(6, dtype=dtypes.int32, device="CPU").realize()
    b = Tensor([7, 8], device="CPU:1").realize()
    rhs = b.to(a.device)
    a[0:2].assign(rhs)
    out = a[4:6].assign(rhs)
    linear = out.schedule_linear()
    self.assertEqual([call.src[0].op for call in linear.src], [Ops.COPY, Ops.SINK, Ops.SINK])
    run_linear(linear)
    self.assertListEqual(a.tolist(), [7, 8, 0, 0, 7, 8])
    self.assertListEqual(rhs.tolist(), [7, 8])

  def test_cross_device_copy_snapshot_before_source_write(self):
    src = Tensor([10, 11, 12, 13], dtype=dtypes.int32, device="CPU:1").realize()
    new = Tensor([90, 91], dtype=dtypes.int32, device="CPU:1").realize()
    dst = Tensor.zeros(6, dtype=dtypes.int32, device="CPU").realize()
    snapshot = src[1:3].to(dst.device)
    src[1:3].assign(new)
    out = dst[2:4].assign(snapshot)
    linear = out.schedule_linear(src)
    self.assertEqual([call.src[0].op for call in linear.src], [Ops.COPY, Ops.SINK])
    run_linear(linear)
    self.assertListEqual(dst.tolist(), [0, 0, 11, 12, 0, 0])
    self.assertListEqual(src.tolist(), [10, 90, 91, 13])

  def test_cross_device_copy_snapshot_before_alias_write(self):
    src = Tensor([10, 11, 12, 13, 14, 15], dtype=dtypes.int32, device="CPU:1").realize()
    alias = src.reshape(2, 3)
    new = Tensor([90, 91], dtype=dtypes.int32, device="CPU:1").realize()
    dst = Tensor.zeros(7, dtype=dtypes.int32, device="CPU").realize()
    snapshot = alias.flatten()[2:5].to(dst.device)
    alias[0, 1:3].assign(new)
    out = dst[3:6].assign(snapshot)
    linear = out.schedule_linear(alias)
    self.assertEqual([call.src[0].op for call in linear.src], [Ops.COPY, Ops.SINK])
    run_linear(linear)
    self.assertListEqual(dst.tolist(), [0, 0, 0, 12, 13, 14, 0])
    self.assertListEqual(src.tolist(), [10, 90, 91, 13, 14, 15])

  def test_cross_device_snapshot_in_mixed_expression(self):
    src = Tensor([10, 11, 12, 13], device="CPU:1").realize()
    snapshot = src[1:3].to("CPU")
    mixed = snapshot.to(src.device) + src[1:3]
    src[1:3].assign(Tensor([90, 91], device=src.device))
    self.assertListEqual(snapshot.tolist(), [11, 12])
    self.assertListEqual(mixed.tolist(), [101, 103])

  def test_cross_device_slice_copy_jit_rebinds_destination_view(self):
    @TinyJit
    def f(dst:Tensor, src:Tensor): dst[1:3].assign(src.to(dst.device)).realize()
    outputs = []
    for fill, vals in enumerate(([11, 12], [21, 22], [31, 32], [41, 42])):
      dst = Tensor.full(4, fill, dtype=dtypes.int32, device="CPU").realize()
      src = Tensor(vals, dtype=dtypes.int32, device="CPU:1").realize()
      f(dst, src)
      outputs.append((dst, [fill, *vals, fill]))
    for dst, expected in outputs: self.assertListEqual(dst.tolist(), expected)
    self.assertEqual(len(f.captured._linear.src), 1)
    call = f.captured._linear.src[0]
    self.assertIs(call.src[0].op, Ops.COPY)
    self.assertIs(call.src[1].op, Ops.SHRINK)
    self.assertIs(call.src[1].src[0].op, Ops.PARAM)
    self.assertEqual(call.src[1].contiguous_view_offset(), 1)

if __name__ == '__main__':
  unittest.main()
