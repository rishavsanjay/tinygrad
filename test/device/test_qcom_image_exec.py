import os, unittest
from unittest import mock
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Buffer
from tinygrad.engine.realize import compile_linear, get_runtime
from tinygrad.helpers import Context, IMAGE
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import QCOMComputeQueue
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

QCOM_IR3 = os.getenv("DEV", "").startswith("QCOM:IR3")

def image_programs(t:Tensor):
  linear = compile_linear(t.schedule_linear())
  return [get_runtime("QCOM", call.src[0], cache=False) for call in linear.src if call.src and call.src[0].op is Ops.PROGRAM]

def multi_image_kernel(out_add:UOp, in_a:UOp, out_mul:UOp, in_b:UOp) -> UOp:
  y, x, c = UOp.range(out_add.shape[0], 0), UOp.range(out_add.shape[1], 1), UOp.range(4, 2, AxisType.UPCAST)
  add = out_add[y, x, c].store(in_a[y, x, c] + in_b[y, x, c])
  mul = out_mul[y, x, c].store(in_a[y, x, c] * in_b[y, x, c])
  return UOp.group(add, mul).end(y, x, c).sink(arg=KernelInfo(name="qcom_multi_image"))

def read_write_image_kernel(image:UOp) -> UOp:
  y, x, c = UOp.range(image.shape[0], 0), UOp.range(image.shape[1], 1), UOp.range(4, 2, AxisType.UPCAST)
  return image[y, x, c].store(image[y, x, c] * 2 + 1).end(y, x, c).sink(arg=KernelInfo(name="qcom_read_write_image"))

@unittest.skipUnless(QCOM_IR3, "run with DEV=QCOM:IR3")
class TestQCOMImageExecution(unittest.TestCase):
  def test_image_mode_and_descriptor_counts(self):
    values = np.arange(7*16*4, dtype=np.float32).reshape(7, 16, 4)
    def make_probe(): return (Tensor(values, device="QCOM").contiguous().realize() + 1).contiguous()
    programs, probe = image_programs(make_probe()), make_probe()
    self.assertTrue(programs)
    image_regs = (mesa.REG_A6XX_SP_CS_TSIZE, mesa.REG_A6XX_SP_CS_USIZE, mesa.REG_A6XX_SP_CS_SAMPLER_BASE,
                  mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, mesa.REG_A6XX_SP_CS_UAV_BASE, mesa.REG_A7XX_SP_CS_UAV_BASE)
    register_writes = []
    original_reg = QCOMComputeQueue.reg
    def record_reg(queue, reg, *values):
      if reg in image_regs: register_writes.append((reg, values))
      return original_reg(queue, reg, *values)
    with mock.patch.object(QCOMComputeQueue, "reg", record_reg): probe.numpy()
    image_writes = [(reg, values[0]) for reg,values in register_writes]
    if IMAGE.value == 0:
      self.assertNotIn("QCOM_IMAGE_PITCH_ALIGNMENT", Device["QCOM"].arch)
      self.assertTrue(all((p.tex_cnt, p.ibo_cnt) == (0, 0) for p in programs))
      self.assertEqual(image_writes, [])
    elif IMAGE.value == 1:
      self.assertIn("QCOM_IMAGE_PITCH_ALIGNMENT=16", Device["QCOM"].arch)
      self.assertTrue(any(p.samp_cnt == p.tex_cnt > 0 for p in programs))
      self.assertTrue(any(reg == mesa.REG_A6XX_SP_CS_TSIZE and count > 0 for reg,count in image_writes))
    elif IMAGE.value == 2:
      self.assertIn("QCOM_IMAGE_PITCH_ALIGNMENT=16", Device["QCOM"].arch)
      self.assertTrue(any(p.tex_cnt > 0 and p.ibo_cnt > 0 and p.samp_cnt == p.tex_cnt for p in programs))
      regs = {reg for reg,_ in image_writes}
      self.assertTrue({mesa.REG_A6XX_SP_CS_TSIZE, mesa.REG_A6XX_SP_CS_USIZE, mesa.REG_A6XX_SP_CS_SAMPLER_BASE,
                       mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, mesa.REG_A7XX_SP_CS_UAV_BASE}.issubset(regs), image_writes)
      self.assertNotIn(mesa.REG_A6XX_SP_CS_UAV_BASE, regs)

  @unittest.skipUnless(IMAGE.value in (1, 2), "run with IMAGE=1 or IMAGE=2")
  def test_multiple_sampled_images_and_buffer_interop(self):
    rng = np.random.default_rng(830)
    for dtype,npdtype,atol in ((dtypes.half, np.float16, 3e-2), (dtypes.float, np.float32, 1e-6)):
      values = [rng.standard_normal((5, 16, 4), dtype=np.float32).astype(npdtype) for _ in range(3)]
      images = [Tensor(v, device="QCOM", dtype=dtype).contiguous().realize() for v in values]
      # IR3 image arithmetic is float32 even when the resource is FP16; exercise a matching scalar buffer input.
      bias = Tensor([1.75], device="QCOM", dtype=dtypes.float).contiguous().realize()
      def make_out(): return ((images[0] + images[1]) * images[2] + bias).contiguous()
      programs, out = image_programs(make_out()), make_out()
      self.assertTrue(any(p.tex_cnt >= 3 and p.ibo_cnt >= 4 for p in programs), [(p.tex_cnt, p.ibo_cnt) for p in programs])
      np.testing.assert_allclose(out.numpy(), (values[0] + values[1]) * values[2] + 1.75, atol=atol, rtol=atol)

  @unittest.skipUnless(IMAGE.value == 2, "run with IMAGE=2")
  def test_interleaved_multiple_sampled_and_storage_images(self):
    a = np.arange(5*16*4, dtype=np.float32).reshape(5, 16, 4) / 17
    b = np.flip(a, axis=1).copy() + 0.5
    in_a, in_b = Tensor(a, device="QCOM").contiguous().realize(), Tensor(b, device="QCOM").contiguous().realize()
    def make_outputs():
      results = Tensor.custom_kernel(Tensor.empty(*a.shape, device="QCOM"), in_a, Tensor.empty(*a.shape, device="QCOM"), in_b,
                                     fxn=multi_image_kernel)
      return results[0], results[2]
    programs, (out_add, out_mul) = image_programs(make_outputs()[0]), make_outputs()
    self.assertTrue(any(p.tex_cnt == 2 and p.ibo_cnt == 4 for p in programs), [(p.tex_cnt, p.ibo_cnt) for p in programs])
    Tensor.realize(out_add, out_mul)
    np.testing.assert_allclose(out_add.numpy(), a + b, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(out_mul.numpy(), a * b, atol=1e-6, rtol=1e-6)

  @unittest.skipUnless(IMAGE.value == 2, "run with IMAGE=2")
  def test_coherent_read_write_same_image(self):
    values = np.arange(5*16*4, dtype=np.float32).reshape(5, 16, 4) / 11
    programs = image_programs(Tensor.custom_kernel(Tensor.empty(*values.shape, device="QCOM"), fxn=read_write_image_kernel)[0])
    image = Tensor.custom_kernel(Tensor(values, device="QCOM").contiguous().realize(), fxn=read_write_image_kernel)[0]
    self.assertTrue(any(p.tex_cnt == 0 and p.ibo_cnt == 1 for p in programs), [(p.tex_cnt, p.ibo_cnt) for p in programs])
    np.testing.assert_allclose(image.numpy(), values * 2 + 1, atol=1e-6, rtol=1e-6)

  @unittest.skipUnless(IMAGE.value == 2, "run with IMAGE=2")
  def test_aligned_subbuffer_image_view_and_misaligned_rejection(self):
    values = np.arange(5*16*4, dtype=np.float32).reshape(5, 16, 4)
    base = Buffer("QCOM", 528, dtypes.float).allocate()  # 64-byte offset plus the 2,048-byte padded image layout
    view = base.view(values.size, dtypes.float, 64).allocate()
    image = Tensor(UOp.from_buffer(view).reshape(values.shape))
    with Context(IMAGE=0): image.assign(Tensor(values, device="QCOM")).realize()
    np.testing.assert_allclose((image + 1).contiguous().numpy(), values + 1, atol=1e-6, rtol=1e-6)

    bad_view = base.view(values.size, dtypes.float, 16).allocate()
    bad_image = Tensor(UOp.from_buffer(bad_view).reshape(values.shape))
    with self.assertRaisesRegex(ValueError, "unaligned QCOM image address"):
      (bad_image + 1).contiguous().realize()

  @unittest.skipUnless(IMAGE.value == 2, "run with IMAGE=2")
  def test_fp16_fp32_load_store_arithmetic(self):
    for dtype, npdtype, atol in ((dtypes.half, np.float16, 2e-3), (dtypes.float, np.float32, 1e-6)):
      values = np.arange(7*16*4, dtype=npdtype).reshape(7, 16, 4) / 8
      got = (((Tensor(values, device="QCOM", dtype=dtype) + 1).contiguous().realize() * 2) + 3).contiguous().numpy()
      np.testing.assert_allclose(got, (values + 1) * 2 + 3, atol=atol, rtol=atol)

    # A scalar buffer is ineligible for image lowering, so this mixes a sampled/storage image with a regular buffer argument.
    values = np.arange(7*16*4, dtype=np.float32).reshape(7, 16, 4) / 8
    bias = Tensor([2], device="QCOM", dtype=dtypes.float).realize()
    got = ((Tensor(values, device="QCOM") + 1).contiguous().realize() * bias).contiguous().numpy()
    np.testing.assert_allclose(got, (values + 1) * 2, atol=1e-6, rtol=1e-6)

  @unittest.skipUnless(IMAGE.value == 2, "run with IMAGE=2")
  def test_small_odd_dimensions_and_zero_border(self):
    x = np.arange(1*4*5*7, dtype=np.float32).reshape(1, 4, 5, 7)
    w = np.zeros((4, 1, 3, 3), dtype=np.float32)
    w[:, 0, 1, 1] = 1
    got = Tensor(x, device="QCOM").conv2d(Tensor(w, device="QCOM"), padding=1, groups=4).numpy()
    np.testing.assert_array_equal(got, x)

  @unittest.skipUnless(IMAGE.value == 2, "run with IMAGE=2")
  def test_symbolic_image_address_replay(self):
    @TinyJit
    def transform(inp): return ((inp + 1).contiguous().realize() * 2).contiguous().realize()
    addrs = []
    for value in range(1, 17, 2):
      inp = Tensor.full((7, 16, 4), value, device="QCOM", dtype=dtypes.float).contiguous().realize()
      addrs.append(int(inp.uop.buffer._buf.va_addr))
      np.testing.assert_equal(transform(inp).numpy(), np.full((7, 16, 4), (value + 1) * 2, dtype=np.float32))
    self.assertGreater(len(set(addrs)), 1)

  @unittest.skipUnless(IMAGE.value == 2, "run with IMAGE=2")
  def test_multiple_symbolic_image_addresses_replay(self):
    @TinyJit
    def transform(out_add, a, out_mul, b):
      results = Tensor.custom_kernel(out_add, a, out_mul, b, fxn=multi_image_kernel)
      Tensor.realize(results[0], results[2])
      return results[0], results[2]
    address_sets = []
    for value in range(1, 12, 2):
      a = Tensor.full((5, 16, 4), value, device="QCOM").contiguous().realize()
      b = Tensor.full((5, 16, 4), value / 2, device="QCOM").contiguous().realize()
      out_add, out_mul = (Tensor.zeros(5, 16, 4, device="QCOM").contiguous().realize() for _ in range(2))
      address_sets.append(tuple(int(t.uop.buffer._buf.va_addr) for t in (out_add, a, out_mul, b)))
      transform(out_add, a, out_mul, b)
      np.testing.assert_allclose(out_add.numpy(), np.full((5, 16, 4), value + value/2), atol=1e-6)
      np.testing.assert_allclose(out_mul.numpy(), np.full((5, 16, 4), value * value/2), atol=1e-6)
    self.assertGreater(len(set(address_sets)), 1)

@unittest.skipUnless(QCOM_IR3 and IMAGE.value in (1, 2), "run with DEV=QCOM:IR3 IMAGE=1 or IMAGE=2")
class TestQCOMImageConv(unittest.TestCase):
  def check(self, cin, cout, groups):
    rng = np.random.default_rng(cin*100 + cout*10 + groups)
    x = rng.standard_normal((1, cin, 9, 13), dtype=np.float32)
    w = rng.standard_normal((cout, cin//groups, 3, 3), dtype=np.float32)
    ref = Tensor(x, device="CPU").conv2d(Tensor(w, device="CPU"), padding=1, groups=groups).numpy()
    got = Tensor(x, device="QCOM").conv2d(Tensor(w, device="QCOM"), padding=1, groups=groups).numpy()
    np.testing.assert_allclose(got, ref, atol=3e-4, rtol=3e-4)

  def test_regular(self): self.check(4, 8, 1)
  def test_grouped(self): self.check(8, 12, 4)
  def test_depthwise(self): self.check(8, 8, 8)

if __name__ == "__main__": unittest.main()
