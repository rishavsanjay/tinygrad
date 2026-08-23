import os, unittest
from unittest import mock
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.engine.realize import compile_linear, get_runtime
from tinygrad.helpers import IMAGE
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import QCOMComputeQueue
from tinygrad.uop.ops import Ops

QCOM_IR3 = os.getenv("DEV", "").startswith("QCOM:IR3")

def image_programs(t:Tensor):
  linear = compile_linear(t.schedule_linear())
  return [get_runtime("QCOM", call.src[0], cache=False) for call in linear.src if call.src and call.src[0].op is Ops.PROGRAM]

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
