import os, unittest
import numpy as np

from tinygrad import Tensor
from tinygrad.helpers import Context, IMAGE

QCOM_IR3 = os.getenv("DEV", "").startswith("QCOM:IR3")

@unittest.skipUnless(QCOM_IR3 and IMAGE.value, "run with DEV=QCOM:IR3 IMAGE=1 or IMAGE=2")
class TestQCOMImageConv(unittest.TestCase):
  @staticmethod
  def _conv(device, image, x, w, stride=1, padding=0, groups=1, dtype=np.float32):
    with Context(IMAGE=image):
      return Tensor(x.astype(dtype), device=device).conv2d(Tensor(w.astype(dtype), device=device), stride=stride,
                                                            padding=padding, groups=groups).numpy()

  def _check(self, cin, cout, kernel, stride, padding, groups=1, height=11, width=17, dtype=np.float32):
    rng = np.random.default_rng(cin*1000 + cout*100 + kernel*10 + stride + groups)
    x = rng.standard_normal((1, cin, height, width), dtype=np.float32)
    w = rng.standard_normal((cout, cin//groups, kernel, kernel), dtype=np.float32)
    ref = self._conv("CPU", 0, x, w, stride, padding, groups, dtype)
    buf = self._conv("QCOM:IR3", 0, x, w, stride, padding, groups, dtype)
    img = self._conv("QCOM:IR3", IMAGE.value, x, w, stride, padding, groups, dtype)
    atol, rtol = ((2e-2, 2e-2) if dtype == np.float16 else (2e-4, 2e-4))
    np.testing.assert_allclose(buf, ref, atol=atol, rtol=rtol)
    np.testing.assert_allclose(img, ref, atol=atol, rtol=rtol)

  def test_regular(self):
    for kernel in (1, 3, 5):
      for stride in (1, 2):
        for padding in (0, kernel//2):
          args = (4, 4, kernel, stride, padding)
          with self.subTest(args=args): self._check(*args)
    for args in ((5, 7, 3, 1, 1), (5, 7, 5, 2, 0)):
      with self.subTest(args=args): self._check(*args)

  def test_grouped(self):
    self._check(6, 6, 3, 1, 1, groups=2)
    self._check(8, 12, 5, 2, 0, groups=4)

  def test_depthwise(self):
    self._check(5, 5, 3, 1, 1, groups=5)
    self._check(8, 8, 5, 2, 0, groups=8)

  def test_fp16(self): self._check(4, 4, 3, 1, 1, dtype=np.float16)

  def test_xy_rgba_and_border(self):
    # A depthwise identity distinguishes every X/Y/channel lane; padding forces clamp-to-border reads.
    x = np.arange(1*4*5*7, dtype=np.float32).reshape(1, 4, 5, 7)
    w = np.zeros((4, 1, 3, 3), dtype=np.float32)
    w[:, 0, 1, 1] = 1
    out = self._conv("QCOM:IR3", IMAGE.value, x, w, padding=1, groups=4)
    np.testing.assert_array_equal(out, x)

if __name__ == "__main__": unittest.main()
