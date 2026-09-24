import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from test.helpers import call_is_graph
from test.device.test_qcom_image_exec import QCOM_IR3

@unittest.skipUnless(QCOM_IR3, 'run with DEV=QCOM:IR3')
class TestQCOMQualification(unittest.TestCase):
  def test_integer_widths(self):
    for dtype in (dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64):
      with self.subTest(dtype=dtype):
        x = Tensor([0, 1, 7, 15], device='QCOM', dtype=dtype)
        self.assertEqual((x*2+1).tolist(), [1, 3, 15, 31])
        self.assertEqual((x < 7).tolist(), [True, True, False, False])
    x = Tensor([2**33+7, 2**34+3], device='QCOM', dtype=dtypes.int64)
    self.assertEqual((x+5).tolist(), [2**33+12, 2**34+8])

  def test_attention_and_gradients(self):
    rng = np.random.default_rng(830)
    a, b = rng.normal(size=(7,16)).astype(np.float32), rng.normal(size=(16,9)).astype(np.float32)
    x, w = Tensor(a, device='QCOM'), Tensor(b, device='QCOM')
    logits = x @ w
    logits.square().sum().backward()
    result = logits.softmax(-1).numpy()
    ref = a @ b
    exp = np.exp(ref - ref.max(axis=-1, keepdims=True))
    np.testing.assert_allclose(result, exp/exp.sum(axis=-1, keepdims=True), atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(x.grad.numpy(), 2*(a@b)@b.T, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(w.grad.numpy(), 2*a.T@(a@b), atol=2e-4, rtol=2e-4)

  def test_long_lived_graph(self):
    dev = Device['QCOM']
    @TinyJit
    def run(x): return ((x*3).realize()+2).realize()
    sources = [Tensor([float(i+offset) for i in range(257)], device='QCOM').realize() for offset in (0, 7, -3)]
    run(sources[0])
    run(sources[0])
    self.assertTrue(any(call_is_graph(call) for call in run.captured.linear.src))
    for i in range(5000):
      offset = (0, 7, -3)[i % len(sources)]
      result = run(sources[i % len(sources)])
      if i % 100 == 0:
        self.assertEqual(result.tolist(), [(j+offset)*3+2 for j in range(257)])
    self.assertEqual(result.tolist(), [(j+offset)*3+2 for j in range(257)])
    dev.synchronize()
    self.assertEqual(sum(len(s._command_timestamps) for s in dev._submitted_retirement_signals), 0)

if __name__ == '__main__': unittest.main()
