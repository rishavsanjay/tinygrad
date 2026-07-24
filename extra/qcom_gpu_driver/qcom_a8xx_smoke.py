"""Compact A8xx QCOM buffer/image smoke test for Android."""
import argparse, os, platform, subprocess, time
import numpy as np
from tinygrad import Context, Device, Tensor, TinyJit

def sync_time(fxn, dev, iterations=20):
  fxn()
  Device[dev].synchronize()
  start = time.perf_counter()
  for _ in range(iterations): fxn()
  Device[dev].synchronize()
  return (time.perf_counter() - start) * 1e3 / iterations

def main(latency=False):
  dev = "QCOM:IR3"
  device = Device[dev]
  revision = os.getenv("TINYGRAD_REVISION")
  if revision is None:
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], text=True, capture_output=True, check=False)
    revision = git.stdout.strip() if git.returncode == 0 else "unknown"
  print(f"device={device.arch.split(',')[0]} python={platform.python_version()} tinygrad={revision}")

  with Context(IMAGE=0):
    got = ((Tensor(np.arange(256), device=dev).reshape(16, 16) * 3) + 1).numpy()
  np.testing.assert_equal(got, np.arange(256).reshape(16, 16) * 3 + 1)
  print("QCOM buffer basic-op PASS")

  values = np.arange(7 * 16 * 4, dtype=np.float32).reshape(7, 16, 4)
  with Context(IMAGE=2):
    image = (Tensor(values, device=dev) + 1).contiguous().realize()
    got = (image * 2).contiguous().numpy()
  np.testing.assert_equal(got, (values + 1) * 2)
  print("QCOM image round-trip PASS")

  x = np.arange(1 * 4 * 7 * 9, dtype=np.float32).reshape(1, 4, 7, 9) / 32
  w = np.zeros((4, 1, 3, 3), dtype=np.float32)
  w[:, 0, 1, 1] = 1
  with Context(IMAGE=2): got = Tensor(x, device=dev).conv2d(Tensor(w, device=dev), padding=1, groups=4).numpy()
  np.testing.assert_allclose(got, x, rtol=1e-5, atol=1e-5)
  print("QCOM convolution PASS")

  @TinyJit
  def replay(inp): return ((inp * 5) + 2).contiguous().realize()
  for value in (3, 7, 11):
    got = replay(Tensor.full((256,), value, device=dev)).numpy()
    np.testing.assert_equal(got, np.full(256, value * 5 + 2))
  print("TinyJit replay PASS")

  if latency:
    x = Tensor(values, device=dev)
    with Context(IMAGE=0): buffer_ms = sync_time(lambda: (x + 1).contiguous().realize(), dev)
    with Context(IMAGE=2): image_ms = sync_time(lambda: (x + 1).contiguous().realize(), dev)
    print(f"latency buffer={buffer_ms:.3f}ms image={image_ms:.3f}ms")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--latency", action="store_true")
  main(parser.parse_args().latency)
