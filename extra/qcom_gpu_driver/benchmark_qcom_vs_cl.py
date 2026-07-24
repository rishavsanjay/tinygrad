"""Optional QCOM:IR3 versus CL transformer benchmark; not a correctness test."""
import argparse, time
from tinygrad import Context, Device, Tensor, TinyJit
from tinygrad.nn import LayerNorm, Linear

class TransformerBlock:
  def __init__(self, embed_dim=256, num_heads=8):
    self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
    self.qkv, self.out = Linear(embed_dim, 3 * embed_dim), Linear(embed_dim, embed_dim)
    self.ff1, self.ff2 = Linear(embed_dim, 4 * embed_dim), Linear(4 * embed_dim, embed_dim)
    self.ln1, self.ln2 = LayerNorm(embed_dim), LayerNorm(embed_dim)

  def __call__(self, x:Tensor):
    batch, tokens, channels = x.shape
    qkv = self.qkv(self.ln1(x)).reshape(batch, tokens, 3, self.num_heads, self.head_dim).transpose(2, 1).transpose(1, 0)
    q, k, v = qkv[0], qkv[1], qkv[2]
    y = (((q @ k.transpose(-2, -1)) * self.head_dim**-0.5).softmax(-1) @ v).transpose(1, 2).reshape(batch, tokens, channels)
    x = x + self.out(y)
    return x + self.ff2(self.ff1(self.ln2(x)).gelu())

def benchmark(device:str, iterations:int) -> float:
  with Context(DEV=device): block, x = TransformerBlock(), Tensor.randn(1, 128, 256).realize()

  @TinyJit
  def run(inp): return block(inp).realize()

  run(x)
  Device[device].synchronize()
  start = time.perf_counter()
  for _ in range(iterations): run(x)
  Device[device].synchronize()
  return (time.perf_counter() - start) / iterations

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--iterations", type=int, default=50)
  args = parser.parse_args()
  for dev in ("QCOM:IR3", "CL"): print(f"{dev}: {benchmark(dev, args.iterations) * 1e3:.3f} ms")
