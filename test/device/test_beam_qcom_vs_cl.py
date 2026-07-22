import unittest
import time
from tinygrad import Tensor, Device, TinyJit, Context
from tinygrad.nn import Linear, LayerNorm

# A standard Transformer Layer (Multi-Head Self Attention + FFN)


class TransformerBlock:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Attention Projections
        self.qkv = Linear(embed_dim, 3 * embed_dim)
        self.out = Linear(embed_dim, embed_dim)

        # Feed Forward Network Projections
        self.ff1 = Linear(embed_dim, 4 * embed_dim)
        self.ff2 = Linear(4 * embed_dim, embed_dim)

        # Normalization
        self.ln_1 = LayerNorm(embed_dim)
        self.ln_2 = LayerNorm(embed_dim)

    def __call__(self, x: Tensor):
        B, T, C = x.shape

        # 1. Multi-Head Self Attention
        # Shape: (B, T, 3 * C) -> (B, T, 3, num_heads, head_dim) -> (3, B, num_heads, T, head_dim)
        qkv = self.qkv(self.ln_1(x)).reshape(
            B, T, 3, self.num_heads, self.head_dim).transpose(2, 1).transpose(1, 0)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention: (B, num_heads, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.softmax(axis=-1)

        # Output Projection
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.out(y)

        # 2. Feed Forward Network (with GELU activation)
        x = x + self.ff2(self.ff1(self.ln_2(x)).gelu())
        return x


class TestTransformerBeam(unittest.TestCase):
    def benchmark_transformer(self, device_name: str, iters: int = 50) -> float:
        print(f"\n--- Initializing Transformer Block on {device_name} ---")

        # Force the device globally so all layer weights instantiate directly on the metal
        with Context(DEV=device_name):
            # Dim=256, Heads=8 is small enough for fast BEAM search, big enough to stress the GPU
            block = TransformerBlock(embed_dim=256, num_heads=8)

            # Dummy sequence (Batch=1, SeqLen=128, EmbedDim=256)
            x = Tensor.randn(1, 128, 256).realize()

        @TinyJit
        def run_block(inputs):
            return block(inputs).realize()

        print(f"--- Warming up and BEAM searching on {device_name} ---")
        with Context(DEBUG=1, DEV=device_name):
            _ = run_block(x)
            Device[device_name].synchronize()

        print(f"--- Running {device_name} ---")
        Device[device_name].synchronize()
        st = time.perf_counter()

        for _ in range(iters):
            _ = run_block(x)

        Device[device_name].synchronize()
        et = time.perf_counter()

        avg_time = (et - st) / iters
        print(f"{device_name} average block time ({
              iters} runs): {avg_time:.6f} seconds")
        return avg_time

    def test_transformer_comparison(self):
        qcom_time = self.benchmark_transformer("QCOM:IR3")
        cl_time = self.benchmark_transformer("CL")

        print(f"\nSummary: QCOM:IR3={qcom_time:.6f}s, CL={cl_time:.6f}s")


if __name__ == "__main__":
    unittest.main()
