from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tinygrad import Device, GlobalCounters, Tensor, dtypes
from extra.models.llama import TransformerBlock, precompute_freqs_cis


@dataclass(frozen=True)
class HarnessConfig:
  dim: int = 128
  hidden_dim: int = 256
  n_heads: int = 4
  n_kv_heads: int = 4
  max_context: int = 128
  prompt_tokens: int = 8
  dtype: str = "float32"
  max_abs_error: float = 2e-4
  max_rel_error: float = 2e-3


class ModelShell:
  def __init__(self, layer: Any, config: HarnessConfig):
    self.layers = [layer]
    self.max_context = config.max_context
    self.forward_jit = None


def _candidate_path(argument: str | None) -> Path:
  value = argument or os.environ.get("RADEON_FORGE_CANDIDATE")
  if not value: raise ValueError("candidate path is required")
  path = Path(value).resolve()
  if not path.is_file(): raise FileNotFoundError(path)
  return path


def _parameters() -> dict[str, Any]:
  raw = os.environ.get("RADEON_FORGE_CANDIDATE_JSON", "")
  if not raw: return {}
  payload = json.loads(raw)
  values = payload.get("parameters", {}) if isinstance(payload, Mapping) else {}
  if not isinstance(values, Mapping): raise ValueError("candidate parameters must be an object")
  return dict(values)


def _load_candidate(path: Path, parameters: Mapping[str, Any]):
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  spec = importlib.util.spec_from_file_location(f"radeon_forge_candidate_{digest[:16]}", path)
  if spec is None or spec.loader is None: raise RuntimeError(f"cannot import candidate {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  setattr(module, "RADEON_FORGE_PARAMETERS", dict(parameters))
  configure = getattr(module, "configure", None)
  if configure is not None:
    if not callable(configure): raise TypeError("optional configure must be callable")
    configure(dict(parameters))
  factory = getattr(module, "build_replacement", None)
  if not callable(factory): raise TypeError("candidate must define build_replacement(original, layer_index, model, context)")
  return factory


def _block(config: HarnessConfig) -> TransformerBlock:
  return TransformerBlock(config.dim, config.hidden_dim, config.n_heads, config.n_kv_heads,
                          norm_eps=1e-5, max_context=config.max_context)


def _reset_cache(block: TransformerBlock) -> None:
  attention = getattr(block, "attention", None)
  if attention is not None and hasattr(attention, "cache_kv"): delattr(attention, "cache_kv")


def _dtype(name: str):
  return {"float32":dtypes.float32, "float16":dtypes.float16, "bfloat16":dtypes.bfloat16}[name]


def _inputs(config: HarnessConfig, prompt_tokens: int, seed: int):
  Tensor.manual_seed(seed)
  dtype = _dtype(config.dtype)
  prompt = Tensor.randn(1, prompt_tokens, config.dim).cast(dtype).contiguous().realize()
  first = Tensor.randn(1, 1, config.dim).cast(dtype).contiguous().realize()
  decode = Tensor.randn(1, 1, config.dim).cast(dtype).contiguous().realize()
  freqs = precompute_freqs_cis(config.dim // config.n_heads, config.max_context * 2).cast(dtype).contiguous().realize()
  mask = Tensor.full((1, 1, prompt_tokens, prompt_tokens), float("-inf"), dtype=dtype).triu(1)
  return prompt, first, decode, freqs, mask


def _run_transition(callable_block, config: HarnessConfig, prompt_tokens: int, seed: int) -> list[np.ndarray]:
  prompt, first, decode, freqs, mask = _inputs(config, prompt_tokens, seed)
  outputs = [
    callable_block(prompt, 0, freqs[:, :prompt_tokens], mask).realize().numpy(),
    callable_block(first, prompt_tokens, freqs[:, prompt_tokens:prompt_tokens+1], None).realize().numpy(),
    callable_block(decode, prompt_tokens+1, freqs[:, prompt_tokens+1:prompt_tokens+2], None).realize().numpy(),
  ]
  return outputs


def _error(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict[str, Any]:
  max_abs = max(float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64)))) for left, right in zip(reference, candidate))
  max_rel = 0.0
  checked = 0
  for left, right in zip(reference, candidate):
    left64, right64 = left.astype(np.float64), right.astype(np.float64)
    denominator = np.maximum(np.abs(left64), 1e-8)
    max_rel = max(max_rel, float(np.max(np.abs(left64-right64)/denominator)))
    checked += left.size
  return {"max_abs_error":max_abs, "max_rel_error":max_rel, "checked_values":checked}


def validate(candidate_path: Path, config: HarnessConfig, prompt_lengths: tuple[int, ...], parameters: Mapping[str, Any]) -> dict[str, Any]:
  block = _block(config)
  model = ModelShell(block, config)
  factory = _load_candidate(candidate_path, parameters)
  max_abs = max_rel = 0.0
  checked = 0
  cases = []
  for case_index, prompt_tokens in enumerate(prompt_lengths):
    _reset_cache(block)
    reference = _run_transition(block, config, prompt_tokens, seed=1000+case_index)
    _reset_cache(block)
    context = {"stage":"transition_oracle", "prompt_tokens":prompt_tokens, "batch_size":1,
               "parameters":dict(parameters), "device":Device.DEFAULT}
    replacement = factory(block, 0, model, context)
    if not callable(replacement): raise TypeError("build_replacement must return a callable block")
    candidate = _run_transition(replacement, config, prompt_tokens, seed=1000+case_index)
    result = _error(reference, candidate)
    max_abs, max_rel = max(max_abs, result["max_abs_error"]), max(max_rel, result["max_rel_error"])
    checked += result["checked_values"]
    cases.append({"prompt_tokens":prompt_tokens, **result})
  passed = max_abs <= config.max_abs_error and max_rel <= config.max_rel_error
  return {"passed":passed, "max_abs_error":max_abs, "max_rel_error":max_rel, "checked_values":checked,
          "reason":"" if passed else "candidate diverged from transformer-block transition reference", "cases":cases}


def benchmark(candidate_path: Path, config: HarnessConfig, budget: int, parameters: Mapping[str, Any]) -> tuple[list[float], dict[str, Any]]:
  block = _block(config)
  model = ModelShell(block, config)
  factory = _load_candidate(candidate_path, parameters)
  replacement = factory(block, 0, model, {"stage":"decode_benchmark", "parameters":dict(parameters), "device":Device.DEFAULT})
  if not callable(replacement): raise TypeError("build_replacement must return a callable block")
  _reset_cache(block)
  prompt, _, _, freqs, mask = _inputs(config, config.prompt_tokens, seed=2026)
  replacement(prompt, 0, freqs[:, :config.prompt_tokens], mask).realize()
  samples = []
  kernel_counts = []
  for index in range(budget):
    position = config.prompt_tokens + index
    Tensor.manual_seed(3000+index)
    value = Tensor.randn(1, 1, config.dim).cast(_dtype(config.dtype)).contiguous().realize()
    GlobalCounters.reset()
    started = time.perf_counter_ns()
    replacement(value, position, freqs[:, position:position+1], None).realize()
    samples.append((time.perf_counter_ns()-started)/1000.0)
    kernel_counts.append(GlobalCounters.kernel_count)
  return samples, {"kernel_count_mean":sum(kernel_counts)/len(kernel_counts), "device":Device.DEFAULT,
                   "prompt_tokens":config.prompt_tokens, "stage":"decode", "parameters":dict(parameters)}


def main() -> None:
  parser = argparse.ArgumentParser(description="Radeon Forge transformer-block oracle and benchmark")
  parser.add_argument("--candidate")
  parser.add_argument("--mode", choices=("correctness", "benchmark", "transition"), default="correctness")
  parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
  parser.add_argument("--dim", type=int, default=128)
  parser.add_argument("--hidden-dim", type=int, default=256)
  parser.add_argument("--prompt-tokens", type=int, default=8)
  args = parser.parse_args()
  config = HarnessConfig(dim=args.dim, hidden_dim=args.hidden_dim, prompt_tokens=args.prompt_tokens, dtype=args.dtype)
  candidate_path, parameters = _candidate_path(args.candidate), _parameters()
  prompt_lengths = (1, 4, args.prompt_tokens, min(24, config.max_context-2)) if args.mode == "transition" else (args.prompt_tokens,)
  correctness = validate(candidate_path, config, tuple(dict.fromkeys(prompt_lengths)), parameters)
  budget = max(1, int(os.environ.get("RADEON_FORGE_BUDGET", "5")))
  samples, metrics = (benchmark(candidate_path, config, budget, parameters) if args.mode == "benchmark" and correctness["passed"] else ([], {}))
  payload = {
    "samples_us":samples,
    "correctness":correctness,
    "resources":{},
    "compile_ok":True,
    "stable":bool(correctness["passed"]),
    "metrics":metrics,
    "mode":args.mode,
    "candidate_sha256":hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
  }
  print(json.dumps(payload, default=str))
  raise SystemExit(0 if correctness["passed"] else 2)


if __name__ == "__main__": main()
