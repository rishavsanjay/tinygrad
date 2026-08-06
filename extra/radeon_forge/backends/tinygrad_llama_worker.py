from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

from tinygrad import Device, GlobalCounters, Tensor
from tinygrad.nn.state import get_parameters
from examples.llama3 import Tokenizer, build_transformer


def send(payload):
  sys.stdout.write(json.dumps(payload, default=str) + "\n")
  sys.stdout.flush()


def main() -> None:
  parser = argparse.ArgumentParser(description="Persistent tinygrad Llama backend for Radeon Forge")
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--tokenizer", type=Path)
  parser.add_argument("--size", choices=("1B", "8B", "70B", "405B"), default="1B")
  parser.add_argument("--quantize", choices=("int8", "nf4", "float16", "fp8"))
  parser.add_argument("--max-context", type=int, default=8192)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  if not args.model.exists(): raise FileNotFoundError(args.model)
  tokenizer_path = args.tokenizer or ((args.model if args.model.is_dir() else args.model.parent) / "tokenizer.model")
  if not tokenizer_path.exists(): raise FileNotFoundError(tokenizer_path)
  Tensor.manual_seed(args.seed)

  tokenizer = Tokenizer(str(tokenizer_path))
  device = Device.DEFAULT
  model = build_transformer(args.model, model_size=args.size, quantize=args.quantize, device=device, max_context=args.max_context)
  param_bytes = sum(x.nbytes() for x in get_parameters(model))

  def encode_role(role: str) -> list[int]:
    return [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")

  def encode_message(message) -> list[int]:
    content = message.get("content", "")
    if not isinstance(content, str): content = json.dumps(content, default=str)
    return encode_role(str(message.get("role", "user"))) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"]]

  active_session: str | None = None
  active_tokens: list[int] = []
  send({"kind": "ready", "name": f"tinygrad-llama-{args.size}", "device": str(device), "model": str(args.model),
        "parameter_bytes": param_bytes, "capabilities": {"streaming": True, "structured_tools": False,
        "prefix_cache": True, "persistent_kv": True, "kernel_metrics": True, "cancellation": False}})

  for line in sys.stdin:
    try:
      payload = json.loads(line)
      if payload.get("op") == "shutdown": return
      if payload.get("op") != "generate": raise ValueError("unsupported operation")
      request = payload["request"]
      session_id = str(request["session_id"])
      messages = request["messages"]
      max_tokens = max(1, min(int(request.get("max_tokens", 256)), args.max_context))
      temperature = float(request.get("temperature", 0.0))

      prompt = [tokenizer.bos_id]
      for message in messages: prompt += encode_message(message)
      if not messages or messages[-1].get("role") != "assistant": prompt += encode_role("assistant")
      if len(prompt) >= args.max_context: raise ValueError(f"prompt has {len(prompt)} tokens, max context is {args.max_context}")

      common = 0
      if active_session == session_id:
        for old, new in zip(active_tokens, prompt):
          if old != new: break
          common += 1
      active_session = session_id
      prefill_start = time.perf_counter_ns()
      prefill_gpu_s, prefill_kernels, prefill_mem, prefill_ops = 0.0, 0, 0, 0
      for position in range(common, len(prompt) - 1):
        GlobalCounters.reset()
        model(Tensor([[prompt[position]]], device=device), position, 0.0, 0, 0.0, 0.0, 0.0).realize()
        prefill_gpu_s += GlobalCounters.time_sum_s
        prefill_kernels += GlobalCounters.kernel_count
        prefill_mem += GlobalCounters.global_mem
        prefill_ops += GlobalCounters.global_ops
      prefill_wall_ms = (time.perf_counter_ns() - prefill_start) / 1e6
      active_tokens = list(prompt)
      send({"kind": "prefill", "metrics": {"wall_ms": prefill_wall_ms, "gpu_ms": prefill_gpu_s * 1e3,
            "prompt_tokens": len(prompt), "prefix_reused_tokens": common, "new_prompt_tokens": max(0, len(prompt) - 1 - common),
            "kernel_count": prefill_kernels, "global_mem_bytes": prefill_mem, "global_ops": prefill_ops}})

      start_pos, last_tok = len(prompt) - 1, prompt[-1]
      generated = 0
      for index in range(max_tokens):
        GlobalCounters.reset()
        wall_start = time.perf_counter_ns()
        tok = model(Tensor([[last_tok]], device=device), start_pos, temperature, 0, 0.0, 0.0, 0.0).item()
        wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        gpu_ms = GlobalCounters.time_sum_s * 1e3
        start_pos += 1
        last_tok = tok
        if tok in tokenizer.stop_tokens:
          send({"kind": "done", "finish_reason": "stop", "metrics": {"generated_tokens": generated}})
          break
        active_tokens.append(tok)
        generated += 1
        send({"kind": "token", "text": tokenizer.decode([tok]), "metrics": {"index": index, "wall_ms": wall_ms,
              "gpu_ms": gpu_ms, "kernel_count": GlobalCounters.kernel_count, "global_mem_bytes": GlobalCounters.global_mem,
              "global_ops": GlobalCounters.global_ops, "parameter_bandwidth_gbs": (param_bytes / max(GlobalCounters.time_sum_s, 1e-12)) / 1e9}})
      else: send({"kind": "done", "finish_reason": "length", "metrics": {"generated_tokens": generated}})
    except Exception as exc:
      send({"kind": "error", "error": str(exc), "error_type": type(exc).__name__})


if __name__ == "__main__": main()
