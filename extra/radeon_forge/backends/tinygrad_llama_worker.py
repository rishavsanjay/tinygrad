from __future__ import annotations

import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
from typing import Any, Callable, Mapping

from tinygrad import Context, Device, GlobalCounters, Tensor
from tinygrad.device import Compiled
from tinygrad.nn.state import get_parameters
from examples.llama3 import Tokenizer, build_transformer

from ..synthesis.hooks import ExecutionContext, ExecutionStage
from .stage_hooks import ModelStageHookRuntime


MAX_PROFILE_EVENTS_PER_PHASE = 8192


def send(payload):
  sys.stdout.write(json.dumps(payload, default=str) + "\n")
  sys.stdout.flush()


def _name(value: Any) -> str: return str(getattr(value, "display_name", value))


def _collect_profile_events(start: int) -> list[dict[str, Any]]:
  """Flatten tinygrad device profile ranges into portable kernel evidence."""
  raw = list(Compiled.profile_events[start:])
  del Compiled.profile_events[start:]
  ret: list[dict[str, Any]] = []
  for event in raw:
    if hasattr(event, "ents") and hasattr(event, "sigs"):
      for entry in event.ents:
        try: duration_us = float(event.sigs[entry.en_id] - event.sigs[entry.st_id])
        except Exception: continue
        device = str(getattr(entry, "device", "UNKNOWN"))
        if device in {"CPU", "TINY"} or duration_us < 0: continue
        ret.append({"name": _name(getattr(entry, "name", "kernel")), "device": device,
                    "duration_ms": duration_us / 1000.0, "profile_event_type": type(event).__name__,
                    "source": "tinygrad.Compiled.profile_events"})
    elif hasattr(event, "st") and hasattr(event, "en") and getattr(event, "en") is not None:
      try: duration_us = float(event.en - event.st)
      except Exception: continue
      device = str(getattr(event, "device", "UNKNOWN"))
      if device in {"CPU", "TINY"} or duration_us < 0: continue
      ret.append({"name": _name(getattr(event, "name", "kernel")), "device": device,
                  "duration_ms": duration_us / 1000.0, "profile_event_type": type(event).__name__,
                  "source": "tinygrad.Compiled.profile_events"})
  return ret


def _profiled(fn: Callable[[], Any]) -> tuple[Any, list[dict[str, Any]]]:
  start = len(Compiled.profile_events)
  try:
    with Context(PROFILE=1): result = fn()
  except BaseException:
    _collect_profile_events(start)
    raise
  return result, _collect_profile_events(start)


def _emit_kernel_events(events: list[dict[str, Any]], *, stage: str, token_index: int | None = None) -> tuple[int, bool]:
  truncated = len(events) > MAX_PROFILE_EVENTS_PER_PHASE
  emitted = events[:MAX_PROFILE_EVENTS_PER_PHASE]
  for sequence, event in enumerate(emitted):
    metrics = {**event, "stage": stage, "sequence": sequence}
    if token_index is not None: metrics["token_index"] = token_index
    send({"kind": "kernel", "metrics": metrics})
  return len(emitted), truncated


def _weight_identity(path: Path, size: str, quantize: str | None) -> str:
  try:
    stat = path.stat()
    identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{size}:{quantize}"
  except OSError: identity = f"{path}:{size}:{quantize}"
  return hashlib.sha256(identity.encode()).hexdigest()


def _shape(value: Any) -> list[int | str]:
  try: return [int(item) for item in value.shape]
  except Exception: return [str(getattr(value, "shape", "unknown"))]


def _architecture_identity(model: Any, size: str, quantize: str | None) -> tuple[dict[str, Any], str]:
  first = model.layers[0] if getattr(model, "layers", None) else None
  attention = getattr(first, "attention", None)
  feed_forward = getattr(first, "feed_forward", None)
  payload = {
    "family": "llama", "size_label": size, "quantize": quantize or "model_default",
    "layers": len(getattr(model, "layers", ())), "max_context": int(getattr(model, "max_context", 0)),
    "n_heads": int(getattr(attention, "n_heads", 0)), "n_kv_heads": int(getattr(attention, "n_kv_heads", 0)),
    "head_dim": int(getattr(attention, "head_dim", 0)),
    "embedding_shape": _shape(getattr(getattr(model, "tok_embeddings", None), "weight", None)),
    "output_shape": _shape(getattr(getattr(model, "output", None), "weight", None)),
    "ffn_w1_shape": _shape(getattr(getattr(feed_forward, "w1", None), "weight", None)),
  }
  return payload, hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _git_revision() -> str:
  try:
    root = Path(__file__).resolve().parents[3]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
  except Exception: return "unknown"


def _hook_event(runtime: ModelStageHookRuntime, hooks: list[Mapping[str, Any]], context: ExecutionContext) -> None:
  result = runtime.apply(hooks, context)
  if result.get("changed") or result.get("rolled_back"):
    send({"kind": "hook", "metrics": {**result, "context": context.flattened()}})


def _prefill_without_output_head(model: Any, token_ids: list[int], start_pos: int, device: str) -> None:
  """Populate KV state for a prompt chunk without projecting every prompt token to vocabulary logits."""
  if not token_ids: return
  tokens = Tensor([token_ids], device=device)
  _, seqlen = tokens.shape
  h = model.tok_embeddings(tokens).contiguous()
  freqs_cis = model.freqs_cis.cast(h.dtype)[:, start_pos:start_pos+seqlen, :, :, :]
  mask = (Tensor.full((1, 1, seqlen, start_pos+seqlen), float("-inf"), dtype=h.dtype, device=h.device).triu(start_pos+1)
          if model.max_context != 0 and seqlen > 1 else None)
  for layer in model.layers: h = layer(h, start_pos, freqs_cis, mask)
  h.realize()


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
  architecture_data, architecture_hash = _architecture_identity(model, args.size, args.quantize)
  hook_runtime = ModelStageHookRuntime(model)
  device_obj = Device[device]
  architecture = str(getattr(device_obj, "arch", ""))

  def encode_role(role: str) -> list[int]:
    return [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")

  def encode_message(message) -> list[int]:
    content = message.get("content", "")
    if not isinstance(content, str): content = json.dumps(content, default=str)
    return encode_role(str(message.get("role", "user"))) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"]]

  active_session: str | None = None
  cached_tokens: list[int] = []  # only tokens whose KV entries are actually materialized
  send({"kind": "ready", "name": f"tinygrad-llama-{args.size}", "device": str(device), "gpu": str(device),
        "architecture": architecture, "runtime": "tinygrad", "runtime_revision": _git_revision(),
        "model": str(args.model), "model_family": "llama", "model_hash": _weight_identity(args.model, args.size, args.quantize),
        "model_architecture_hash": architecture_hash, "model_architecture": architecture_data,
        "dtype": args.quantize or "model_default", "shapes": {"batch": 1, "max_context": args.max_context},
        "parameter_bytes": param_bytes, "capabilities": {"streaming": True, "structured_tools": False,
        "prefix_cache": True, "persistent_kv": True, "kernel_metrics": True, "cancellation": False}})

  for line in sys.stdin:
    try:
      payload = json.loads(line)
      if payload.get("op") == "shutdown":
        hook_runtime.close()
        return
      if payload.get("op") != "generate": raise ValueError("unsupported operation")
      request = payload["request"]
      metadata = request.get("metadata", {})
      hooks = metadata.get("active_hooks", [])
      if not isinstance(hooks, list): hooks = []
      session_id = str(request["session_id"])
      messages = request["messages"]
      max_tokens = max(1, min(int(request.get("max_tokens", 256)), args.max_context))
      temperature = float(request.get("temperature", 0.0))
      tool_round = int(metadata.get("tool_round", 0))
      resume_after_tool = bool(metadata.get("resume_after_tool", False))

      prompt = [tokenizer.bos_id]
      for message in messages: prompt += encode_message(message)
      if not messages or messages[-1].get("role") != "assistant": prompt += encode_role("assistant")
      if len(prompt) >= args.max_context: raise ValueError(f"prompt has {len(prompt)} tokens, max context is {args.max_context}")

      if active_session != session_id:
        active_session, cached_tokens = session_id, []
      common = 0
      for old, new in zip(cached_tokens, prompt):
        if old != new: break
        common += 1

      prefill_context = ExecutionContext(ExecutionStage.PREFILL, batch_size=1, prompt_tokens=len(prompt),
        context_tokens=max(0, len(prompt)-1), generated_token_index=-1, prefix_reused_tokens=common,
        tool_round=tool_round, warm=bool(cached_tokens), attributes={"resume_after_tool": resume_after_tool, "session_id": session_id})
      _hook_event(hook_runtime, hooks, prefill_context)

      prefill_ids = list(prompt[common:-1])
      GlobalCounters.reset()
      prefill_start = time.perf_counter_ns()
      _, prefill_profile = _profiled(lambda: _prefill_without_output_head(model, prefill_ids, common, device))
      prefill_wall_ms = (time.perf_counter_ns() - prefill_start) / 1e6
      prefill_gpu_ms = GlobalCounters.time_sum_s * 1e3
      cached_tokens = list(prompt[:-1])
      send({"kind": "prefill", "metrics": {"wall_ms": prefill_wall_ms, "gpu_ms": prefill_gpu_ms,
            "prompt_tokens": len(prompt), "prefix_reused_tokens": common, "new_prompt_tokens": len(prefill_ids),
            "prefill_chunk_tokens": len(prefill_ids), "prefix_cache_hit_ratio": common / max(1, len(prompt)-1),
            "kernel_count": GlobalCounters.kernel_count, "global_mem_bytes": GlobalCounters.global_mem,
            "global_ops": GlobalCounters.global_ops, "profile_kernel_events": len(prefill_profile),
            "resume_after_tool": resume_after_tool, "output_head_skipped": True}})
      emitted, prefill_truncated = _emit_kernel_events(prefill_profile, stage="prefill")
      if prefill_truncated:
        send({"kind": "metric", "metrics": {"name": "profile_truncated", "stage": "prefill",
              "captured": emitted, "available": len(prefill_profile)}})

      start_pos, last_tok = len(prompt) - 1, prompt[-1]
      generated = 0
      for index in range(max_tokens):
        stage = ExecutionStage.FIRST_TOKEN if index == 0 else ExecutionStage.DECODE
        decode_context = ExecutionContext(stage, batch_size=1, prompt_tokens=len(prompt), context_tokens=start_pos,
          generated_token_index=index, prefix_reused_tokens=common, tool_round=tool_round, warm=True,
          attributes={"resume_after_tool": resume_after_tool, "session_id": session_id})
        _hook_event(hook_runtime, hooks, decode_context)
        GlobalCounters.reset()
        wall_start = time.perf_counter_ns()
        input_tok = last_tok
        tok, profile = _profiled(lambda: model(Tensor([[input_tok]], device=device), start_pos, temperature, 0, 0.0, 0.0, 0.0).item())
        wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        gpu_ms = GlobalCounters.time_sum_s * 1e3
        cached_tokens.append(input_tok)  # this model invocation materialized input_tok's KV entry
        start_pos += 1
        last_tok = tok
        if tok in tokenizer.stop_tokens:
          send({"kind": "done", "finish_reason": "stop", "metrics": {"generated_tokens": generated,
                "materialized_kv_tokens": len(cached_tokens)}})
          break
        generated += 1
        send({"kind": "token", "text": tokenizer.decode([tok]), "metrics": {"index": index, "stage": stage.value,
              "wall_ms": wall_ms, "gpu_ms": gpu_ms, "kernel_count": GlobalCounters.kernel_count,
              "global_mem_bytes": GlobalCounters.global_mem, "global_ops": GlobalCounters.global_ops,
              "profile_kernel_events": len(profile), "context_tokens": start_pos,
              "parameter_bandwidth_gbs": (param_bytes / max(GlobalCounters.time_sum_s, 1e-12)) / 1e9}})
        emitted, truncated = _emit_kernel_events(profile, stage=stage.value, token_index=index)
        if truncated:
          send({"kind": "metric", "metrics": {"name": "profile_truncated", "stage": stage.value, "token_index": index,
                "captured": emitted, "available": len(profile)}})
      else:
        send({"kind": "done", "finish_reason": "length", "metrics": {"generated_tokens": generated,
              "materialized_kv_tokens": len(cached_tokens), "pending_uncached_output_token": True}})
    except Exception as exc:
      send({"kind": "error", "error": str(exc), "error_type": type(exc).__name__})


if __name__ == "__main__": main()
