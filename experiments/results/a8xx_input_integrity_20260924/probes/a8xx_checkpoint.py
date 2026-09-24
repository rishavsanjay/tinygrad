"""Collect comparable ONNX node values on one tinygrad device."""
import argparse
import json
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.helpers import FLOAT16, OPENPILOT_HACKS, JIT_BATCH_SIZE
from experiments.a830_openpilot import deterministic_inputs

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--seed", type=int, required=True)
p.add_argument("--output", required=True)
p.add_argument("--indices", required=True)
p.add_argument("--prefix-seed", type=int)
a = p.parse_args()
r = OnnxRunner(a.model)
prefix_inputs = deterministic_inputs(r, a.prefix_seed) if a.prefix_seed is not None else None
inputs = deterministic_inputs(r, a.seed)
if prefix_inputs is not None:
  prefix_out = next(iter(r(prefix_inputs).values())).cast(dtypes.float).numpy().copy()
  print(f"prefix seed={a.prefix_seed} shape={prefix_out.shape}", flush=True)
out = next(iter(r(inputs).values())).cast(dtypes.float).numpy().copy()
arrays = {"final": out}
records = []
for index in (int(v) for v in a.indices.split(",")):
  node = r.graph_nodes[index]
  for name in node.outputs:
    value = r.graph_values[name]
    if not isinstance(value, Tensor): continue
    arr = value.cast(dtypes.float).numpy().copy()
    key = f"{index}:{name}"
    arrays[key] = arr
    records.append({"index": index, "op": node.op, "name": name, "shape": arr.shape,
                    "min": float(np.min(arr)), "max": float(np.max(arr)), "finite": bool(np.isfinite(arr).all())})
    print(f"{key} {node.op} shape={arr.shape} finite={np.isfinite(arr).all()}", flush=True)
np.savez_compressed(a.output + ".npz", **arrays)
with open(a.output + ".json", "w") as f:
  json.dump({"device": Device.DEFAULT, "seed": a.seed,
             "context": {"FLOAT16": FLOAT16.value, "OPENPILOT_HACKS": OPENPILOT_HACKS.value, "JIT_BATCH_SIZE": JIT_BATCH_SIZE.value},
             "records": records}, f, indent=2)
print(f"saved {a.output}", flush=True)
