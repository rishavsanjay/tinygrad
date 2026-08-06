from __future__ import annotations

from .workspace import CandidateWorkspace, KernelSpec


def install_default_specs(workspace: CandidateWorkspace) -> list[KernelSpec]:
  specs = [
    KernelSpec(
      name="rdna3-vopd-gemm-schedule",
      operation="C = A @ B; batch-one projection/GEMM scheduling research",
      target="gfx1100", language="python", extension=".py", entrypoint="build_kernel",
      shapes={"M":"multiple of 128", "N":"multiple of 128", "K":"multiple of 128"},
      dtypes={"A":"float32", "B":"float32", "C":"float32"},
      invariants=("Numerically match tinygrad matmul within MSE 1e-6", "No unsupported RDNA3 instructions",
                  "No out-of-workspace file access", "MockGPU timing is never used as a speed metric"),
      objective="minimize median and P95 W7900 latency without correctness loss",
      mockgpu_command=("python3", "{candidate}"), hardware_command=("python3", "{candidate}"),
      metadata={"seed":"extra/gemm/amd_asm_matmul.py", "role":"infrastructure and schedule-synthesis gate",
                "hook":{"layer":"kernel", "target":"batch1_projection_gemm", "mode":"replace", "adapter":"request_metadata",
                        "selector":{"program_name":"generated_gemm"}, "exclusive_group":"projection_gemm",
                        "description":"Validated kernel selection forwarded to a backend-specific kernel adapter."}},
    ),
    KernelSpec(
      name="batch1-decode-megakernel",
      operation="One specialized transformer decode block for batch-one private-agent inference",
      target="gfx1100", language="python", extension=".py", entrypoint="build_kernel",
      shapes={"batch":1, "sequence":"decode token", "model":"fixed by workload contract"},
      dtypes={"activations":"bf16/fp16", "accumulation":"fp32 where required"},
      invariants=("Match the trusted tinygrad block reference", "Preserve KV-cache semantics", "No spills unless explicitly allowed",
                  "Generated code is disposable; specification and oracle are authoritative"),
      objective="minimize inter-token latency and kernel launches/token",
      metadata={"status":"oracle command must be bound after selecting the model", "role":"final technical target",
                "hook":{"layer":"transformer_block", "target":"llama.decode.block", "mode":"replace",
                        "adapter":"python_transformer_block", "selector":{"indices":"all", "phase":"decode"},
                        "exclusive_group":"decode_transformer_block",
                        "description":"Replace selected tinygrad Llama blocks inside the isolated model process."}},
    ),
  ]
  existing = {x.spec_id for x in workspace.specs()}
  for spec in specs:
    if spec.spec_id not in existing: workspace.save_spec(spec)
  return specs
