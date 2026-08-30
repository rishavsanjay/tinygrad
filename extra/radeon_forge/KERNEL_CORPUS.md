# Kernel and optimization corpus

This file records the provenance and intended use of initial Radeon Forge implementation families. Inclusion does not imply that a kernel is correct or fast on the W7900. Every candidate must still pass target compilation, resource, numerical, benchmark, and held-out gates.

## Native RDNA3 / gfx1100 seed

### `extra/gemm/amd_asm_matmul.py`

Status: first hardware family.

Properties:

- direct RDNA3 instruction construction through tinygrad's AMD DSL;
- explicitly targets gfx1100;
- 128x128 float32 GEMM tile;
- 128-thread workgroup;
- VOPD-aware accumulator and operand register placement;
- LDS staging and double-buffer-style global prefetch;
- reference check against tinygrad matmul already exists.

Initial tuning/synthesis dimensions:

- FMAC pair order and instruction scheduling;
- prefetch placement and distance;
- `s_clause` grouping;
- waitcnt placement;
- LDS swizzle and padding;
- occupancy limiter used by the harness;
- matrix-shape specialization.

Unsafe dimensions are not exposed as blind scalar knobs. Structural variants must regenerate a complete program and pass the reference oracle.

## Llama fused-kernel patterns

Most current files under `extra/llama_kernels` compile with `HIPCCCompiler("gfx950", ...)`. They are therefore architectural patterns, not accepted gfx1100 candidates.

### Fused RMSNorm / multiply / FP8 quantization

Source:

- `extra/llama_kernels/fused_rmsnorm_mul_quantize_fp8/`

Useful ideas:

- remove intermediate HBM materializations;
- one workgroup per row with grid-stride row processing;
- vectorized BF16 loads;
- fused normalization, weighting, quantization, saved backward state, and amax;
- compile-time workgroup and grid parameters.

Candidate structural variants:

- per-workgroup amax followed by a second reduction;
- scalar global atomic amax;
- workgroup/thread-count choices valid for hidden dimension;
- residual-add fusion;
- direct C/HIP versus UOp representation.

Required before use on W7900:

- successful gfx1100 compilation;
- confirmation that the relevant dtype/intrinsics are supported and performant;
- numerical validation against tinygrad;
- resource metadata and no-spill gate;
- end-to-end relevance to the selected inference model.

### Fused cross entropy

Provenance pattern: upstream tinygrad PR #16263 replaced handwritten C with a portable UOp version while preserving fusion and memory savings. The PR notes that the then-current BEAM-tuned UOp forward remained much slower than handwritten C, while end-to-end training differed far less.

Use in Forge:

- compare direct target-specific code and portable UOp families;
- optimize the end-to-end objective, not an isolated kernel ratio;
- retain the portable family as fallback.

### Atomic amax

Provenance pattern: upstream tinygrad PR #17063 replaces a two-stage amax reduction with atomics to remove repeated reduction-kernel launches.

Use in Forge:

- structural alternative, not assumed winner;
- compare under representative tensor size and contention;
- reject if numerical or end-to-end behavior regresses.

## Compiler-resource oracle

Provenance pattern: upstream tinygrad PR #3641 demonstrates extraction of COMGR executable metadata including VGPR/SGPR counts, LDS, scratch, occupancy, and spill counts.

Forge uses these as hard or diagnostic signals:

- spills may be forbidden by workload contract;
- excess VGPR/LDS use can reject a candidate before expensive trials;
- a kernel that loses occupancy is not automatically rejected, but the trade-off is measured and retained in evidence.

## Corpus policy

1. Record source and target architecture.
2. Distinguish a reusable idea from code verified on gfx1100.
3. Preserve rejected variants and reasons.
4. Never import a claimed benchmark without reproducing it on the assigned W7900.
5. Prefer end-to-end inference improvement over isolated-kernel speedup.
6. Keep tinygrad/UOp implementations as reference and fallback even when direct generated code wins.
