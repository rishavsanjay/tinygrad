# Build a native Adreno assembly backend for tinygrad

Use this as the implementation prompt. The published branch containing this file is a **Mesa NIR/IR3 plus raw KGSL baseline**, not a native Adreno assembler. The older [Adreno 8xx specification](adreno8xx-backend-spec.md) describes that Mesa-based path and its qualification contract. Its compiler architecture is superseded by the requirements below; its memory, execution, correctness, and validation requirements remain useful.

## Objective and hard boundary

Implement a complete new `DEV=ADRENO` compute backend for Adreno 8xx, starting with the physical A830 Android/KGSL device. Tinygrad must lower its own UOps to Adreno machine instructions, allocate registers, schedule instructions, encode a shader binary, construct command streams, and submit them directly through KGSL. **Do not invoke Mesa NIR/IR3, OpenCL, Vulkan, a vendor shader compiler, `libtinymesa.so`, or an existing compiled shader blob to produce production kernels.** Mesa source/disassembly and Qualcomm OpenCL may be used only as independent research and correctness oracles. No CPU fallback may silently make an advertised Adreno operation pass.

This must be a genuinely separate backend with its own renderer/compiler and `ADRENO` device selection. Extract shared KGSL/HCQ transport code from `QCOM` only when that reduces duplication without routing `ADRENO` through `QCOM:IR3`. Keep the current QCOM path available as a control until the native backend meets its gates. Preserve the dirty checkouts and saved failure artifacts; work in an isolated checkout and make reviewable commits.

## Read first

1. Read this entire prompt, [the Mesa-based specification](adreno8xx-backend-spec.md), [implementation status](adreno8xx-implementation.md), and [input integrity findings](a8xx-input-integrity-findings.md). Make a requirements-to-source-and-tests checklist before editing.
2. Inspect the current fork branch, source ancestry, and local dirty state. The baseline for this handoff is the published `codex/adreno8xx-backend` branch, originally based on `ed1c3a4172fb62fe293f24e7c243bd3f8b422e70`. Verify live refs; do not assume branch names prove provenance.
3. Study the existing `tinygrad/runtime/ops_qcom.py`, `tinygrad/renderer/nir.py`, `tinygrad/runtime/support/compiler_mesa.py`, HCQ graph code, `tinygrad/device.py`, and their tests. They provide a KGSL transport and behavior oracle, not the new compiler.
4. Pin Mesa 26.2.1 source at `da14d65e4499e66468094be52bff9ea0915a695e` for A8xx device data, ISA encoding/disassembly, register fields, image descriptors, and production compute packet ordering. Record exact source paths and copied constants. Do not depend on that library at runtime or compile time.
5. Read `tinygrad/viz/README.md` before changing rewrite rules or claiming profiler evidence.

## Implementation sequence

### 1. Hardware and artifact contract

- Define the supported A830 chip ID, KGSL UAPI, instruction set variant, wave sizes, register files, shared/private memory, descriptor limits, and launch limits. Reject unknown chips before submission. Do not generalize one phone's pass to A810/A829/A840/X2.
- Create a typed, versioned, checksummed, pointer-free shader artifact with target chip, ISA/schema version, compiler options, ordered program signature, binary bytes, constant data, register/shared/private memory requirements, image resources, and dispatch requirements. Validate it before allocation or execution. Cache identity must include all codegen-affecting settings.
- Implement assembler and disassembler tests against independently sourced instruction examples. Verify field width, sign extension, relative branches, alignment, literals, register banks, and malformed binaries. Make encoding deterministic.

### 2. A working vertical slice

- Lower tinygrad UOps directly to a legal Adreno instruction IR. Implement instruction selection, control flow, predicates, constant/literal handling, vector lanes, register allocation and spilling, scheduling/hazard handling, and binary encoding. Start with FP32/int32 buffer loads, stores, arithmetic, comparisons, casts, barriers, and a single workgroup shape.
- Build direct KGSL allocation, argument packing, command emission, submit, exact submission timestamp retirement, host readback, and fault/timeout handling for `DEV=ADRENO`. Reuse reviewed generic HCQ code where appropriate, while keeping assembly compilation independent.
- Prove each step on the A830 with small deterministic kernels before expanding feature coverage. Compare to CPU/NumPy and Qualcomm OpenCL. Inspect disassembly and actual GPU memory values, not only successful submission.

### 3. Full tensor semantics and resources

- Cover the advertised tinygrad paths: FP16/FP32 storage and arithmetic with an explicit rounding/contraction policy; bool and 8/16/32-bit signed/unsigned integers; required 64-bit indexing; masked/noncontiguous loads and stores; reductions; matmul; convolution; normalization; softmax; attention compositions; and representative autograd. Capability-gate unsupported FP64/BF16/FP8, atomics, subgroups, or scopes rather than silently substituting another type.
- Preserve one ordered signature for arbitrary interleaved buffers, scalars, and images, including aliases and views. Validate argument offsets, widths, alignment, and runtime rebinding. Implement legal linear RGBA FP16/FP32 image reads/writes only after descriptor, pitch, backing allocation, alias, and cache transition tests pass. Keep a correct buffer path.
- Implement launch bounds, tail masking, scratch allocation/lifetime, shader preload and instruction cache handling, local memory and barriers, CPU/GPU cache visibility, bounded queue backpressure, ring wrap, LRU reuse, error teardown, and graph ownership. Use exact KGSL retirement for host waits; RAM timeline visibility alone is insufficient on this device.

### 4. Resolve the current correctness blocker

- The saved OpenPilot comparison failure was partly caused by **a contaminated eager reference**: one QCOM `IMAGE=1` run changed 1,655 bytes of a still-live future `big_img` input; feeding those exact altered bytes reproduced the old reference exactly. A clean reference made the same 20 changed-input JIT replays pass with the same shader manifest. This does **not** prove the underlying input overwrite is fixed. See the bundled checkpoint arrays and probes under `experiments/results/a8xx_input_integrity_20260924/`.
- Make reference generation validate each input's bytes and compare adjacent independently generated seeds. Never trust a reference just because it was produced eagerly. Keep expected host arrays outside mutable GPU storage. Capture the first corrupting kernel or allocation event, distinguish compiler, image descriptor, cache, and scratch hypotheses with interventions, then fix the demonstrated invariant. Retain the original failing arrays.
- Prove `ADRENO` eager, capture, first replay, changed-input replay, pruning, and 5,000-run long-lived graph behavior with stable input bytes and bounded metadata. A corrected harness alone is not a backend fix.

### 5. Qualification and delivery

- Run representative host tests with `python -m pytest ... -x -q -n12`, `python -m mypy tinygrad/`, and `python -m ruff check .`. Run phone GPU tests serially with fresh compiler and Python caches. Add focused regressions for each distinct failure mode; do not add a generated matrix that simply mirrors the implementation.
- Qualify A830 first. Test integer exactness, numerical tolerances for each floating-point mode, image/buffer parity, large shader and spill paths, symbolic shapes, allocator churn, faults/timeouts, graph rebinding, and a small training workload. For OpenPilot, require 100 fresh-process captures over at least five seeds, 20 changed-input replays per process, plus 5,000 long-lived replays. Include independently verified reference inputs and outputs. Never turn a small smoke pass into a full model qualification claim.
- Record exact source SHA, device/chip/kernel, compiler artifact format, KGSL/Mesa-oracle versions, model/input hashes, command/binary manifests, flags, elapsed times, and every failure/skip. Distinguish new A830 evidence from historical QCOM results. Keep optional performance tuning after correctness.
- Deliver an architectural explanation, capability table, tests, reproducible build/run instructions, evidence bundle, known gaps, and reviewable commits. Push to the user's fork on a **new** `codex/adreno-assembly-backend` branch after verification; do not force-push or create a PR unless requested.

## Definition of done

`DEV=ADRENO` can compile and execute the advertised tinygrad tensor workload on the A830 without any Mesa/vendor compiler path or precompiled production shader; raw KGSL submissions and their retirement are correct; the original model and input-integrity gates pass; and unsupported operations/devices fail clearly. Anything short of this should be reported as a staged implementation with precise remaining work, not as a complete Adreno assembly backend.
