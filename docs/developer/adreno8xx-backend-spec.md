# Adreno 8xx backend specification

Status: proposed implementation and acceptance contract, 2026-09-22.

This document specifies the historical **Mesa NIR/IR3** path. For the separately requested native Adreno instruction compiler with direct KGSL submission, use [the assembly backend prompt](adreno8xx-assembly-backend-prompt.md). The runtime and qualification requirements here remain relevant, but the compiler architecture below is not the native assembly target.

The target is a production-quality tinygrad compute backend using Mesa NIR/IR3 and direct KGSL submission on Android. Extend the existing `QCOM` backend rather than creating a parallel runtime. Mesa supplies compiler semantics, device properties, register definitions, and the reference for command and image programming. Tinygrad owns tensor scheduling, kernel arguments, allocation lifetimes, submission, and graph replay.

“Full support” means every advertised operation, dtype, resource mode, and execution mode works on each qualified device, including fresh processes and changed-input replay. It does not mean exposing every graphics feature, silently running unsupported operations on CPU, or treating one A830 result as validation of all A8xx chips. Family-wide qualification remains incomplete until the device matrix below is satisfied.

## 1. Recovered implementation and evidence

The following state was inspected locally on 2026-09-22. Paths are evidence locations, not instructions to overwrite or merge their contents wholesale.

| Worktree | Observed branch / HEAD | Role |
| --- | --- | --- |
| `/home/rishav/Projects/tinygrad-a8xx-max` | `agent-a830-continue-20260906`, `ed1c3a4172fb62fe293f24e7c243bd3f8b422e70` | Recovered openpilot optimization stack; Mesa 26.2.1 compiler path, images, retirement handling, profiling, and scheduling experiments. Untracked probes/results must be preserved. |
| `/home/rishav/Projects/tinygrad-a8xx-image-comprehensive` | `codex/a8xx-image-comprehensive`, `f1291136bc53d039dcdee4c8d189eb030cb45a52` | Focused image implementation and semantic tests. |
| `/home/rishav/Projects/tinygrad-a8xx-compute` | `a8xx-buffer-compute-cleanup`, `94d1b1224ede5e8240f6220ca7cabe08a7d39ae1` | Earlier compute work; not the latest openpilot investigation. |
| `/home/rishav/Projects/tinygrad-a8xx-pr-review` | `buff-arg`, `3ac7be497ee352382844278a8000dbb81f573777` | Current documentation workspace; existing dirty runtime/test changes and investigation files. |

The broad experimental stack is not yet a fully qualified release. Its saved `A830_NEXT_PASS.md` and `experiments/results/a830_20260908_stress/` describe intermittent incorrect pruned capture under the faster local-workgroup policy. No-prune and capture-only synchronization passed limited historical stress runs, but neither establishes the root cause. Those results are historical evidence, not tests rerun for this specification.

The locally available `private/latest-a8xx-work` ref points to an older research lineage; a suggestive branch name is insufficient provenance. This audit did not fetch or certify remote branch tips.

### Mesa baseline

Use Mesa **26.2.1**, commit `da14d65e4499e66468094be52bff9ea0915a695e`, as the initial reproducible baseline. This tag and its sources were read from `/home/rishav/Projects/mesa`; that checkout's current HEAD is a different version. `/home/rishav/Projects/mesa-a830` is also a different development revision. Always select the explicit revision when comparing sources.

This is a baseline choice, not a claim that 26.2.1 is the newest or permanently preferred Mesa release. Upgrade compiler, generated bindings, metadata schema, and source-linked runtime assumptions together, after running the acceptance suite.

## 2. Supported scope and capability contract

### Platforms and devices

The first release target is unrooted Android AArch64 with accessible `/dev/kgsl-3d0`, a compatible KGSL UAPI, and the pinned tinymesa library. The primary qualification device is the available A830 phone. On 2026-09-22 SSH reported Android kernel `6.6.118-android15-8-g2e6b9c3812c5-ab15114928-4k`, model `CPH2723`, and sysfs GPU model `Adreno830v2`.

Mesa 26.2.1 contains multiple A8xx profiles, including A810, A829, A830, A840, and X2 parts. Recognition is separate from qualification. Resolve the actual chip ID with `fd_dev_info_raw` / `fd_dev_info`; use Mesa's matching rules and property inheritance. Do not infer hardware generation or register capacity from the marketing name.

Publish capabilities per `(chip ID, Mesa build, kernel/UAPI family)`, with states: qualified, recognized but experimental, or unsupported. Unknown chips must fail initialization with an actionable diagnostic before command submission. Existing A6xx behavior remains a regression target.

Linux MSM DRM support is a separate transport milestone for A8xx devices that do not use KGSL. Share compiler and hardware description code, but implement and qualify DRM allocation, synchronization, submission, and recovery independently. A KGSL-only release must be labeled as such; it cannot claim all-platform A8xx support.

### Tensor and numeric behavior

Required coverage includes elementwise arithmetic, comparisons, casts, indexing, gather/scatter as supported by tinygrad semantics, reductions, matrix multiplication, convolutions, normalization, softmax, attention compositions, views, copies, symbolic shapes, and autograd/training smoke tests. High-level operations should lower through the existing UOp pipeline.

| Feature | Required contract |
| --- | --- |
| FP16 / FP32 | Storage, arithmetic, casts, reductions, and mixed storage/accumulation; maintain the rounding and contraction policy promised by tinygrad. |
| Bool and 8/16/32-bit integers | Exact defined integer results, signedness, casts, shifts, and addressing behavior. |
| 64-bit integer operations / indexing | Explicit native or legal software lowering; test large offsets without requiring giant physical allocations. Reject unsupported operations during lowering. |
| FP64, BF16, FP8 | Currently excluded by the recovered IR3 renderer. Remain unadvertised until a separately tested lowering exists. No implicit precision substitution. |
| Atomics / subgroups | Capability-gated by operation, width, and memory scope. Implement only the semantics required by advertised lowering paths; disallow schedules needing unavailable primitives. |
| Images | Linear 2D RGBA FP16/FP32 tensor images, sampled read-only and coherent writable paths. Buffer execution remains the general correctness path. |
| JIT | Eager, capture, replay, changing input addresses/values, symbolic dimensions, and supported HCQ graph paths. |

Tiled images, UBWC, sparse residency, graphics interoperability, bindless resources, and indirect dispatch are extensions. Their absence does not block the defined tensor-compute release, but must be visible in the capability table. Do not enable their register paths without corresponding allocation and ordering contracts.

## 3. Architecture and code ownership

```text
Tensor / UOp / scheduler
  -> target-aware legal schedules and precision policy
  -> IR3Renderer: UOps -> NIR + ordered resource signature
  -> Mesa IR3: validated machine code + normalized shader metadata
  -> QCOMProgram: resource and launch validation
  -> QCOMArgsState: constants, buffers, images, descriptors
  -> QCOMComputeQueue / HCQ graph: command packets and dependencies
  -> KGSL transport: allocations, submit timestamps, retirement
```

Keep implementation in the existing renderer/compiler/runtime boundaries:

- `tinygrad/renderer/nir.py`: operation semantics, address spaces, parameter order, image access classification, precision controls.
- `tinygrad/runtime/support/compiler_mesa.py`: Mesa interface, artifact validation, compiler cache identity, normalized shader metadata.
- `tinygrad/runtime/ops_qcom.py`: device discovery, allocation, program setup, descriptors, command generation, signals and submission.
- `tinygrad/runtime/support/qcom_profile.py`: optional counter reservation and decoding.
- `tinygrad/engine/jit.py`, HCQ graph code, and allocator code: shared lifetime fixes where the failure belongs to generic execution rather than QCOM.
- Generated Mesa/KGSL bindings: reproducibly generated from pinned inputs; never hand-maintain duplicate register definitions.

Introduce small typed capability/metadata records where they eliminate repeated inference. Do not build an additional framework around every register. Keep policy (schedule selection, optional features) separate from mandatory hardware programming.

## 4. Mesa integration and shader artifact contract

Mesa's IR3 compiler owns instruction scheduling, register allocation, spills, hazard handling, and encoding. Tinygrad must not independently patch instruction scheduling to hide runtime ordering failures. Mesa's general [IR3 notes](https://docs.mesa3d.org/drivers/freedreno/ir3-notes.html) explain why scheduling is a compiler responsibility; their older-generation timing examples are not A8xx timing guarantees.

The recovered compiler serializes raw `ir3_shader_variant` and `ir3_const_state` bytes, followed by immediates and binary. This is ABI-sensitive and contains pointer-bearing C structures. Define a versioned, pointer-free artifact containing only consumed values:

- Magic/schema version, Mesa commit/build identifier, chip target and compile options.
- Binary size, instruction length, immediate bytes, constant allocation offsets and units.
- Full/half register footprints, branch stack, shared/private memory size and private-memory mode.
- Workgroup/local-ID locations, effective wave size, dispatch constraints, required fairness mode.
- Ordered argument signature, sampler/texture/UAV counts and `tex_to_image` mapping.

Validate bounds and lengths before allocation or submission. Never dereference pointers recovered from disk. If raw struct serialization is retained temporarily, scope it to an exact binding/library ABI, validate every section, and invalidate the cache on all ABI changes.

The compiler cache key must include artifact schema, actual Mesa build identity, chip ID, renderer/lowering version, and every codegen-affecting option, including precision and image policy. A hardcoded version string alone cannot detect a replaced `MESA_PATH` library. Confirm library compatibility before ctypes calls that depend on struct layout; package bindings and library as one tested unit.

Preamble enablement must match runtime support for its constants/resources. The recovered compiler disables preambles; enabling them requires a deliberate implementation and tests. Preserve compiler-declared resource requirements rather than guessing them from generated shader length.

## 5. Device properties and dispatch

Read capacity and errata from the pinned Mesa device database. At this baseline, A830 has six SP/CCU units, three slices, 32 KiB compute shared memory, wave allocation granularity two, and `fibers_per_sp=4096`. The base A8xx register-size property is 96 vec4, while the gen2 profile changes it to 128. These are source properties, not measured occupancy or portable constants for all A8xx devices.

The runtime must validate local dimensions/product, group counts, total invocation extents, shared memory, constants, descriptor counts, address alignment, and private-memory sizing before emitting packets. Revalidate resolved symbolic values on replay. Report the requested and supported limits in errors. Empty tensor operations must avoid invalid zero-sized hardware dispatches.

The current A830 runtime uses a 1024-thread product limit and 32-bit dispatch field bounds. Preserve these as audited limits until replaced by a device-derived rule; never equate a legal launch with an efficient launch. Tail workgroups and non-divisible extents must honor masks and last-local-size programming without out-of-bounds accesses.

Match the production Mesa compute setup using `fd6_compute.cc` and `tu_shader.cc`, with `computerator/a6xx.cc` as a minimal reference. Resolve differences explicitly:

- Devices without double threadsize use the effective wave size from `SP_PS_WAVE_CNTL`; Mesa production code uses `THREAD128` in the relevant WGE/WIE fields. The recovered computerator-derived programming must be reconciled with that property-driven rule, not changed solely because a constant differs.
- Carry compiler dispatch requirements, including linear/tiled invocation ordering, through metadata. Validate local-ID calculations and shared-memory algorithms under each enabled mode.
- Propagate occupancy-bounded workgroup fairness to round-robin configuration. An unconditional `QCOM_COMPUTE_RR` experiment is not the semantic policy. Mesa's A8xx base sets `round_robin_errata=False`; do not import the affected A6/A7 occupancy clamp.
- Respect shader preload/cache limits while programming complete instruction length for long shaders. Test shaders exceeding the instruction cache.
- Derive constant-RAM mode and encoded lengths with explicit units and checked bounds.
- Size private memory using the compiler's mode plus Mesa's per-fiber/per-wave rules, alignment, SP count, and address units. Allocate/grow scratch before submission and retain old scratch until all users retire.

Do not copy graphics initialization tables into the compute backend wholesale. Mesa notes that much A8xx non-context/chicken-bit programming moved into the kernel; establish userspace ownership for each write.

## 6. Memory and resource contract

Track logical length, allocated length, CPU address, GPU address, allocation owner, cache mode actually granted, and outstanding uses. Validate accesses relative to the containing allocation and view offset, including integer overflow. A returned GPU address need not equal a CPU pointer.

For Android KGSL, prefer supported writeback plus IO-coherency for data. Inspect returned flags; if coherence was not granted, use a validated uncached path or explicit cache-maintenance path. Preserve the recovered uncached command/signal allocation behavior until a coherent alternative is proven. CPU writes, GPU writes, host reads, and descriptor updates each need a documented visibility boundary.

Imported host memory requires successful mapping or independently established shared-address support. The recovered `_gpu_map` treats `EFAULT` as an identity-address fallback; audit this before release. An ioctl failure alone is not evidence that an arbitrary host pointer is GPU-addressable. Reject an unproven import or copy into owned memory.

Neither allocation reuse nor command-ring wrap may overwrite storage referenced by pending GPU work. Fence reuse to the last relevant submission, including LRU reuse that does not invoke the underlying free. Keep program binaries, scratch, descriptors, arguments, and graph allocations alive through retirement. Provide bounded backpressure for sustained submission rather than unbounded allocation or arbitrary timing sleeps.

### Ordered arguments

Use a single ordered program signature from renderer to runtime. Support arbitrary buffer/scalar/image interleaving, scalar widths and padding, repeated buffer aliases, and views. Buffer-list order and descriptor-table order are distinct projections of that signature; neither may be reconstructed by sorting argument kinds.

Do not assume that the separate buffer-argument development lineage is already integrated. Audit the selected base and port its semantic fixes with mixed-resource execution tests. Validate CPU-packed values against compiler offsets, then test graph rebinding with different buffers and symbolic values.

### Images

Use Mesa FDL layout/view/sampler sources as the descriptor oracle. For the supported linear RGBA formats, retain the audited 16-pixel row alignment, padded row tail, 4 KiB array-stride encoding, A8xx bit-pitch field, and 64-byte base alignment, with format-specific checked calculations. Verify allocation backing covers the complete hardware layout, not only logical tensor bytes.

The recovered path limits sampled resources to 31 and total image/UAV entries to 32. Enforce its supported limits before launch; derive future expansions from field widths and compiler contracts. Bind a complete UAV table in image-parameter order and construct the sampled table from IR3's `tex_to_image` mapping. Reject out-of-range mappings.

Read-only images may use the sampled path; resources written in the same kernel require coherent access. Classification must account for aliases and views, not merely identical parameter UOps. Where compiler assumptions cannot represent runtime aliasing safely, use a conservative writable variant or reject that combination before execution. Cross-kernel write-to-sample and descriptor-rebinding transitions require the appropriate cache operations.

Unsupported image layouts, symbolic layouts that cannot be represented, and resource overflow must fall back to legal buffer lowering before compiling the kernel, or fail clearly for explicitly requested image-only execution. Never silently reinterpret a noncontiguous view as contiguous image storage. Masked loads and border behavior must preserve tinygrad's values, including nonzero masked alternatives.

## 7. Ordering, retirement, and fault behavior

Define and test three separate boundaries: shader memory visibility, device queue dependencies, and kernel-confirmed submission retirement. RAM signal visibility alone is insufficient evidence for safe host consumption or resource reuse on the observed A830 stack.

Retain the Gen8 `CP_EVENT_WRITE7` completion path and exact KGSL timestamp association. A host wait must establish the appropriate HCQ condition and retirement of its producer submission; it must not wait on whichever command happened to be most recently submitted. Track context identity, timestamp wrap, signal reset/reuse, and multiple submissions publishing the same value.

Register an in-flight submission record before the submit ioctl can expose RAM completion; attach the returned timestamp atomically. On submission failure, remove only that attempt's records and preserve other users. Honor one total timeout budget across RAM and kernel waits; `timeout=0` remains nonblocking. Bound and prune retirement metadata after successful completion.

For graphs, resolve symbolic signal addresses and values on every invocation and associate them with the correct owned physical signal. Submission failure and graph destruction must not leave stale mappings. Repeated replay must not grow bookkeeping indefinitely.

Maintain a reviewed hazard table for: CPU-write to GPU-read, shader-write to shader-read, shader-write to texture-read, descriptor-write to shader-use, shader-write to host-read, and any future GPU-write to CP-consumed indirect data. Each transition must name the required cache/ordering operations and the Mesa source supporting them. `WAIT_FOR_IDLE`, cache writeback, invalidation, and KGSL retirement are not interchangeable operations. A8xx indirect CP consumption requires a separate audit before adding indirect dispatch.

Move helper compilation and other potentially recursive runtime work before GPU submission where needed. Preserve evidence for the recovered warmup-hang fixes, then replace ad hoc queue throttling with explicit ownership/backpressure invariants where possible.

On timeout, fault, allocation failure, or context loss, return an actionable error with chip, context, submission identity, and kernel name. Do not report success, recycle uncertain resources, or replay automatically on a lost context. Guarantee safe teardown; transparent context recovery is an optional later feature.

## 8. Capture/replay correctness blocker

The saved openpilot investigation found wrong outputs under fast local scheduling plus pruning despite identical manifests. Disabling memory planning did not eliminate the failure; limited no-prune and capture-only synchronization experiments passed. This supports investigating ordering and lifetime, but does not prove LRU reuse, missing initialization, or a particular cache operation is the cause.

Before enabling the fast policy by default:

1. Preserve the original failing arrays, manifests, binaries, flags, model hash, and seed.
2. Reproduce in fresh processes with changed inputs and compare eager, capture, first replay, and later replay.
3. Instrument allocation identity/address, aliases, last reader/writer submission, retirement, and reuse at the pruned one-time-work boundary.
4. Locate the first incorrect intermediate; use lifetime retention, synchronization, and no-prune as separate interventions.
5. Implement the smallest proven invariant at the correct layer and add a compact regression reproducer independent of the full model when possible.
6. Re-run the original model reproducer with pruning and the intended production scheduling policy.

A conservative synchronization fallback may be shipped if its semantics and cost are documented and its acceptance tests pass. It must be labeled as a fallback, not a root-cause proof. Defaults must not depend on the user knowing diagnostic environment variables.

## 9. Performance and observability

Optimize only schedules that pass correctness under the same manifest. Use distinct policies for convolution, GEMM, GEMV/decode, reductions, and attention; prioritize coalesced access, reuse, register pressure, and useful workgroup occupancy. Preserve masking and shared-memory/barrier legality in tuning.

The historical openpilot baseline had many one-thread workgroups; enabling locals was a major performance lead but exposed the capture blocker. Treat `NOLOCALS`, image-dot lowering, register tiling, and layout prepacking as measured policies rather than family-wide truths. Charge prepacking time and memory separately and report both first-use and amortized benefit.

Do not enable `QCOM_F16_MAD` by default under strict semantics: the recovered rewrite moves FP16 multiplication into FP32 and removes an intermediate FP16 rounding. Any relaxed mode must be explicit, included in cache identity, and evaluated against its own stated error budget.

Expose kernel name, binary hash, launch dimensions, resource footprint, compiler options, and GPU timestamps through the existing profiling/VIZ path. Use `tinygrad/viz/README.md` for collection and rewrite inspection. Optional PMC uses kernel-managed reservations, verified KGSL group IDs, slice-aware decoding where needed, and reliable release on error. Missing counters must not disable normal compute.

Counter collection perturbs execution. Capture/replay profiling must associate each sample with the actual invocation and report dropped/overflowed samples. Counter-derived occupancy/utilization claims require a validated interpretation; raw counts and theoretical capacities are not measured occupancy.

Benchmark cold initialization/compile, first execution/capture, warm replay, host wall time, GPU kernel time, peak memory, and sustained thermal behavior separately. Record charging/screen state, thermal sensors, available clocks, software hashes, model/input hashes, flags, and failures. Use matched A/B/A/B order; never compare a manifest from one run to timings from another.

## 10. Acceptance gates

Tests below are proposed release requirements, not results of this documentation task. Use concise representative cases for each independent failure mode; preserve targeted regressions when a bug is found.

| Gate | Required evidence |
| --- | --- |
| Build / ABI | Reproducible library and binding generation; Android loader smoke; exact build identity; artifact round-trip and malformed/truncated artifact rejection. |
| Host semantics | Argument offsets/order/aliases; descriptor fields against Mesa-generated fixtures; limit boundaries; chip matching; packet units; signal failure/timeout/wrap; symbolic replay metadata. |
| Device operations | Advertised dtype/op coverage; masked tails; noncontiguous views; large-index arithmetic; reductions; local barriers; scratch/spills; shaders beyond instruction cache. |
| Memory | Fresh CPU-to-GPU inputs, immediate host readback, write-to-sample, image/buffer aliases, imported-buffer behavior, allocator churn, scratch growth and command-ring wrap. |
| Images | IMAGE=0/1/2 where supported; sampled/writable/mixed resources; interleaved scalars and buffers; descriptor limits; alignment and padded allocation boundaries; changed-input replay. |
| Execution | Eager, capture, replay, symbolic changes, graph rebinding, pruning, one-kernel graph, multiple graphs, bounded metadata and repeated device lifecycle. |
| Faults | Mocked submit/alloc failures and timeout paths; real device faults only through controlled diagnosis; no stale records or unsafe reuse. |
| Models | Openpilot regression plus representative convolution, attention, and a small training workload; real/reference inputs, independent expected outputs, all production modes. |
| Device coverage | Exact revision evidence per qualified chip/kernel/Mesa tuple; an A830 pass does not qualify A810/A829/A840/X2. |

For deterministic integer and copy tests require exact equality. For floating-point tests specify reference dtype, accumulation mode, absolute/relative tolerances and special-value behavior before running. Exact hash matching is useful for the preserved openpilot same-configuration regression; it is not a universal cross-compiler floating-point requirement. Reject unexpected NaNs or missing outputs even if aggregate similarity looks good.

Minimum proposed stress gate for the capture fix: 100 fresh-process captures across at least five deterministic seeds, with at least 20 changed-input replays per capture, plus 5,000 replays in a long-lived graph while measuring allocation and retirement-record growth. This is a release sampling threshold, not a mathematical proof. All failures remain in the report.

Host tests use `python -m pytest ... -x -q -n12`, plus `python -m mypy tinygrad/` and `python -m ruff check .`. Select relevant existing QCOM/image/HCQ/symbolic suites from the chosen implementation checkout; do not imply that missing test files exist on older worktrees. Route caches to an isolated directory. Run physical-phone GPU tests serially to prevent competing workloads from invalidating diagnosis; parallelize host-only tests.

No absolute model-latency promise is set before a fresh correct baseline exists. A performance change passes only with unchanged semantics and a repeatable improvement beyond run variability, with cold-start/memory costs disclosed and regressions in representative workloads assessed.

## 11. Implementation sequence and deliverables

1. **Establish a reproducible base.** Inventory committed and dirty changes across recovered worktrees, choose an integration base, carry only reviewed functional fixes, pin Mesa/library/bindings, and retain original research artifacts. Deliver a source manifest and capability table.
2. **Harden compiler and resource contracts.** Version shader metadata, unify ordered arguments, validate capacities/layouts/imports, and make device properties authoritative. Deliver host tests plus small A830 buffer/image execution checks.
3. **Close runtime correctness gaps.** Audit retirement, reuse, scratch and command storage, recursive helper compilation, and failure cleanup. Resolve or conservatively guard the pruned-capture blocker. Deliver the reduced regression and passing original reproducer.
4. **Qualify the tensor backend.** Run all advertised modes, symbolic/graph/training cases and stress gates with fresh caches. Publish exact revisions, failures/skips, and qualified device tuples.
5. **Tune from the qualified baseline.** Evaluate local geometry, image selection, compiler upgrades, GEMM/GEMV layouts and optional precision policies using matched model evidence. Deliver profiles and measured tradeoffs, with defaults justified per target.
6. **Extend family and platform coverage.** Obtain hardware evidence for additional Mesa profiles; add MSM DRM transport separately if Linux support is required. Update qualification without extrapolating A830 results.

Each stage must leave a reviewable, independently testable change. A backend release requires stages 1–4 on every device it calls qualified. Stages 5–6 broaden performance and deployment coverage; missing chips/platforms remain explicitly unqualified.

## 12. Source map

All Mesa links below are pinned to the audited commit. Prefer these implementation sources over old generic Freedreno documentation for A8xx details.

| Area | Mesa source |
| --- | --- |
| Device IDs, capacities and errata | [freedreno_devices.py](https://gitlab.freedesktop.org/mesa/mesa/-/blob/da14d65e4499e66468094be52bff9ea0915a695e/src/freedreno/common/freedreno_devices.py) |
| Production compute setup | [fd6_compute.cc](https://gitlab.freedesktop.org/mesa/mesa/-/blob/da14d65e4499e66468094be52bff9ea0915a695e/src/gallium/drivers/freedreno/a6xx/fd6_compute.cc), [tu_shader.cc](https://gitlab.freedesktop.org/mesa/mesa/-/blob/da14d65e4499e66468094be52bff9ea0915a695e/src/freedreno/vulkan/tu_shader.cc) |
| Minimal compute reference | [computerator/a6xx.cc](https://gitlab.freedesktop.org/mesa/mesa/-/blob/da14d65e4499e66468094be52bff9ea0915a695e/src/freedreno/computerator/a6xx.cc) |
| Image layout and descriptors | [fd6_layout.c](https://gitlab.freedesktop.org/mesa/mesa/-/blob/da14d65e4499e66468094be52bff9ea0915a695e/src/freedreno/fdl/fd6_layout.c), [fd6_view.cc](https://gitlab.freedesktop.org/mesa/mesa/-/blob/da14d65e4499e66468094be52bff9ea0915a695e/src/freedreno/fdl/fd6_view.cc), [tu_sampler.cc](https://gitlab.freedesktop.org/mesa/mesa/-/blob/da14d65e4499e66468094be52bff9ea0915a695e/src/freedreno/vulkan/tu_sampler.cc) |
| Compiler metadata and lowering | [ir3_shader.h](https://gitlab.freedesktop.org/mesa/mesa/-/blob/da14d65e4499e66468094be52bff9ea0915a695e/src/freedreno/ir3/ir3_shader.h), [ir3_shader.c](https://gitlab.freedesktop.org/mesa/mesa/-/blob/da14d65e4499e66468094be52bff9ea0915a695e/src/freedreno/ir3/ir3_shader.c) |
| Ordering reference | [fd6_barrier.cc](https://gitlab.freedesktop.org/mesa/mesa/-/blob/da14d65e4499e66468094be52bff9ea0915a695e/src/gallium/drivers/freedreno/a6xx/fd6_barrier.cc), [tu_cmd_buffer.cc](https://gitlab.freedesktop.org/mesa/mesa/-/blob/da14d65e4499e66468094be52bff9ea0915a695e/src/freedreno/vulkan/tu_cmd_buffer.cc) |

Recovered local investigation entrypoints are `/home/rishav/Projects/tinygrad-a8xx-max/A830_NEXT_PASS.md`, `research/trackA_mesa_compute_delta.md`, `research/trackB2_a830_launch_audit.md`, and `experiments/a830_capture_stress.py`. Treat their performance and validation claims as historical unless attached to a new run on the selected release revision.

## 13. Fresh validation performed for this specification

On 2026-09-22 the existing phone deployment at `/data/data/com.termux/files/home/a830_schedule_audit_20260907` passed a small FP32 buffer smoke test: eager execution, capture, and four changed-input TinyJit replays, each processing 257 elements. Device initialization reported chip `0x44050001`, generation 8, target `a800,chip_id=0x44050001`. No model benchmark or full qualification suite was run.

The run used `DEV=QCOM:IR3`, `IMAGE=0`, `LIBC_PATH=/system/lib64/libc.so`, and `MESA_PATH=/data/data/com.termux/files/home/a830_26_2_1_profile_20260901_v1/a830-26.2.1-phone-deploy/libtinymesa.so`. `CACHEDB`, `PYTHONPYCACHEPREFIX`, and `XDG_CACHE_HOME` pointed into a newly created isolated directory, `/data/data/com.termux/files/usr/tmp/a8xx-spec-20260922.ZBzhRs`. Existing source files and caches were not overwritten.

Smoke test body:

```python
from tinygrad import Tensor, TinyJit, Device
from tinygrad.dtype import dtypes

@TinyJit
def run(x): return (x * 3 + 2).realize()

for i in range(6):
  x = Tensor([float(j+i) for j in range(257)], device="QCOM", dtype=dtypes.float).realize()
  assert run(x).tolist() == [(j+i)*3+2 for j in range(257)]
Device["QCOM"].synchronize()
```

An initial version of this probe used an `arange(device=...)` keyword unsupported by that checkout and stopped with a Python `TypeError`; the corrected constructor-based probe above passed. That harness error is not a GPU failure.

The three phone Python files below match the recovered local `tinygrad-a8xx-max` files byte-for-byte. This establishes provenance for those components, not equivalence of every deployed file. The shared-library hash identifies the tested binary; its directory name alone does not prove its build recipe.

| Artifact | SHA-256 |
| --- | --- |
| `tinygrad/runtime/ops_qcom.py` | `e0bd3264016da5c91eaff2ee5b3702fb5fbc48580b767a8d93cddc4aa2291038` |
| `tinygrad/renderer/nir.py` | `5575ad53fb089675fe0f6a3e406311719da77f80fe249c4f62d76193aeb1fd30` |
| `tinygrad/runtime/support/compiler_mesa.py` | `2619486c1630bebaf74b1edacd9bd9e9194b98ef79218a5e3fd308e46c75d3b0` |
| Phone `libtinymesa.so` | `8b9eb0750287ddfb6df3fc57580bf6106fcbbbc68f501e67a02bb52f8ed29f7d` |

Document checks verified all 11 pinned Mesa source paths exist at the specified commit, balanced code fences, and absence of trailing whitespace. This task changes documentation only. Whole-workspace Ruff reported 17 findings in pre-existing untracked investigation scripts; mypy reported seven errors in the existing `ops_qcom.py` while checking 213 files. Those source files were left unchanged. These checks describe the current documentation workspace, not the recovered implementation's release status. Logs: `/tmp/a8xx-spec-ruff.log` and `/tmp/a8xx-spec-mypy.log`.
