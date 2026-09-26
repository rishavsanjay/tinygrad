# A830 native assembly and image qualification, 2026-09-26

Target: physical Adreno 830v2, chip ID `0x44050001`. Native runs used
`DEV=ADRENO`, `MESA_PATH=/nonexistent`, and fresh compiler caches. The four
local Mesa source files cited below were checked byte-for-byte against Mesa's
official raw files at commit `da14d65e4499e66468094be52bff9ea0915a695e`:
`ir3-cat6.xml`, `ir3_a6xx.c`, `fd6_view.cc`, and `fd6_layout.c` all matched.

## Mesa-grounded image contract

- `src/freedreno/isa/ir3-cat6.xml`, `#instruction-cat6-a6xx-ibo-base`
  and `ldib.b` / `stib.b`: category 6, typed 2D RGBA, direct UAV ordinal,
  `110` in bits 20–22, and adjacent coordinate/data register components.
  The independent IR3 witness words are `0xc02200050361ba00` (load) and
  `0xc022000903677a00` (store); the native encoder matches both exactly.
- `src/freedreno/ir3/ir3_a6xx.c`, `emit_intrinsic_load_image` and
  `emit_intrinsic_store_image`: use a two-component coordinate, four image
  components, and image read/write barrier classes. Native image loads use
  X,Y register order. A physical 5×16 probe caught the reversed order and
  passed after correction.
- `src/freedreno/fdl/fd6_view.cc`, A8xx descriptor and storage descriptor
  branches: base address, type/depth, width/height, format/swizzle, bit pitch,
  and array slice offset. `src/freedreno/fdl/fd6_layout.c` requires linear
  pitch alignment and protects the final level against 16×4 overfetch; its
  layer size is page-aligned. Native image selection now requires the entire
  page-rounded layer span to fit in the argument's logical bytes. This is
  conservative for an owned, page-backed short allocation, but avoids treating
  an end-of-allocation view as if its following page were available. The
  existing QCOM descriptor implementation is shared by native ADRENO and
  checked against those fields. This is a source audit plus A830 execution
  evidence, not a proof for other chips or layouts.

## Controlled Conv node 5 comparison

The input and weight SHA-256 values match between native and QCOM in
`conv5-native-asm-v66.json` and `conv5-qcom-asm-v68.json`. Both kernels take
four FP16 buffer arguments. They use different launch geometries and therefore
their instruction streams are not line-aligned.

| Measure | Native ADRENO | QCOM IR3 |
| --- | ---: | ---: |
| Disassembly lines | 6,432 | 1,157 |
| Artifact bytes | 52,115 | 9,876 |
| `mul.f` | 144 | 144 |
| `add.f` | 160 | 160 |
| Private `ldp` / `stp` | 182 / 182 | 0 / 0 |
| FP32→FP16 / FP16→FP32 moves | 193 / 244 | 0 / 0 |
| Half load / store | 67 `ldg.f16` / 16 `stg.f16` | 50 `ldg.u16` / 4 `stg.u16` |

Native uses scalar FP32 registers and explicit half rounding, for example
`mul.f` followed by `mov.f32f16 (even)` then `mov.f16f32` around native
disassembly lines 2757–2769. QCOM uses `hr` operands for `mul.f`, converts to
half near output with `cov.f32f16`, and stores four half components per memory
instruction. Native also emits conservative six-cycle NOP spacing. These are
assembly differences, not proof of the numerical cause: reduction order and
half-register semantics differ, and both backends' standalone 131,072 sampled
FP16 products matched NumPy bitwise. The controlled QCOM output hashes are in
`conv5-qcom-control-v69.json`; native hashes are in `conv5-controlled-v65.json`.
Input hashes and post-execution bytes stayed stable in the controlled runs.
There is no evidence here that seed generation or input overwrite caused the
Conv disagreement.

## Qualification status

- Host: 29 targeted assembly/image tests passed, two skipped with `-n12`;
  Ruff and mypy passed after the span change.
- A830 `IMAGE=2`: native image tests passed (FP32/FP16, mixed buffer input,
  interleaved images, coherent update, JIT rebinding, aligned view rejection,
  asymmetric border, and masked native image loads). Native full device suite
  passed 22 tests and 13 subtests after the span change.
- A830 `IMAGE=0` and `IMAGE=1`: full native device suite passed 17 tests,
  skipped five image-only tests, and passed eight subtests in each mode after
  the span change.
- A830 image stability: `span-image-100x5000.json` records 100/100
  fresh processes with compiled native UAV image instructions, exact FP32
  output and unchanged inputs. One process completed 5,000 changed-input JIT
  replays with three distinct input addresses and exact outputs. The earlier
  `image-qualification-100x5000.json` used short page-backed images before the
  span change; the new record uses 16×16 RGBA FP32 images with 4,096 logical bytes.
- The previously rejected `IMAGE=2` cross entropy and cumprod ops now pass
  on A830. The broad `IMAGE=2` gate under Android Torch's default eight
  threads reported grouped Conv2D failures and later timed out. That grouped
  Conv result was a faulty reference, as shown below. A fresh-cache complete
  run of `test/backend/test_ops.py` with `OMP_NUM_THREADS=1`
  `OPENBLAS_NUM_THREADS=1`, `DEV=ADRENO`, `IMAGE=2`, and
  `MESA_PATH=/nonexistent` passed: **409 passed, 20 skipped, 109 subtests
  passed** in 541.35 seconds. The earlier 420-second command limit expired
  near the end of an otherwise failure-free single-threaded run.
- An earlier broad `IMAGE=0` gate reported batch-local Conv1D differences,
  also with changing failed batches across processes. With both thread limits,
  the `-k conv1d` selection passed four tests and 17 subtests. A fresh-cache
  complete `IMAGE=0` run then passed: **421 passed, 8 skipped, 126 subtests
  passed** in 529.52 seconds. The earlier Conv1D mismatch has not been
  isolated against an independent scalar reference.
- Full model reference equality, 100 fresh *model captures*, and changed-input
  *model* replay are not qualified by these backend tests.

## Android Torch reference check

`experiments/adreno_grouped_conv_reference.py` uses fixed NumPy seed 0 and
computes grouped Conv2D with explicit scalar products, independent of Torch
and either GPU compiler. On this phone with Torch 2.11.0, ten fresh CPU-only
processes at Torch's default eight threads gave three reference failures;
the bad values occupied complete 63-element batch/group blocks. Ten fresh
processes with `torch.set_num_threads(1)` gave zero failures. Setting only
`OPENBLAS_NUM_THREADS=1` still gave three failures in ten processes, while
setting only `OMP_NUM_THREADS=1` gave zero failures in ten and made Torch
report one thread. The records are `torch-grouped-default-10.json`,
`torch-grouped-one-thread-10.json`, `torch-grouped-openblas-one-10.json`, and
`torch-grouped-omp-one-10.json`. This identifies Android's threaded CPU
reference path; the effective control for this build is `OMP_NUM_THREADS=1`.

With Torch set to one thread, native ADRENO and Mesa QCOM each matched the
scalar reference in five fresh processes with unchanged input tensors;
`grouped-native-5.json` and `grouped-qcom-5.json` record this comparison.
Earlier default-thread `test_grouped_conv2d` failures therefore do not
establish a backend bug or a random-input seed bug. The isolated CPU-only
reproduction identifies the test reference as the source of those mismatches.

The reproducible process/replay probe is `experiments/adreno_image_qualification.py`.
It verifies that native UAV image instructions were compiled, checks outputs
and source bytes, and records each fresh-process result and changed-address
replay. Passing it does not clear model correctness gates.

Do not treat the backend ops suite as production model qualification. The
controlled Conv node 5 numerical difference and full-model capture/replay
remain open gates.
