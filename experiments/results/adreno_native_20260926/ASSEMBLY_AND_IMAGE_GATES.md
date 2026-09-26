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
  layer size is page-aligned. The existing QCOM descriptor implementation is
  shared by native ADRENO and checked against those fields. This is a source
  audit plus A830 execution evidence, not a proof for other chips or layouts.

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

- Host: 11 native assembly/artifact tests passed with `-n12`; Ruff and mypy
  passed after image changes.
- A830 `IMAGE=2`: native image tests passed (FP32/FP16, mixed buffer input,
  interleaved images, coherent update, JIT rebinding, aligned view rejection,
  and asymmetric border). Native full device suite passed 21 tests and 13
  subtests after the border test.
- A830 `IMAGE=0` and `IMAGE=1`: full native device suite passed 17 tests and
  skipped four image-only tests in each mode.
- A830 image stability: `image-qualification-100x5000.json` records 100/100
  fresh processes with compiled native UAV image instructions, exact FP32
  output and unchanged inputs. One process completed 5,000 changed-input JIT
  replays with three distinct input addresses and exact outputs.
- Broad `IMAGE=2` backend ops gate remains **failed**: cross entropy requests
  an image view with only 2,048 accessible bytes where the linear image layout
  needs 4,096. The descriptor rejects it. Excluding that case exposes another
  undersized image view in cumprod (97,024 accessible, 98,304 needed). No
  fallback for these views has been qualified.
- Broad `IMAGE=0` backend ops gate remains **failed** on batch-local Conv1D
  values. The same seed-42 probe fails with the pre-image renderer, and the
  failing batches change between runs. This is an inherited, nondeterministic
  correctness issue rather than evidence that image support introduced it.
- Full model reference equality, 100 fresh *model captures*, and changed-input
  *model* replay are not qualified by the image tests above.

The reproducible process/replay probe is `experiments/adreno_image_qualification.py`.
It verifies that native UAV image instructions were compiled, checks outputs
and source bytes, and records each fresh-process result and changed-address
replay. Passing it is an additional gate; it does not clear the broad ops or
model correctness failures.

Do not treat targeted image execution as production qualification. The
small-view image rejection, Conv1D nondeterminism, and controlled Conv node 5
numerical difference remain open gates.
