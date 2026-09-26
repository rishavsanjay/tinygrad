# Native A830 backend: implementation checkpoint

This is an experimental implementation of `DEV=ADRENO`, starting from published baseline
`53d0811253920253585f00b3580d2ff6c8c0d44f`. It is not yet the complete qualification required by
[the assembly backend contract](adreno8xx-assembly-backend-prompt.md).
Native backend source commit: `2fe10fe60`. The final device run used compiler build
`tinygrad-adreno-a830-v1_37917ae6687b08f4f8e49b4af3ff0ea033e13e901f8cf3768786c6e626990259`.

## Execution path

`UOp -> renderer/adreno.py -> support/compiler_adreno.py -> ADN v1 artifact -> AdrenoProgram -> raw KGSL`.
The renderer selects instructions, allocates/reuses registers, spills into per-thread private memory,
resolves branches, and supplies the ordered argument/resource contract. The Python compiler encodes
instruction words directly. It never calls a shader compiler or uses a compiled kernel template.
The native device shares the existing QCOM allocation, HCQ queue, graph, command and retirement code.
`autogen/adreno.py` contains integer register constants and has no library loader. The original QCOM
renderer and Mesa device discovery are imported lazily only for QCOM. The pure Python `IR3Shader`
dataclass is reused as the execution resource record; it does not compile instructions.

Instruction layouts, independent machine-word witnesses, and chip properties were inspected at Mesa
26.2.1 `da14d65e4499e66468094be52bff9ea0915a695e`: `src/freedreno/isa/ir3-cat{0,1,2,3,4,6,7}.xml`,
`ir3/tests/disasm.c`, `ir3/ir3_legalize.c`, and `common/freedreno_devices.py`. Register constants are
regenerated from the bundled Mesa autogen source by `extra/qcom_gpu_driver/gen_adreno_constants.py`.
Reference notices are retained in `extra/qcom_gpu_driver/ADRENO_REFERENCE_LICENSE`.

## Current capability boundary

| Area | Current implementation and qualification |
| --- | --- |
| Device | Exact A830 chip `0x44050001`, Android KGSL; all other chips rejected |
| FP32/FP16 | Scalar global storage and arithmetic, explicit FP16 ties-to-even conversions; standard arithmetic property tests passed |
| Integers | Bool, 8/16/32-bit storage, exact full-width 32-bit multiply, signed/unsigned restoring division; device-tested |
| 64-bit | Existing tinygrad software tensor arithmetic, pair-valued address casts and shared software decomposition of index ALU; large signed pointer offsets and mixed-width comparisons physically tested |
| Control | Workgroup/local IDs, loops, predicates, tail/masked loads; device-tested |
| Registers | Linear liveness, register slices, private spills up to 4096 bytes/thread; long shader beyond preload cache device-tested |
| Local memory | Up to 32KiB, loads/stores, fence and barrier; exchange across two waves device-tested |
| Arguments | Native checksummed artifact, exact signature match, aligned buffer/scalar layout; mixed interleaved scalar and aliased buffer bindings physically tested |
| Graphs | Affine and full-model capture/replay with changed inputs passed; 5,000-run long-lived stress not yet qualified |
| Images | Not implemented; use `IMAGE=0`; image target initialization is rejected |
| Atomics/subgroups | Unadvertised, no native lowering |
| FP64/BF16/FP8 | Unadvertised |

Scheduling deliberately drains asynchronous work and inserts six dependency cycles after each
non-flow instruction. This is a correctness baseline, not a performance result. The native decoder
covers the instructions emitted here; it is not a general Adreno disassembler.

## New physical evidence, 2026-09-26

Phone: A830 `0x44050001`, kernel `6.6.118-android15-8-g2e6b9c3812c5-ab15114928-4k`, Termux Python 3.14.
A separate owned deployment and fresh Python/compiler caches were used; prior deployments and
failure arrays were preserved.

- Native device suite v62: **17 passed, 8 subtests passed**, 4.89 seconds. Includes affine JIT/input
  integrity, integer widths and division/remainder, half rounding/bitcast, reductions/matmul,
  convolution/normalization/attention, symbolic JIT, mixed and aliased arguments, large signed 64-bit
  offsets, private spills, register slices, and a 128-invocation shared-memory exchange across waves.
  It runs with `MESA_PATH=/nonexistent` and asserts no Mesa/NIR/IR3 compiler module import.
- Standard `test/backend/test_ops.py` v60 with gradients enabled: **363 passed, 66 skipped,
  57 subtests passed** in 298.67 seconds with `SKIP_SLOW_TEST=1` and a single CPU reference thread.
  Four standard dtype arithmetic property tests (FP16, FP32, int32, uint32; 200 Hypothesis examples
  each) passed. Eight additional arithmetic/cast tests produced **6 passed, 2 skipped**.
  Termux's multithreaded `torch.conv1d` gave varying wrong reference batches; the native and QCOM
  GPU results both matched host CPU PyTorch on saved identical inputs, and a one-thread phone reference
  made all 14 `test_conv1d` subtests pass. The backend tests import NIR as a test-harness module;
  the separate native device suite establishes the no-Mesa production path.
- Native full driving model eager seeds 42 and 43 completed with finite outputs and unchanged live
  input bytes. Model hash: `659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b`.
  Flags: `DEV=ADRENO IMAGE=0 FLOAT16=1 OPENPILOT_HACKS=1 NOLOCALS=0 JIT_BATCH_SIZE=0`.
  Final-source v62: one full-model capture and **20 changed-input JIT replays** matched their native eager outputs exactly;
  all inputs remained byte-for-byte unchanged. The captured manifest has 154 kernels and SHA-256
  `6cecdddd087889651f0413616436dc468086e927c6c5631d3d97eda7044c16d1`.
- Independent ONNX Runtime CPU outputs were generated on the host with the same model and input hashes.
  At `rtol=atol=0.002`, native differs in **971/2576** outputs for seed 42 (max absolute error 6)
  and **669/2576** for seed 43 (max 3). QCOM IR3 differs from ONNX Runtime in 1961/2576 and
  1940/2576, respectively. Native versus QCOM IR3 also fails this tolerance. Thus the model output
  correctness gate is **not passed**, despite exact native replay and input integrity.
- Node checkpoints show exact native/QCOM equality through ONNX node 4 (`Div`) and first divergence
  at node 5 (`Conv`): 617/524288 FP16 values exceed the same tolerance. A float32 CPU convolution
  rounded to FP16 is closer to native than QCOM at that node (1268 versus 1864 out-of-tolerance
  values). ONNX Runtime's actual node 5 output is also closer to native (2425 versus 3329
  out-of-tolerance values); its node 4 output has no values outside tolerance for either backend.
  The data support a numerical/rounding difference, but do not establish an acceptable
  full-model tolerance or rule out another downstream numerical issue.
- A matched host `DEV=CPU` tinygrad ONNX run with the same flags, model, inputs, and seeds also
  failed the ONNX Runtime output gate: **664/2576** outputs for seed 42 and **633/2576** for seed 43.
  Native versus tinygrad CPU still differed in **462/2576** and **572/2576** outputs, so there is
  both an inherited tinygrad/ONNX Runtime gap and a backend-specific gap. At node 4, tinygrad CPU
  matched ONNX Runtime bitwise; native differed only within tolerance. Re-running native node-5 Conv
  with the *exact* CPU node-4 tensor changed just **16/524288** outputs beyond tolerance versus
  native's original input, while native Conv versus tinygrad CPU Conv on that same input differed in
  **1418/524288**. A sampled **131072 FP16 products** matched NumPy bitwise. This localizes the
  larger node-5 gap to the Conv computation and is consistent with an accumulation or reduction
  order difference; the precise mechanism and acceptable model tolerance remain unproven. See
  `experiments/results/adreno_native_20260926/cpu-and-conv-controls.json` and
  `experiments/adreno_conv5_control.py`.
- Two Qualcomm OpenCL reference attempts failed at platform discovery with status -1001. QCOM IR3
  and ONNX Runtime CPU were usable independent references.
- Final-source host focused regression: **69 passed, 4 skipped** with pytest `-n12`; mypy passed
  all 223 source files; Ruff and `git diff --check` passed. The skips represent unavailable host
  hardware/compiler paths. The full model loads an ONNX CPU-side NIR module but no Mesa compiler;
  the native GPU device suite independently checks that no Mesa/NIR compiler modules load.

Implementation failures found and corrected by these runs: wrong workgroup-ID register file,
FP16 conversion rounding, casted register indices and slices, shared-memory lowering, address casts,
integer remainder lowering, mixed 32/64-bit comparisons and casts, an FP16 scratch-register collision,
and late sine expansion after wide dtype decomposition. Large-angle sine/cosine now pass the standard
backend tests. The historical QCOM `IMAGE=1` future-input overwrite remains unresolved; passing
native `IMAGE=0` input checks does not identify its source or establish a fix.

## Reproduce

```sh
DEV=ADRENO IMAGE=0 MESA_PATH=/nonexistent python -m pytest test/device/test_adreno.py -x -q
python -m pytest test/unit/test_adreno_asm.py test/unit/test_ir3_artifact.py test/unit/test_qcom_pmc.py test/device/test_qcom.py test/device/test_qcom_images.py -x -q -n12
python -m mypy tinygrad/
python -m ruff check .
```

On Android also set `LIBC_PATH=/system/lib64/libc.so`, `PYTHONPATH=.`, and separate writable
`PYTHONPYCACHEPREFIX`/`CACHEDB` paths for each fresh qualification process. Use
`experiments/adreno_model_qualification.py --help` for host-owned model input snapshots, reference
provenance checks, eager comparisons and 20 changed-input pruned graph replays. Use
`experiments/adreno_ort_reference.py` to regenerate the ONNX Runtime CPU reference. The
`--replay-despite-reference-mismatch` option collects replay evidence while retaining a failed
independent-reference verdict. Evidence JSON and selected arrays are saved under
`experiments/results/adreno_native_20260926/`. Raw model arrays remain in the owned local/phone
evidence directories and are ignored by Git; the pushed reports contain hashes and metrics.

Remaining required work includes native linear RGBA images, complete relevant index/scalar ABI
coverage, independent model comparisons, allocator/fault/teardown stress, the original corruption
investigation, 100 fresh captures over five seeds, and 5000 long-lived model graph replays.
