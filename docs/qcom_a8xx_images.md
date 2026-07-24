# Qualcomm A8xx linear images

This document describes the QCOM image implementation in this branch. It is tied to Mesa/tinymesa **26.1.4** and to the generated declarations in `tinygrad/runtime/autogen/mesa.py`. The hardware results below are from an Adreno A830. Gen6 and Gen7 behavior was source-reviewed and unit-tested, but was not run on A6xx or A7xx hardware.

## 1. Previous QCOM image architecture

QCOM already shared tinygrad's late image conversion with CL. An eligible linear buffer was reinterpreted as an `HxWx4` FP16 or FP32 image. The NIR renderer lowered two-dimensional `LOAD` and `STORE` UOps to `nir_intrinsic_image_load` and `nir_intrinsic_image_store`. IR3 supplied the sampled-image mapping, and `QCOMArgsState` placed one 64-byte texture descriptor per argument in the kernel-argument allocation.

The runtime assumed the A6xx texture-memory-object and sampler layouts for every supported GPU. It also assumed tightly packed rows, the Gen6 register path, a fixed 1 KiB constant footprint, `THREAD64`, and the old private-memory sizing formula. Those assumptions were sufficient for the existing Gen6 path but are not the A8xx ABI.

## 2. Why A830 failed

The A830 KGSL chip ID is in the newer encoded format. The previous decoder treated it as Gen7. That selected A6xx/A7xx descriptors and register programming even though A8xx moves the address, dimensions, format, swizzles, pitch, and sampler fields.

There were four independent failure classes:

1. Gen8 was selected as Gen7.
2. A6xx descriptors and samplers were submitted to A8xx.
3. A8xx state registers, descriptor bases/counts, compiled constant footprint, wave metadata, and private-memory metadata were not fully programmed.
4. Image rows and the final four-row linear over-fetch region could exceed the contiguous buffer that the image aliases.

The Mesa Python declarations were already generated from Mesa 26.1.4, while the optional dependency still declared tinymesa 25.2.7.2. That could load a library whose C layouts did not match the generated ctypes structs. The dependency and generated-library error text now both name 26.1.4.

## 3. GPU-generation detection

`_decode_chip_id` keeps the legacy KGSL format direct: the top byte is the major generation and the next byte is the decimal minor. New-format IDs are passed to Mesa's `fd_dev_name(struct fd_dev_id *)`; the returned `aNNN` name determines both GPU ID and generation.

This avoids a local table of individual products. It correctly maps `0x44050001` to `a830`, Gen8, using the same 26.1.4 device database that IR3 uses. Removing the Mesa lookup would either restore model-specific tables or misclassify future new-format IDs.

`QCOMDevice` accepts generations 6, 7, and 8. It obtains `fd_dev_info` for IR3 resource sizing and exposes the generation in the renderer architecture string. The architecture contains `QCOM_IMAGE_PITCH_ALIGNMENT=16` on Gen8 and `64` on Gen6/7 only when images are enabled. The QCOM-specific key lets generic CL and Python image selection retain their existing interpretation.

## 4. Gen6/7 versus Gen8 descriptors and registers

All image descriptors remain 16 dwords. The two families encode those dwords differently.

| Dword | Gen6/Gen7 `A6XX_TEX_MEMOBJ` | Gen8 `A8XX_TEX_MEMOBJ` |
|---:|---|---|
| 0 | format, samples, tile mode, RGBA swizzle | aligned base address bits 6–31 |
| 1 | width bits 0–14, height bits 15–29 | base high bits 0–16, type bits 17–19, depth bits 20–31 |
| 2 | pitch alignment bits 0–3, byte pitch bits 7–28, type bits 29–31 | width bits 0–14, height bits 15–29 |
| 3 | array pitch and flags | format bits 0–7, swap bits 8–9, swizzles bits 10–21 |
| 4–5 | base address, with depth in dword 5 | tile/flag metadata; zero for the linear views here |
| 6 | LOD/filter metadata | row pitch in **bits** at 0–23, minimum line alignment at 24–27 |
| 7 | plane/LOD metadata | array-slice offset in 4 KiB units at 0–22 |

The Gen6/7 branch preserves the established descriptor encoding. The Gen8 branch follows Mesa 26.1.4 `a8xx_descriptors.xml` and `fd6_view.cc`. Keeping the branches explicit matters: trying to share dword construction would hide fields that merely have similar names but different units and locations.

Register selection is shared only where Mesa defines the same programming model. Gen6 uses `REG_A6XX_*`; Gen7 and Gen8 use the A7xx NDRANGE/program/UAV registers, while Gen8 selects `REG_A8XX_SP_UPDATE_CNTL`, the A8 marker, and `REG_A8XX_CP_ALWAYS_ON_COUNTER`. Gen8 update control has stage-state bits but no Gen7 `CS_UAV` bit. Texture/UAV descriptor bases and `SP_CS_TSIZE`/`SP_CS_USIZE` are programmed explicitly.

## 5. Sampled-image descriptor construction

`qcom_image_descriptor(gen, dtype, shape, addr, storage=False, row_pitch=None)` is the single descriptor entry point.

For Gen6/7 it emits the existing linear 2D layout: float format and XYZW swizzle in dword 0, dimensions in dword 1, pitch/type in dword 2, and address in dwords 4–5.

For Gen8 it emits:

- 64-byte-aligned base address split between dwords 0 and 1;
- `A6XX_TEX_2D` and depth 1 in dword 1;
- width/height in dword 2;
- `FMT6_16_16_16_16_FLOAT` or `FMT6_32_32_32_32_FLOAT` and A8xx `X/Y/Z/W` swizzles `3/4/5/6` in dword 3;
- linear pitch multiplied by eight in dword 6 because `TEX_LINE_OFFSET` is measured in bits;
- `pitchalign - 6` in dword 6, matching Mesa's `MIN_LINE_OFFSET` field;
- the four-row-padded array pitch in dword 7's 4 KiB units.

Concrete addresses are checked for the generation's descriptor alignment. Symbolic graph addresses are intentionally not converted to Python integers at descriptor-template creation; they are resolved when graph arguments are rebound.

## 6. Storage-image descriptor construction

IR3 reports sampled textures through `image_mapping` and the remaining UAVs as storage images. `QCOMProgram` keeps that ordering: sampled descriptors begin at byte 2048 and storage descriptors immediately follow them in separate 64-byte slots. `QCOMArgsState` uses `tex_to_image` to bind sampled arguments in compiler order, then binds storage arguments in UAV order.

Gen6/7 keeps its existing storage-format dword-0 form. Gen8 uses the same 2D memory-object descriptor for sampled and storage access, as Mesa's A8xx view construction does for these linear float formats. Removing the storage branch would break Gen6/7; inventing a separate A8xx layout would disagree with Mesa.

## 7. Sampler construction

`qcom_sampler_descriptor` builds nearest, unnormalized, clamp-to-border samplers.

| Field | Gen6/Gen7 | Gen8 |
|---|---:|---:|
| wrap S/T/R | dword 0 bits 5/8/11 | dword 0 bits 6/9/12 |
| unnormalized coordinates | dword 1 bit 5 | dword 1 bit 31 |
| zero border | border-color allocation plus clamp | dword 2 `FASTBORDERCOLOREN` |

The coordinate convention is integer `(x, y)`. `IR3Renderer.tovec` receives tinygrad's index order as `(idx_y, idx_x)` and constructs NIR coordinates as `(idx_x, idx_y, undef, undef)`. Hardware tests distinguish X from Y and all RGBA lanes, including zero-valued SAME-padding reads.

## 8. Pitch, alignment, padding, and allocation

`QCOMImageLayout` records height, width, byte row pitch, array pitch, and alignment. `qcom_image_layout` accepts only `HxWx4` FP16/FP32 and follows Mesa 26.1.4 `fdl6_layout_image` for explicit linear layouts:

- rows are aligned to 16 pixels;
- RGBA16F therefore uses 128-byte alignment;
- RGBA32F therefore uses 256-byte alignment;
- an explicit pitch must cover the logical row and meet that alignment;
- the final level is padded to a four-row height because hardware blits can over-fetch at 16x4 granularity.

The image aliases an existing contiguous tensor buffer; no hidden repack allocation is inserted. `image_valid_dims` therefore rejects candidates whose padded rows and four-row tail do not fit in the page-rounded KGSL allocation. It also rejects element counts that are not divisible by four. This is why odd channel counts either find a valid generic RGBA factorization or remain buffers.

Ordinary tensor allocations use KGSL writeback mappings. CPU-to-GPU and GPU-to-CPU copies call `IOCTL_KGSL_GPUMEM_SYNC_CACHE` on the exact sub-buffer range. CPU-written command, signal, and kernel-argument mappings are uncached. These rules are generation-independent and are required for both buffer and image correctness.

## 9. Texture/UAV register programming

`QCOMComputeQueue.exec` separates three concerns.

Shared behavior loads shader code, constants, samplers, sampled descriptors, storage descriptors, descriptor counts, launch sizes, and work-item registers in compiler order.

Gen6 retains its A6 marker, update bits, NDRANGE layout, register IDs, and A6 sampler/texture/UAV descriptor layout. Gen7 uses the A7 NDRANGE and wave-control family. Gen8 uses the A8 marker/update register and counter while sharing the A7+ NDRANGE register family defined by Mesa.

For buffer-only programs the constant load is exactly the previous 256 dwords and `CONSTLEN_256`; descriptor storage at byte 2048 is not loaded as constants. Image programs use IR3's compiled `constlen` and matching 128/192/256/512 constant-RAM mode. Removing this split would change `IMAGE=0` and can overrun the compiled footprint for image kernels.

IR3 metadata supplies thread size, instruction length, early-preamble/merged-register flags, branch-stack alignment, and private-memory-per-wave behavior. Private-memory allocation uses the device's `fibers_per_sp` and `num_sp_cores`. These are compiler/device facts, not A830 tuning knobs.

## 10. Compiler and NIR image mappings

The generated ctypes layer and loaded library are one ABI unit. Mesa 26.1.4 reports these native/ctypes sizes:

| Structure | Bytes |
|---|---:|
| `ir3_shader` | 1224 |
| `ir3_shader_variant` | 2080 |
| `ir3_const_state` | 1424 |
| `nir_shader_compiler_options` | 296 |
| `nir_builder` | 40 |
| `nir_def` | 24 |

`rzalloc` allocates exactly `ctypes.sizeof`; version guesses and padding are deliberately absent. `IR3Compiler` keeps shader and variant objects as pointers, attaches the variant and const state to the shader's ralloc tree, and frees the shader root after assembly. If values or undersized copies were passed instead, Mesa would retain pointers to temporary ctypes storage or write beyond the allocation.

NIR's constructors return pointers. `_def_ptr` and `_instr_ptr` recover the embedded live objects, `nir_instr` writes flexible sources and intrinsic constant indices into the live allocation, and `_BuilderPointer` ensures every builder call mutates the same by-value `nir_builder` returned by `nir_builder_init_simple_shader`. `nchannel` handles `nir_alu_instr.src`, a flexible array represented as zero length by ctypes. `nimm_set` finds the containing load-constant object because Mesa 26.1's `nir_def` no longer carries the old parent pointer.

`IR3Renderer` lowers image loads/stores to NIR 2D intrinsics with FP16/FP32 source/destination types. A `DEBUG=4` A830 run shows `@image_load` and `@image_store` in image mode. The identical `IMAGE=0` program shows only `@load_global` and `@store_global`.

## 11. Symbolic addresses and graph replay

`HCQGraph` now gives each kernel-argument sub-buffer its exact allocation size. QCOM descriptor filling clears and writes the whole per-program argument block, so a size-less offset view could expose the parent buffer's remaining capacity and make validation or clearing touch unrelated graph arguments.

Graph argument rebinding rewrites buffer addresses and all sampled/storage descriptors for the current inputs. The hardware regression changes image input allocations on each replay and checks both output values and HCQ graph capture. Scalar arguments, symbolic stores, multiple input/output buffers, and inter-kernel visibility have separate focused regressions.

Command PM4 and standalone kernargs are bump-allocated. `BumpAllocator.wrap_callback` waits at the wrap boundary before older live command bytes or kernargs can be overwritten. `_submit` retains a dynamically built KGSL command object through the ioctl call. Removing either lifetime rule makes failures dependent on allocation pressure and garbage-collection timing.

## 12. Scheduler/codegen image selection

Image selection remains in tinygrad's existing late coalescing mechanism. QCOM adds only layout eligibility:

1. IMAGE must be enabled.
2. The base type must be FP16 or FP32.
3. The element count must form RGBA texels.
4. A candidate width must satisfy the generation's row alignment.
5. The logical rows plus four-row tail must fit the page-rounded allocation.

No model, layer, convolution size, phone name, image-slot state, local-size policy, or upcast tuning is carried through the scheduler. `IMAGE=1` uses the normal automatic mechanism; `IMAGE=2` forces every eligible conversion through the same correctness checks.

## 13. Buffer fallback behavior

`IMAGE=0` does not add the QCOM image architecture key and uses the original buffer descriptor/constant path. In image modes, an unsupported type, shape, alignment, pitch, or allocation size simply yields no image candidate. The UOps remain global buffer loads/stores.

There is no row-padding copy, compatibility shim, or model-specific exception. This keeps the fallback cheap and makes unsupported odd channel/layout cases explicit.

## 14. Supported formats and limitations

Supported now:

- sampled and storage RGBA16F;
- sampled and storage RGBA32F;
- linear 2D, one mip level, depth 1;
- nearest unnormalized coordinates;
- zero border color;
- padded aligned pitch and odd logical heights/widths when the backing allocation is large enough.

Not supported by this path:

- integer, normalized, three-channel, or other packed image formats;
- normalized/linear/anisotropic sampling or nonzero border colors;
- tiled or UBWC images;
- image arrays, cube maps, 3D images, mip chains, or multisampling;
- automatic insertion of padding/repacking allocations;
- arbitrary misaligned external pointers.

A6xx/A7xx descriptors are preserved and unit-tested, but no claim of A6xx/A7xx hardware validation is made.

## 15. Testing strategy

Portable tests cover chip-ID generation selection, Mesa-optional skips, Gen6/7 and Gen8 descriptor dwords, samplers, formats, address alignment, pitch/alignment/tail calculations, odd dimensions, RGBA/XY interpretation, border behavior, scheduler eligibility, exact HCQ sub-buffer sizing, cache-range ioctls, and bump-arena wrap callbacks.

Hardware tests cover initialization, allocation through normal Tensor/Buffer use, copy-in/out, scalar and elementwise operations, reductions, GEMM/FP16 GEMM, convolutions, synchronization, TinyJit/HCQ replay, symbolic scalar and buffer addresses, changed image descriptor addresses, multiple kernels/buffers, terminal GPU-write visibility, and compilation caching.

The optional QCOM-versus-CL transformer timing script lives under `extra/qcom_gpu_driver/benchmark_qcom_vs_cl.py`; it is not automatically collected as a correctness test. Hardware-only test modules skip before opening KGSL on non-QCOM hosts. Compiler/device tests skip if tinymesa is absent.

## 16. A830 hardware validation

The tested device was an Adreno A830 (`a830`) on Android 16 through unrooted Termux. An isolated exact Mesa 26.1.4 Android library was built with `/home/rishav/Projects/build_tinymesa_android.sh`, copied beside the isolated test tree, and selected with `MESA_PATH`. The phone's installed tinymesa and existing tinygrad checkout were not modified.

Focused results:

- `IMAGE=0`: 20 QCOM device/coherency tests passed; a broader scalar/copy/reduction/GEMM/JIT/symbolic/HCQ matrix also passed 20 tests plus one subtest.
- `IMAGE=1`: 31 image/convolution/coherency tests passed with 14 convolution subtests; final graph-address regressions passed 11 tests.
- `IMAGE=2`: the same 31 tests and 14 subtests passed; final graph-address regressions passed 11 tests.
- screenshot smoke: buffer basic op, odd-height image round trip, SAME depthwise convolution, and TinyJit replay all passed.
- exact IR3 A830 compilation and NIR serialization succeeded.
- a second EDSR process using the same cache made ten cache lookups and zero compiler calls.

The official EDSR input produced SHA-256 `8971af481d5c30d00e8bbe1423ef7f452ed97a473eebd696b98284303bc717bc` in all three image modes. Maximum absolute error versus the pinned TFLite output was `0.0002899169921875`; mean absolute error was `2.1062049633689675e-05`. Every mode captured one graph submission per inference.

CPU and CL comparison matrices both passed on the phone. Directly opening `/vendor/lib64/libOpenCL.so` is denied by Android's Termux linker namespace; the established copied loader plus `OCL_ICD_FILENAMES` configuration was then used successfully. The official EDSR input also passed on CPU and CL, with maximum errors `0.000335693359375` and `0.0003662109375` respectively.

## 17. Performance results (separate from correctness)

These are short synchronized diagnostics, not an MLPerf submission and not tuning evidence. Each EDSR number below has two warmups and five timed iterations in a fresh process:

| Mode | Median latency | Correct | Graph submissions |
|---|---:|---:|---:|
| QCOM buffer, `IMAGE=0` | 931.207 ms | yes | 1 |
| QCOM automatic, `IMAGE=1` | 585.968 ms | yes | 1 |
| QCOM forced eligible images, `IMAGE=2` | 586.992 ms | yes | 1 |
| CL comparison | 403.824 ms | yes | 13 |
| CPU comparison (one timed iteration) | 1852.879 ms | yes | 1 |

The compact smoke measured 1.952 ms for its buffer expression and 1.831 ms for its image expression. Those tiny timings are presentation-only.

An isolated same-library A/B reintroduced only the removed scheduler image policy files; the same EDSR automatic workload slowed to a 2675.507 ms median while remaining correct. The cleanup therefore did not cause the apparent difference from an older thermally separate ~527 ms report. That older run used a different code/library state and is not a controlled regression baseline. No tuning was added here.

## Changed backend symbols and removal consequences

| Symbol/change | Role and hardware/Mesa correspondence | Scope | If removed |
|---|---|---|---|
| `_decode_chip_id` | Uses `fd_dev_name(fd_dev_id*)` for new KGSL IDs | shared generation detection | A830 is selected as Gen7 or needs a model table |
| `QCOMImageLayout`, `qcom_image_layout` | Implements `fdl6_layout_image` linear pitch and four-row tail | shared A6+ fact | descriptors can read beyond rows/allocation |
| `qcom_image_descriptor` | Encodes `A6XX_TEX_MEMOBJ` or `A8XX_TEX_MEMOBJ` | split Gen6/7 vs Gen8 | Gen8 address/format/pitch fields are invalid |
| `qcom_sampler_descriptor` | Encodes generation-specific sampler bit positions | split Gen6/7 vs Gen8 | coordinates/border behavior is wrong on Gen8 |
| `QCOMSignal.wait` | Couples timeline visibility to KGSL submission completion | generic QCOM | CPU can observe the signal before terminal GPU effects |
| `QCOMComputeQueue.timestamp` | Selects A8 always-on-counter register | Gen8-specific | timestamps read the Gen6/7 register |
| `QCOMComputeQueue._submit` local object | Retains dynamically built KGSL command storage through ioctl | generic QCOM | temporary PM4 metadata can die before submission |
| `QCOMComputeQueue.exec` generation branches | Programs marker/update/NDRANGE/wave/descriptor bases and counts | split plus shared loads | A830 launches or descriptors fail; fixed buffer constants would regress |
| `QCOMArgsState._tex` | Calls the one descriptor encoder in compiler argument order | shared | old A6 descriptor is emitted for all GPUs |
| `QCOMProgram` IR3 metadata | Carries constlen, wave, registers, pvtmem, image mapping | shared IR3/A8-required | launch resource state disagrees with compiled shader |
| `QCOMAllocator._sync_cache` and copies | Maps KGSL cache maintenance to exact buffer ranges | generic QCOM | host/device copies can be stale |
| `QCOMDevice._wait_for_command_arena` | Protects wrapped live PM4 by KGSL timestamp | generic QCOM | high allocation pressure can overwrite in-flight commands |
| `QCOMDevice` register/dev-info selection | Separates A6, A7, and A8 register families | generation split | Gen8 uses wrong registers/resource data |
| `QCOMDevice._gpu_alloc` cache mode | Declares writeback versus uncached mappings explicitly | generic QCOM | cache-sync policy is ambiguous/wrong |
| `image_valid_dims` QCOM filtering | Ensures aliasable RGBA rows and tail fit | QCOM selection | automatic images can overrun backing buffers |
| `_def_ptr`, `_instr_ptr`, `nir_instr`, `nchannel`, `nimm_set`, `_BuilderPointer` | Preserve Mesa 26.1 pointer/flexible-array ABI | shared NIR | ctypes temporaries or wrong offsets corrupt NIR |
| `NIRRenderer.prerender` options pointer | Passes the exact live IR3 option block | shared NIR/IR3 | copied/truncated compiler options can mismatch ABI |
| `rzalloc`, `IR3Compiler.compile` | Allocates exact 26.1 structs and one ralloc tree | shared IR3 | under-allocation, dangling pointers, or leaks |
| `disas_adreno` duplicated fd | Lets `fclose` flush without closing `TemporaryFile`'s fd | shared IR3 tooling | disassembly lifetime can double-close the descriptor |
| `HCQGraph` sized args offset | Bounds each graph kernarg/descriptor block | generic HCQ, needed by QCOM | clearing/writing can spill into adjacent arguments |
| `BumpAllocator.wrap_callback` | Adds a small lifetime hook at arena wrap | generic mechanism used by QCOM | in-flight command/kernarg storage can be reused |
| Mesa autogen command and `pyproject.toml` pin | Both select Mesa/tinymesa 26.1.4 | ABI declaration | generated ctypes and loaded package can disagree |

The test modules, moved benchmark, and smoke utility contain no backend behavior. They validate or present the symbols above.
