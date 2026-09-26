# Native A830 backend implementation checklist

Base: `53d0811253920253585f00b3580d2ff6c8c0d44f`. Reference Mesa source:
`da14d65e4499e66468094be52bff9ea0915a695e` (26.2.1), inspected from a separate source archive.
Existing dirty workspaces and saved failure arrays are preserved.

| Requirement | Inspected source | Verification / checkpoint |
| --- | --- | --- |
| Independent UOp compiler and device selection | `codegen/__init__.py`, `renderer/__init__.py`, `device.py`, `renderer/nir.py` | Native rendering must work with no Mesa/compiler library installed; production kernels cannot call IR3/NIR |
| Instruction encoding, registers and branches | Mesa `isa/ir3-cat0.xml` through `ir3-cat6.xml`, `ir3/tests/disasm.c` | Independent known machine words, immediate boundaries, malformed instruction rejection |
| Scheduling and resource footprint | Mesa `ir3/ir3_legalize.c`, `ir3/ir3_compiler.c`, `ir3/ir3_shader.h` | A830 ALU/non-ALU delays, SS/SY completion, register allocation, loop liveness; private spills and long shaders now have a physical checkpoint |
| Chip and limits | Mesa `common/freedreno_devices.py`; physical phone reports chip `0x44050001` | Reject unsupported chip before submission; explicit wave, register, launch and shared-memory limits |
| Artifact and ordered ABI | `runtime/support/ir3.py`, `runtime/support/compiler_mesa.py`, `runtime/ops_qcom.py` | Native version/checksum/target/signature checks, interleaved and repeated argument bindings |
| Raw transport, allocation and retirement | `runtime/ops_qcom.py`, `runtime/support/hcq2.py`, `runtime/graph/hcq.py` | Share KGSL transport; native shader path emits raw commands without library discovery; exact timestamp retirement and graph rebinding |
| Numeric and tensor semantics | `codegen/decomp`, UOp pipeline and QCOM qualification tests | FP32/int32 buffer vertical slice first; then widths/casts, reductions, masked loads, matmul, conv, normalization, training |
| Local memory, images and spills | Mesa `isa/ir3-cat6.xml`, `fdl/fd6_view.cc`, QCOM image tests | Local memory/barriers and private spills physically tested; image descriptors remain a later checkpoint |
| Existing input overwrite | `a8xx-input-integrity-findings.md`, saved checkpoint/probe bundle | Preserve host inputs; locate first corruption with QCOM/OpenCL controls; clean references alone are not a fix |
| Full qualification | Complete handoff/specification and `tinygrad/viz/README.md` read | Host pytest -n12, mypy/Ruff; serial A830 tests; 100 fresh captures and 5,000 graph replays; record every gap |

The checklist is an implementation map, not a completion claim. Device evidence must identify the new source and native compiler.
