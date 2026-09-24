# A830 OpenPilot input integrity investigation (2026-09-24)

## Verified result

- Device: Qualcomm Adreno 830 (`DEV=QCOM:IR3`, `IMAGE=1`), Mesa 26.2.1 IR3 compiler and direct KGSL submission.
- In an ordered seed 42 then seed 43 eager run, seed 43 `big_img` was exact immediately after allocation but 1,655 bytes changed after the seed 42 model execution. `img` remained exact. Changed bytes formed 64-byte stripes separated by 512 bytes in two regions of the 393,216-byte input allocation.
- The old seed 42 reference pair's seed 43 output differs from a clean independently generated seed 43 output at 2,569 of 2,576 elements (max absolute difference 57.7958984375). The old trial's alleged failed output equals the clean output byte for byte.
- Feeding the recorded 1,655 altered bytes to a fresh seed 43 run reproduced the old reference output exactly. Thus the old reference was generated from altered input data.
- All four old adjacent reference pairs disagree on their shared seed. A new independently generated seed 42/43 pair agrees exactly.
- The full 22-phase JIT sequence (two warmups plus 20 changed-input replays) passes exactly against the new clean seed 42/43 pair, with the same kernel manifest and source hashes as the old failing trial.
- Qualcomm vendor OpenCL (`DEV=CL`, `OPENCL_PATH=$HOME/libOpenCL.so`) and QCOM `IMAGE=0` both kept the image inputs intact in the matched probes. These are controls, not proof that either path is universally safe.

## Source of the underlying write

Unresolved. No declared IR3 output buffer overlapped the live seed 43 `big_img` allocation in the traced run. A scratch-guard trial stayed intact, but contemporaneous baseline runs also stayed intact, so that trial did not isolate scratch. CPU copy-in cache flushing did not distinguish the paths in runs without corruption. Later fresh runs, including fresh compiler and Python caches, produced clean inputs and references. Do not treat any cache or scratch change as a demonstrated fix.

## Local evidence

- `experiments/results/a8xx_input_integrity_20260924/stress/ref-42.npz`: old contaminated reference.
- `experiments/results/a8xx_input_integrity_20260924/stress/trial-0.failure.npy`: old alleged failed output; equals new clean seed 43 output.
- `experiments/results/a8xx_input_integrity_20260924/checkpoints/input-probe-ordered.npz`: expected and observed input bytes before and after seed 42 execution.
- `experiments/results/a8xx_input_integrity_20260924/checkpoints/ref-test-42.npz` and `ref-test-43.npz`: newly consistent adjacent reference pairs.
- `experiments/results/a8xx_input_integrity_20260924/checkpoints/clean-trial.json`: 22 exact checks; manifest matches old `trial-0.json`.
- `experiments/results/a8xx_input_integrity_20260924/probes/`: checkpoint, input, JIT, and corrupted-input replay programs.

No backend source, existing results, or phone deployment files were overwritten during this investigation. These files are investigation evidence; the input overwrite remains unresolved.
