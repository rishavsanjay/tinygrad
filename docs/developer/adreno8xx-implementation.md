# Adreno 8xx implementation and validation

Implementation branch: `codex/adreno8xx-backend` in `/home/rishav/Projects/tinygrad-adreno8xx-backend`.
Base: `ed1c3a4172fb62fe293f24e7c243bd3f8b422e70` from the recovered openpilot worktree. These changes came from an uncommitted checkout; this published snapshot was assembled in an isolated clone without resetting or modifying that checkout.

This is a hardened experimental Android KGSL backend, not a claim of completed family-wide qualification. The governing specification remains `/home/rishav/Projects/tinygrad-a8xx-pr-review/docs/developer/adreno8xx-backend-spec.md`.

## Changes

- IR3 artifacts now contain a versioned, checksummed, pointer-free metadata record, immediates, and machine code. Truncated, malformed, unsupported-version, and invalid resource records are rejected.
- Compiler cache identity hashes the actual loaded Mesa library and generated bindings. Runtime checks artifact target/build against the active compiler. Old raw-struct cache entries use a different namespace.
- The renderer records each non-image parameter's compact call slot, aligned byte offset, and width in actual NIR order. Runtime argument packing consumes that layout, including interleaved scalars and repeated buffer references.
- Shader write slots travel with the artifact. Sampled image aliases of writable arguments are rejected before submission and after symbolic rebinding. This is an explicit rejection, not automatic recompilation to a coherent variant.
- Gen8 launch limits are checked both eagerly and after symbolic resolution. Shared-memory limits come from Mesa device properties.
- Gen8 waits use a total deadline; zero timeout is nonblocking and deadlocked contexts fail explicitly. The existing exact-submission retirement bookkeeping remains in place.
- Gen8 imported pointers require a successful mapping. A6xx's existing fallback is preserved and remains outside this change's qualification.
- LRU reuse retires preceding device work before returning cached storage. This is conservative device synchronization, not a per-allocation dependency optimization.
- Command and argument rings wait for the last submitted KGSL timestamp before wrapping. They reject oversized allocations. The wait deliberately does not target a timeline value reserved for a command that has not yet been submitted.
- Failed CPU mappings release their newly allocated KGSL object before propagating the error.
- Programs retain their own scratch owner when the device grows its scratch pool, preventing captured command streams from retaining freed scratch addresses.
- Compiler-required round-robin mode reaches command generation independently of the existing experimental override.

The default precision policy is unchanged. In particular, the existing optional FP16-to-FP32 multiplication rewrite remains opt-in.

## Regression tests

Host tests cover artifact round-trip/corruption, argument order and widths, sampled aliases on replay, launch boundaries, symbolic revalidation, wait deadlines, scratch lifetime, LRU reuse and ring wrapping. Existing image heuristic tests now reflect the recorded `ed1c3a417` change: independent reduction/output vectorization and selected image slots `(1, 3)`. The aligned-subbuffer device fixture now allocates the current 4 KiB image layout plus its offset; runtime bounds checks were not weakened.

Phone tests cover integer widths including 64-bit values, matmul/softmax and gradients against NumPy, symbolic JIT, image resource execution, HCQ behavior, and 5,000 graph executions rotating between three distinct input buffers. The graph stress test explicitly checks graph capture.

## Reproduction environment

Phone deployment: `/data/data/com.termux/files/home/a8xx_backend_impl_20260922.cUKjkB`.
SSH: `ssh -p 8022 u0_a600@100.106.106.51`.
Device: A830, chip `0x44050001`, Android KGSL.

```sh
export PYTHONPATH=.
export LIBC_PATH=/system/lib64/libc.so
export MESA_PATH=/data/data/com.termux/files/home/a830_26_2_1_profile_20260901_v1/a830-26.2.1-phone-deploy/libtinymesa.so
export DEV=QCOM:IR3
export PYTHONPYCACHEPREFIX="$PWD/cache/py"
export CACHEDB="$PWD/cache/new-run.db"
IMAGE=2 python -m pytest test/device/test_qcom.py test/device/test_qcom_images.py test/device/test_qcom_image_exec.py test/device/test_hcq.py -q
IMAGE=0 python -m pytest test/device/test_qcom_qualification.py test/backend/test_symbolic_jit.py -q
```

Run phone tests serially. Use a new cache name to reproduce cold compilation. Host checks use `pytest -n12`, `python -m mypy tinygrad/`, and `python -m ruff check .`, with caches routed to `/tmp/a8xx-impl-cache`.

The tested phone library SHA-256 is `8b9eb0750287ddfb6df3fc57580bf6106fcbbbc68f501e67a02bb52f8ed29f7d`. The loaded binary was reused, not rebuilt during this implementation. Its hash establishes artifact identity, not proof that an arbitrary Mesa library matches the ctypes ABI. The package index available here did not provide `tinymesa==26.2.1`; qualification does not rely on claiming that installation works.

## Remaining specification work

| Requirement | Status |
| --- | --- |
| Pointer-free cached shaders, ordered arguments, launch/alias checks | Implemented; targeted host and A830 tests |
| Submission retirement and bounded waits | Existing retirement model retained; deadline/fault semantics hardened |
| LRU, ring and scratch lifetime | Conservative ownership fixes implemented; targeted regressions |
| Capture corruption root cause | Fresh-process changed-input stress FAILED on trial 8 (ninth process), phase 3, maximum absolute error 8.637065887451172; root cause unresolved |
| Cold/warm image and graph execution | A830 validation only; see recorded results |
| Reproducible Mesa build and pre-ctypes ABI handshake | Still required; build hashing is not an ABI handshake |
| Complete dtype/op/atomic/subgroup qualification | Representative coverage only; exhaustive advertised-operation qualification remains |
| Mesa production dispatch differences and full hazard table | Requires further audited reconciliation; no speculative packet changes |
| Automatic buffer fallback for runtime image aliases | Not implemented; unsafe aliases fail explicitly |
| Detailed imported-memory, allocation-failure, context-loss and long-shader/spill coverage | Incomplete |
| A810/A829/A840/X2 and A6xx regression hardware | Not available in this run; unqualified |
| Linux MSM DRM transport, UBWC and other extensions | Not implemented |
| Performance promotion | No new speed claim; synchronization costs not benchmarked |

Do not merge or label the entire specification complete based on these targeted results. Preserve the original failing openpilot artifacts and continue the remaining gates on an explicitly identified revision.

## September 24 validation continuation

Host checks after scratch ownership: 67 passed, 5 skipped; mypy passed all 219 source files; Ruff and diff whitespace checks passed. The A830 qualification suite after scratch ownership passed all 3 tests and 8 subtests, including explicit graph-capture detection and 5,000 replays.

The earlier 100-process model gate stopped at its first failure after 8 successful processes. Each successful process had 20 exact changed-input replays; the ninth failed at phase 3 on input 1 (seed pair 45/46), with finite but incorrect output and maximum absolute error 8.637065887451172. That deployment predates scratch ownership. Its saved failure is `fresh_capture/trial-8.failure.npy`, and its source snapshot is `~/a8xx-pre-scratch-source.tgz` on the phone. The post-scratch rerun failed identically. Disabling pruning and enabling image invalidation separately also failed identically. New eager execution produced exactly the same failing output. Finally, the unchanged baseline backend reproduced the same changed-input failure under matching settings, showing this is inherited rather than specific to the new runtime changes. The underlying cause is unresolved.

The five saved reference pairs were cross-checked: each pair's second input output equals the next pair's first input output exactly. The failed output differs from both independent seed-46 references at 2,569 of 2,576 elements. The 100-process acceptance gate remains failed, not partially qualified. Future reports include the actual imported source hashes, effective precision/JIT context, environment settings, and kernel manifests on failure as well as success. Earlier references lack complete configuration provenance and are retained as investigation evidence.

Final A830 `IMAGE=2` suite on the current runtime: **64 passed, 13 subtests passed**, including the changed-input 5,000-replay graph test. The initial combined run exposed a harness issue: forward materialization could discard the dependency graph before `.backward()`. The test now constructs gradients first and still checks both forward output and gradients against NumPy with the original tolerances.

Evidence is saved under `experiments/results/a8xx_backend_20260924/`: source hashes, runtime diff, host/phone logs, the original failed output and reference pairs, and the post-scratch/pruning/invalidation/unchanged-baseline controls.

## Later input-integrity investigation

A later, separate seed-42/43 run found that one `IMAGE=1` QCOM eager execution changed 1,655 bytes of a still-live future `big_img` input. Its seed-43 reference was therefore contaminated. Feeding those recorded bytes to a fresh seed-43 run reproduced the bad reference exactly. A clean reference made the same 20 changed-input JIT replays pass with an identical shader manifest and source hashes. The older seed-45/46 failure described above had independently matching reference pairs and is **not** retroactively explained by the seed-42/43 contamination. The underlying input overwrite also remains unresolved. Details, probes, and arrays are in [the input-integrity findings](a8xx-input-integrity-findings.md) and `experiments/results/a8xx_input_integrity_20260924/`.

The tested OpenPilot model was 59 MiB with SHA-256 `659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b`. The model binary and Mesa library are external validation inputs, not bundled source files in this branch.
