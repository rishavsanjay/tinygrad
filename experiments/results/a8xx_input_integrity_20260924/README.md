# A830 input-integrity evidence, 2026-09-24

See `docs/developer/a8xx-input-integrity-findings.md` for conclusions and limits. `checkpoints/` contains complete selected ONNX node arrays and reference/replay comparisons; `stress/` contains the earlier fresh-process run; `probes/` contains the exact temporary Python diagnostics used on the phone. `SHA256SUMS` records this bundle's contents.

These arrays use deterministic synthetic OpenPilot inputs. The model binary is not bundled. The tested 59 MiB `driving_supercombo.onnx` had SHA-256 `659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b`. The external Mesa 26.2.1 library had SHA-256 `8b9eb0750287ddfb6df3fc57580bf6106fcbbbc68f501e67a02bb52f8ed29f7d`. The model tests ran on the user's A830 Android/KGSL phone with `DEV=QCOM:IR3`; Qualcomm OpenCL controls used `DEV=CL` and the device's vendor `libOpenCL.so`.

The probes are captured as research artifacts, not a new production test suite. The true GPU write that altered the live input has not been located or fixed.
