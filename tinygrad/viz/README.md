VIZ is a tool for inspecting tinygrad's rewrites and runtime profiling.

to use:
1. Run tinygrad with `VIZ=1` (this saves the pkls and launches the server in interactive shells)
2. That's it!

# VIZ in the command line

Use `python -m tinygrad.viz.cli` (add --json for scripting) to view the full timeline of events.

### Environment variables

Setting `DEBUG` includes the following data in the output stream. These show up as raw `{"value": "..."}` lines when using `--json`.

| DEBUG | Includes |
|------:|----------|
| 3 | Base AST |
| 4 | Generated source |
| 5 | Rewrite steps and kernel graph |
| 6 | All UOp graphs |
| 7 | All rewrites |


VIZ defaults to colored output. Set `NO_COLOR=1` to disable colors.

### Profiling examples

Get kernel times and ASTs

```bash
DEBUG=3 python -m tinygrad.viz.cli --json > /tmp/events.jsonl
```

Select events between two markers

Markers are set using the `profile_marker` helper in user code. To list them:
```
python -m tinygrad.viz.cli | rg MARKER
```
Then:
```
python -m tinygrad.viz.cli --interval "train @ 2" "train @ 3"
```

Set `-t` to aggregate events.

### Rewrites Debugging example

First, find the rewrite you are looking for. This can be a schedule or kernel:

```bash
python -m tinygrad.viz.cli -s TINY | rg Schedule
python -m tinygrad.viz.cli -s TINY | rg E_3
```

List all rewrite passes:

Rewrite pass names come from `graph_rewrite(..., name="...")` in user code.
```bash
python -m tinygrad.viz.cli -s TINY "Schedule 6 Kernels n1" --ls
```

Show the input graph for each pass

```bash
DEBUG=6 python -m tinygrad.viz.cli -s TINY "Schedule 6 Kernels n1"
```

Show all rewrites

```bash
# for the entire scheduler
DEBUG=7 python -m tinygrad.viz.cli -s TINY "Schedule 6 Kernels n1"

# or for a specific pass
DEBUG=7 python -m tinygrad.viz.cli -s TINY "Schedule 6 Kernels n1" "earliest rewrites"
```

# SQTT / PMC profiling (DEV=AMD only)

SQTT has additional overhead. Set VIZ=2 to include it in pkls.

Examples:

Get all SQTT packets
```bash
python -m tinygrad.viz.cli -s "kernel SQTT SE:0 PKTS" --json
```

Get bank conflicts:
```bash
python -m tinygrad.viz.cli -s "gemm PMC" | rg -A 16 SQC_LDS_BANK_CONFLICT
```

# PMC profiling (DEV=QCOM, Adreno A8xx only)

Per-kernel hardware counters are captured inline with each dispatch (begin/end
`CP_REG_TO_MEM` snapshots around `CP_EXEC_CS` in the same submission) and joined
with the GPU timestamp timeline in viz as `QCOM {kernel}` counter views plus an
`All Counters` aggregate. Each record carries raw counter deltas, derived metrics
(SP ALU utilization, stall fractions, MAD full/half share, ICL1 miss rate, VBIF
bytes), launch geometry, and IR3 resource metadata (register footprints, shared
memory, instruction length).

```bash
# on the A830 phone (needs KGSL perfcounter access, validated on stock SM8750)
LIBC_PATH=/system/lib64/libc.so DEV=QCOM PROFILE=1 QCOM_PMC=1 python3 my_kernel.py
python -m tinygrad.viz.cli -s "E_512_32_4 PMC"
```

`VIZ=2` enables `QCOM_PMC` automatically (same convention as AMD). Snapshot WFI
stalls are part of the measured kernel durations. True wave-level instruction
tracing (SQTT-class) is not exposed by the stock KGSL UAPI and is intentionally
out of scope.
