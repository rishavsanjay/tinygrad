# Radeon Forge

Radeon Forge is an oracle-guided, fully local performance-engineering agent for private AI workloads on AMD Radeon / ROCm.

The durable project is the workload intent, invariants, correctness oracle, benchmark protocol, hardware evidence, and experiment history. Generated kernel implementations are disposable candidates: Forge may regenerate or replace them, but it may not deploy one that fails a hard oracle.

## Track 2 scenario

A user asks Forge to optimize a locally deployed private agent under explicit constraints, for example:

```text
Minimize P95 end-to-end task latency on a Radeon PRO W7900.
Do not use remote model APIs. Preserve task success within 1% of baseline,
require valid tool calls, forbid kernel spills, and stay below the VRAM limit.
```

Forge decomposes the request, invokes compiler/test/benchmark tools, remembers prior experiments, requests permission for consequential actions, and deploys or rolls back a measured configuration.

## Optimization loop

```text
contract
  -> baseline
  -> choose kernel/runtime family
  -> compile for gfx1100
  -> reject invalid resource usage
  -> correctness oracle
  -> short hardware trials
  -> successive halving
  -> held-out validation
  -> approval
  -> deploy or revert
```

The agent may propose structural rewrites. The deterministic tuner evaluates parameters within each implementation family. Neither may override a failed correctness, quality, stability, privacy, or resource gate.

## Metrics

Primary submission metric:

- P95 end-to-end latency on a frozen private-agent task suite.

Supporting metrics:

- time to first token;
- inter-token latency and decode throughput;
- kernel/subgraph median and P95 latency;
- task success and tool-call validity;
- numerical maximum absolute and relative error;
- VGPR, SGPR, LDS, scratch, spills, occupancy, and peak VRAM;
- crash rate and repeated-run stability;
- external network calls.

Speed is optimized subject to correctness and quality constraints. A faster invalid candidate cannot win.

## Tinygrad's role

Tinygrad is used as:

1. the trusted reference implementation and end-to-end workload;
2. the baseline and fallback runtime;
3. a corpus of AMD kernel implementations and optimization patterns;
4. the low-level AMD execution path for selected generated candidates.

Radeon Forge is kept under `extra/radeon_forge` instead of being embedded deeply into tinygrad's compiler. This makes the oracle and tuning loop independently auditable and lets generated HIP/UOp/AMD-DSL implementations remain replaceable.

## Current implementation

Implemented:

- workload, objective, correctness, and resource contracts;
- hard feasibility gates;
- append-only JSONL experiment ledger;
- successive-halving tuner;
- AMD metadata parsing for registers, LDS, scratch, occupancy, and spills;
- target-specific HIP compilation backend;
- executable JSON benchmark protocol;
- kernel-family search-space representation;
- seed family based on tinygrad's fused RMSNorm/multiply/FP8 implementation;
- unit tests proving metadata extraction and that invalid fast candidates cannot win.

Not yet claimed or implemented:

- no W7900 performance result has been recorded;
- the gfx950 Llama kernels are pattern references until individually compiled and validated on gfx1100;
- no generated candidate is approved for deployment;
- local planner-model serving and the interactive approval UI are still pending;
- end-to-end private-agent evaluation is still pending.

## Benchmark executable protocol

A workload harness receives:

- `RADEON_FORGE_CANDIDATE_JSON`
- `RADEON_FORGE_BUDGET`

Its final non-empty stdout line must be a JSON object such as:

```json
{
  "compile_ok": true,
  "stable": true,
  "samples_us": [8.4, 8.2, 8.3],
  "correctness": {
    "passed": true,
    "max_abs_error": 0.0002,
    "max_rel_error": 0.0008,
    "checked_values": 65536
  },
  "resources": {
    "vgprs": 72,
    "sgprs": 32,
    "lds_bytes": 8192,
    "scratch_bytes": 0,
    "spilled_vgprs": 0,
    "spilled_sgprs": 0
  },
  "evidence": {
    "target": "gfx1100",
    "reference": "tinygrad"
  }
}
```

## Development validation

```bash
python3 -m unittest test.test_radeon_forge -v
python3 -m compileall -q extra/radeon_forge
```

Hardware evidence must be generated on the assigned Radeon Cloud W7900 before any speed claim is made.
