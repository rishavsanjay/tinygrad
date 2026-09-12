# Adreno gen8 performance counters via the kernel-managed KGSL perfcounter ioctls.
#
# gen8 removes userspace perfcounter selector programming (Mesa c2708afbc7e "tu: only
# support userspace-managed perfcounters on a7xx and earlier"): the kernel owns the SEL
# registers. Counters are reserved with IOCTL_KGSL_PERFCOUNTER_GET, which returns the
# LO/HI counter register offsets (verified against Mesa's A8xx RBBM_PERFCTR_* tables:
# e.g. SP offset 0x292 == REG_A8XX_RBBM_PERFCTR_SP(18)). IOCTL_KGSL_PERFCOUNTER_READ is
# EPERM for unprivileged contexts on stock kernels, so counters are read from command
# streams with CP_REG_TO_MEM exactly like freedreno's compute queries: WFI for register
# visibility and free-running 64-bit values differenced host-side.
from __future__ import annotations
import weakref, contextlib, time
from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from tinygrad.runtime.ops_qcom import QCOMComputeQueue, QCOMDevice
from tinygrad.helpers import getenv, ProfileEvent
from tinygrad.runtime.autogen import kgsl, mesa

# Fixed rate of CP_ALWAYS_ON_COUNTER (measured 19.25-19.73 MHz on-device; nominal XO).
AO_HZ = 19_200_000.0

# High-value counter set from the Mesa 26.2.1 a8xx countable tables; group ids are the
# KGSL_PERFCOUNTER_GROUP_* values. Countable ids are hardware selector semantics shared
# by the kernel tables, so validate them with discriminating workloads before trusting
# any single counter for optimization decisions.
SP, UCHE, CP = kgsl.KGSL_PERFCOUNTER_GROUP_SP, kgsl.KGSL_PERFCOUNTER_GROUP_UCHE, kgsl.KGSL_PERFCOUNTER_GROUP_CP
RBBM, TP = kgsl.KGSL_PERFCOUNTER_GROUP_RBBM, kgsl.KGSL_PERFCOUNTER_GROUP_TP
# IOCTL_KGSL_PERFCOUNTER_GET programs the Gen8 selector in the kernel. Qualcomm's Gen8
# perfcounter implementation also writes reg_dependency[] selectors (including slice
# selector registers where a group has them) and restores RESTORE groups across
# preemption/IFPC. Do not duplicate Turnip's userspace selector programming here.
# Core set: always reserved. Viz rows and the host unit test depend on these names.
COUNTERS_CORE:list[tuple[str, int, int]] = [
  ("SP_BUSY_CYCLES", SP, 1), ("SP_ALU_WORKING_CYCLES", SP, 2), ("SP_STALL_CYCLES_TP", SP, 4),
  ("SP_STALL_CYCLES_UCHE", SP, 5), ("SP_CS_INVOCATIONS", SP, 100), ("SP_FULL_ALU_MAD_INSTRUCTIONS", SP, 109),
  ("SP_HALF_ALU_MAD_INSTRUCTIONS", SP, 110), ("SP_FULL_ALU_MUL_INSTRUCTIONS", SP, 111),
  ("SP_HALF_ALU_MUL_INSTRUCTIONS", SP, 112), ("SP_FULL_ALU_ADD_INSTRUCTIONS", SP, 113),
  ("SP_HALF_ALU_ADD_INSTRUCTIONS", SP, 114), ("SP_ICL1_REQUESTS", SP, 51), ("SP_ICL1_MISSES", SP, 52),
  ("SP_LM_BANK_CONFLICTS", SP, 61), ("UCHE_READ_REQUESTS_SP", UCHE, 11), ("UCHE_WRITE_REQUESTS_SP", UCHE, 21),
  ("UCHE_VBIF_READ_BEATS_CH0", UCHE, 32), ("UCHE_VBIF_READ_BEATS_CH1", UCHE, 33),
  ("UCHE_VBIF_WRITE_BEATS_CH0", UCHE, 34), ("UCHE_VBIF_WRITE_BEATS_CH1", UCHE, 35), ("CP_ALWAYS_COUNT", CP, 1),
]
# Opt-in groups via QCOM_PMC_EXTRA (comma-separated) or QCOM_PMC_EXTRA=all. Each answers a
# live analysis question; keep the core set lean since KGSL reservation slots are limited.
COUNTERS_GROUPS:dict[str, list[tuple[str, int, int]]] = {
  # complete the SP stall breakdown (core has TP+UCHE only)
  "stall": [("SP_STALL_CYCLES_VPC_BE", SP, 3), ("SP_STALL_CYCLES_RB", SP, 6)],
  # occupancy / register pressure: wave starvation, GPR conflicts, GM load latency
  "occ": [("SP_LOW_EFFICIENCY_STARVED_BY_TP", SP, 68), ("SP_STARVE_CYCLES_HLSQ", SP, 69),
          ("SP_GPR_READ_CONFLICT", SP, 80), ("SP_GPR_WRITE_CONFLICT", SP, 81),
          ("SP_GM_LOAD_LATENCY_CYCLES", SP, 82), ("SP_GM_LOAD_LATENCY_SAMPLES", SP, 83)],
  # decompose the UCHE stall: arbiter pressure and VBIF (sysmem) latency
  "mem": [("UCHE_BUSY_CYCLES", UCHE, 1), ("UCHE_STALL_CYCLES_ARBITER", UCHE, 2),
          ("UCHE_VBIF_LATENCY_CYCLES", UCHE, 7), ("UCHE_VBIF_LATENCY_SAMPLES", UCHE, 8)],
  # texture pipe: the IMAGE path runs through TP and the core set measures none of it
  "tp": [("TP_BUSY_CYCLES", TP, 1), ("TP_STALL_CYCLES_UCHE", TP, 2), ("TP_LATENCY_CYCLES", TP, 3),
         ("TP_L1_CACHELINE_MISSES", TP, 8), ("TP_STARVE_CYCLES_SP", TP, 55), ("TP_STARVE_CYCLES_UCHE", TP, 56)],
  # system: GPU/CP busy fractions and CP preemption (frame-spike attribution)
  "sys": [("RBBM_STATUS_MASKED", RBBM, 3), ("CP_BUSY_CYCLES", CP, 3),
          ("CP_NUM_PREEMPTIONS", CP, 4), ("CP_PREEMPTION_REACTION_DELAY", CP, 5)],
}
def resolve_counters(extra:str) -> list[tuple[str, int, int]]:
  if extra.strip().lower() == "all": return COUNTERS_CORE + [c for g in COUNTERS_GROUPS.values() for c in g]
  want = [w.strip().lower() for w in extra.split(",") if w.strip()]
  if (bad:=[w for w in want if w not in COUNTERS_GROUPS]):
    raise ValueError(f"unknown QCOM_PMC_EXTRA groups: {bad} (pick from {sorted(COUNTERS_GROUPS)} or all)")
  return COUNTERS_CORE + [c for w in want for c in COUNTERS_GROUPS[w]]
COUNTERS = resolve_counters(getenv("QCOM_PMC_EXTRA", ""))
KGSL_SYSFS = "/sys/class/kgsl/kgsl-3d0"

@dataclass(frozen=True)
class QCOMPMCSample:  # noqa: E702
  # One u64 delta per entry. All geometry fields are 1 so tinygrad.viz.serve.unpack_pmc
  # consumes exactly one blob value per sched entry with no AMD-specific decoding.
  name:str; xcc:int=1; inst:int=1; se:int=1; sa:int=1; wgp:int=1; off:int=0  # noqa: E702

@dataclass(frozen=True)
class QCOMProfilePMCEvent(ProfileEvent):  # noqa: E702
  # Matches ProfilePMCEvent's core fields so viz treats it as a per-kernel PMC record,
  # plus an optional JSON-safe info dict (launch geometry, IR3 resource metadata).
  device:str; kern:int; sched:list[QCOMPMCSample]; blob:bytes; exec_tag:int; info:dict|None=None  # noqa: E702

class QCOMPerfCounters:
  """Reserve kernel-managed gen8 counters and encode command-stream snapshots."""
  def __init__(self, dev:QCOMDevice):
    self.dev = weakref.ref(dev)
    self.names:list[str] = []
    self.regs:list[int] = []
    for name, group, countable in COUNTERS:
      try: self.regs.append(kgsl.IOCTL_KGSL_PERFCOUNTER_GET(dev.fd, groupid=group, countable=countable).offset)
      except OSError: continue  # slot exhausted or countable unsupported: profile what we can
      self.names.append(name)
    if len(self.names) < len(COUNTERS): print(f"QCOMPerfCounters: reserved {len(self.names)}/{len(COUNTERS)} counters")
    if not self.names: raise RuntimeError("no gen8 performance counters could be reserved")
    # CP_ALWAYS_ON_COUNTER is a fixed 19.2 MHz free-running clock and needs no reservation.
    self.names.append("ALWAYS_ON_CYCLES")
    self.regs.append(mesa.REG_A8XX_CP_ALWAYS_ON_COUNTER)
    self.slot_vals = len(self.regs)

  def emit_snapshot(self, q:QCOMComputeQueue, base_addr:int):
    # One counter window snapshot: WFI for register visibility, then a
    # CP_REG_TO_MEM per reserved counter into consecutive 64B slots at base_addr. The LO/HI
    # pair lands in the first two dwords of each slot (Turnip counter-pool layout).
    # NOTE: QCOM_PERF_SCOPE is deliberately not honored here. An armed preemption-disable
    # scope wedges the context if the host waits mid-window, and per-kernel profiling always
    # waits between kernels to copy records out.
    q.cmd(mesa.CP_WAIT_FOR_IDLE)
    for i, reg in enumerate(self.regs):
      q.reg_to_mem(reg, base_addr + i * 64, count=0)

  def fetch_record(self, cpu, base_off:int, last_ao:int|None) -> tuple[dict[str, int], int]:
    # Poll the end ALWAYS_ON slot until this record's end snapshot lands, then difference
    # the begin/end slot pairs. Submissions on one KGSL context execute in order, so an end
    # snapshot landing guarantees its begin snapshot has landed too. Returns (deltas, end_ao).
    half = self.slot_vals * 64
    ao_dw = (self.slot_vals - 1) * 16
    deadline = time.monotonic() + 5.0
    while True:
      end_dws = cpu.view(base_off + half, half, 'I')
      ao = end_dws[ao_dw] | (end_dws[ao_dw + 1] << 32)
      if ao and (last_ao is None or ao > last_ao): break
      if time.monotonic() > deadline: raise RuntimeError("gen8 perf counter snapshot never landed")
      time.sleep(0.0005)
    begin_dws = cpu.view(base_off, half, 'I')
    def counter_vals(dws) -> list[int]:  # LO/HI pair = first two dwords of each 64B slot
      return [dws[16*i] | (dws[16*i+1] << 32) for i in range(self.slot_vals)]
    begin_vals, end_vals = counter_vals(begin_dws), counter_vals(end_dws)
    return ({n: (e - b) & 0xffffffffffffffff for n, b, e in zip(self.names, begin_vals, end_vals)}, ao)

  @staticmethod
  def metrics(deltas:dict[str, int]) -> dict[str, float|None]:
    def ratio(num:str, den:str) -> float|None:
      return (n / d) if (n:=deltas.get(num)) is not None and (d:=deltas.get(den)) else None
    def count(*names:str) -> float|None:
      vs = [deltas[n] for n in names if n in deltas]
      return float(sum(vs)) if vs else None
    beats = sum(deltas.get(f"UCHE_VBIF_{rw}_BEATS_CH{i}", 0) for rw in ("READ", "WRITE") for i in (0, 1))
    ao, cp_always = deltas.get("ALWAYS_ON_CYCLES"), deltas.get("CP_ALWAYS_COUNT")
    return {
      "sp_alu_utilization": ratio("SP_ALU_WORKING_CYCLES", "SP_BUSY_CYCLES"),
      "sp_stall_tp_frac": ratio("SP_STALL_CYCLES_TP", "SP_BUSY_CYCLES"),
      "sp_stall_uche_frac": ratio("SP_STALL_CYCLES_UCHE", "SP_BUSY_CYCLES"),
      "sp_stall_rb_frac": ratio("SP_STALL_CYCLES_RB", "SP_BUSY_CYCLES"),
      "sp_stall_vpc_frac": ratio("SP_STALL_CYCLES_VPC_BE", "SP_BUSY_CYCLES"),
      "sp_starve_hlsq_frac": ratio("SP_STARVE_CYCLES_HLSQ", "SP_BUSY_CYCLES"),
      "sp_starved_by_tp_rate": ratio("SP_LOW_EFFICIENCY_STARVED_BY_TP", "SP_BUSY_CYCLES"),
      "sp_gpr_conflicts": count("SP_GPR_READ_CONFLICT", "SP_GPR_WRITE_CONFLICT"),
      "gm_latency_per_sample": ratio("SP_GM_LOAD_LATENCY_CYCLES", "SP_GM_LOAD_LATENCY_SAMPLES"),
      # Mesa fdperf reserves PERF_CP_ALWAYS_COUNT specifically for GPU-frequency measurement.
      # Unlike CP_BUSY_CYCLES, ALWAYS_COUNT advances at the core clock even when the CP is idle.
      "gpu_clock_mhz": (cp_always / ao * AO_HZ / 1e6) if ao and cp_always else None,
      "cp_busy_frac": ratio("CP_BUSY_CYCLES", "CP_ALWAYS_COUNT"),
      # RBBM_STATUS_MASKED is a core-clock-domain busy counter; normalize by the always-counting core clock.
      "rbbm_busy_frac": ratio("RBBM_STATUS_MASKED", "CP_ALWAYS_COUNT"),
      "uche_arb_stall_frac": ratio("UCHE_STALL_CYCLES_ARBITER", "UCHE_BUSY_CYCLES"),
      "vbif_latency_per_sample": ratio("UCHE_VBIF_LATENCY_CYCLES", "UCHE_VBIF_LATENCY_SAMPLES"),
      "tp_stall_uche_frac": ratio("TP_STALL_CYCLES_UCHE", "TP_BUSY_CYCLES"),
      "tp_starve_sp_frac": ratio("TP_STARVE_CYCLES_SP", "TP_BUSY_CYCLES"),
      "tp_starve_uche_frac": ratio("TP_STARVE_CYCLES_UCHE", "TP_BUSY_CYCLES"),
      "tp_l1_misses": count("TP_L1_CACHELINE_MISSES"),
      "preemptions": count("CP_NUM_PREEMPTIONS"),
      "preempt_delay_per": ratio("CP_PREEMPTION_REACTION_DELAY", "CP_NUM_PREEMPTIONS"),
      "half_mad_share": (lambda f, h: h / (f + h) if f is not None and h is not None and f + h else None)(
        deltas.get("SP_FULL_ALU_MAD_INSTRUCTIONS"), deltas.get("SP_HALF_ALU_MAD_INSTRUCTIONS")),
      "icl1_miss_rate": ratio("SP_ICL1_MISSES", "SP_ICL1_REQUESTS"),
      "vbif_beats": float(beats),
      "vbif_bytes_est": beats * 32.0,
    }

  def __del__(self):
    if (dev:=self.dev()) is None: return
    with contextlib.suppress(OSError):
      for name, group, countable in COUNTERS:
        if name in self.names: kgsl.IOCTL_KGSL_PERFCOUNTER_PUT(dev.fd, groupid=group, countable=countable)
