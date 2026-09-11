import decimal, struct, unittest

from tinygrad.device import ProfileProgramEvent
from tinygrad.helpers import ProfileEvent, ProfileRangeEvent
from tinygrad.runtime.support.qcom_profile import COUNTERS, COUNTERS_CORE, COUNTERS_GROUPS, QCOMPerfCounters, QCOMPMCSample, \
  QCOMProfilePMCEvent, resolve_counters
from tinygrad.viz.serve import VizData, load_gpu_counters, unpack_pmc

# Synthetic per-kernel deltas in QCOMPerfCounters.COUNTERS order (names only, values arbitrary but sane).
NAMES = [n for n, _, _ in COUNTERS]

def make_deltas(**over) -> dict:
  d = {n: 0 for n in NAMES}
  d.update({"SP_BUSY_CYCLES": 1000, "SP_ALU_WORKING_CYCLES": 800, "SP_STALL_CYCLES_TP": 50, "SP_STALL_CYCLES_UCHE": 150,
            "SP_FULL_ALU_MAD_INSTRUCTIONS": 400, "SP_HALF_ALU_MAD_INSTRUCTIONS": 100,
            "SP_FULL_ALU_MUL_INSTRUCTIONS": 10, "SP_FULL_ALU_ADD_INSTRUCTIONS": 10,
            "SP_ICL1_REQUESTS": 100, "SP_ICL1_MISSES": 5, "ALWAYS_ON_CYCLES": 19200, "CP_ALWAYS_COUNT": 1000000,
            "UCHE_VBIF_READ_BEATS_CH0": 100, "UCHE_VBIF_READ_BEATS_CH1": 100,
            "UCHE_VBIF_WRITE_BEATS_CH0": 10, "UCHE_VBIF_WRITE_BEATS_CH1": 10})
  d.update(over)
  return d

def make_event(deltas:dict, kern=0, tag=1, info=None) -> QCOMProfilePMCEvent:
  names = list(deltas.keys())
  sched = [QCOMPMCSample(n, off=8*i) for i, n in enumerate(names)]
  return QCOMProfilePMCEvent("QCOM", kern, sched, struct.pack(f"<{len(names)}Q", *(deltas[n] for n in names)), tag, info)

class TestQCOMPMCEvent(unittest.TestCase):
  def test_unpack_consumes_one_u64_per_entry(self):
    d = make_deltas()
    table = unpack_pmc(make_event(d))
    rows = {r[0]: r[1] for r in table["rows"]}
    for n, v in d.items(): self.assertEqual(rows[n], v)

  def test_derived_metric_rows(self):
    rows = {r[0]: r[1] for r in unpack_pmc(make_event(make_deltas()))["rows"]}
    self.assertEqual(rows["SP ALU utilization"], "80.0%")
    self.assertEqual(rows["SP stall TP"], "5.0%")
    self.assertEqual(rows["MAD full share"], f"{100*400/520:.1f}%")
    self.assertEqual(rows["ICL1 miss rate"], "5.00%")
    self.assertEqual(rows["VBIF bytes"], f"{32*220:,}")

  def test_missing_counters_skip_metrics_without_crash(self):
    d = {"SP_BUSY_CYCLES": 100, "ALWAYS_ON_CYCLES": 50}  # subset: no ALU/stall/ICL1/VBIF names
    rows = {r[0]: r[1] for r in unpack_pmc(make_event(d))["rows"]}
    self.assertEqual(rows["SP_BUSY_CYCLES"], 100)
    self.assertNotIn("SP ALU utilization", rows)

  def test_info_rows_surface_geometry_and_ir3_meta(self):
    info = {"global_size": "(64, 1, 1)", "local_size": "(64, 1, 1)", "fregs": 12, "hregs": 6, "threadsize": 64}
    rows = {r[0]: r[1] for r in unpack_pmc(make_event(make_deltas(), info=info))["rows"]}
    self.assertEqual(rows["info:global_size"], "(64, 1, 1)")
    self.assertEqual(rows["info:fregs"], "12")

  def test_metrics_static(self):
    m = QCOMPerfCounters.metrics(make_deltas())
    self.assertAlmostEqual(m["sp_alu_utilization"], 0.8)
    self.assertAlmostEqual(m["half_mad_share"], 100/500)
    self.assertEqual(m["vbif_bytes_est"], 220*32.0)

  def test_load_qcom_counters_builds_ctxs(self):
    d = make_deltas()
    profile: list[ProfileEvent] = [
      ProfileProgramEvent("QCOM", "test_kernel", None, None, 3),
      ProfileRangeEvent("QCOM", "test_kernel", decimal.Decimal(100), decimal.Decimal(200)),
      make_event(d, kern=3, tag=7, info={"global_size": "(64, 1, 1)"}),
    ]
    data = VizData()
    load_gpu_counters(data, profile, "QCOM")
    names = [c["name"] for c in data.ctxs]
    self.assertIn("All Counters", names)
    self.assertIn("QCOM test_kernel", names)
    pmc_step = next(s for c in data.ctxs if c["name"] == "QCOM test_kernel" for s in c["steps"] if s["name"] == "PMC")
    self.assertEqual(unpack_pmc(pmc_step["_data"])["rows"][0][0], next(iter(d)))

class TestQCOMPMCExtra(unittest.TestCase):
  def test_resolve_default_is_core_only(self):
    self.assertEqual(resolve_counters(""), COUNTERS_CORE)
    self.assertEqual(len(COUNTERS), len(COUNTERS_CORE))  # default env: no extras

  def test_resolve_all_and_subset(self):
    self.assertEqual(len(resolve_counters("all")), len(COUNTERS_CORE) + sum(len(g) for g in COUNTERS_GROUPS.values()))
    sub = resolve_counters("stall,sys")
    self.assertEqual([n for n, _, _ in sub[len(COUNTERS_CORE):]],
                     ["SP_STALL_CYCLES_VPC_BE", "SP_STALL_CYCLES_RB",
                      "RBBM_STATUS_MASKED", "CP_BUSY_CYCLES", "CP_NUM_PREEMPTIONS", "CP_PREEMPTION_REACTION_DELAY"])
    with self.assertRaises(ValueError): resolve_counters("bogus")

  def test_metrics_extended(self):
    m = QCOMPerfCounters.metrics({"ALWAYS_ON_CYCLES": 19200, "CP_ALWAYS_COUNT": 1000000,  # 1ms window @1000MHz
                                  "CP_BUSY_CYCLES": 800000, "RBBM_STATUS_MASKED": 900000, "SP_BUSY_CYCLES": 1000,
                                  "SP_STALL_CYCLES_RB": 100, "SP_STALL_CYCLES_VPC_BE": 50,
                                  "SP_STARVE_CYCLES_HLSQ": 200, "SP_GPR_READ_CONFLICT": 3, "SP_GPR_WRITE_CONFLICT": 4,
                                  "SP_GM_LOAD_LATENCY_CYCLES": 500, "SP_GM_LOAD_LATENCY_SAMPLES": 10,
                                  "UCHE_BUSY_CYCLES": 800, "UCHE_STALL_CYCLES_ARBITER": 80,
                                  "TP_BUSY_CYCLES": 600, "TP_STALL_CYCLES_UCHE": 60,
                                  "CP_NUM_PREEMPTIONS": 2, "CP_PREEMPTION_REACTION_DELAY": 40})
    self.assertAlmostEqual(m["gpu_clock_mhz"], 1000.0)
    self.assertAlmostEqual(m["cp_busy_frac"], 0.8)
    self.assertAlmostEqual(m["rbbm_busy_frac"], 0.9)
    self.assertAlmostEqual(m["sp_stall_rb_frac"], 0.1)
    self.assertAlmostEqual(m["sp_starve_hlsq_frac"], 0.2)
    self.assertEqual(m["sp_gpr_conflicts"], 7.0)
    self.assertAlmostEqual(m["gm_latency_per_sample"], 50.0)
    self.assertAlmostEqual(m["uche_arb_stall_frac"], 0.1)
    self.assertAlmostEqual(m["tp_stall_uche_frac"], 0.1)
    self.assertEqual(m["preemptions"], 2.0)
    self.assertAlmostEqual(m["preempt_delay_per"], 20.0)
    # absent extras stay None instead of crashing
    m2 = QCOMPerfCounters.metrics(make_deltas())
    self.assertIsNone(m2["rbbm_busy_frac"])
    self.assertIsNone(m2["preemptions"])
    self.assertAlmostEqual(m2["gpu_clock_mhz"], 1000.0)  # core set includes CP_ALWAYS_COUNT

  def test_viz_extra_rows(self):
    d = make_deltas(SP_STALL_CYCLES_RB=100, SP_STALL_CYCLES_VPC_BE=50, SP_STARVE_CYCLES_HLSQ=200,
                    SP_GPR_READ_CONFLICT=3, SP_GPR_WRITE_CONFLICT=4, CP_ALWAYS_COUNT=1005000, CP_BUSY_CYCLES=800000,
                    RBBM_STATUS_MASKED=900000, TP_BUSY_CYCLES=600, TP_STALL_CYCLES_UCHE=60,
                    TP_L1_CACHELINE_MISSES=7, CP_NUM_PREEMPTIONS=2)
    rows = {r[0]: r[1] for r in unpack_pmc(make_event(d))["rows"]}
    self.assertEqual(rows["GPU clock"], "1005MHz")
    self.assertEqual(rows["SP stall RB"], "10.0%")
    self.assertEqual(rows["SP GPR conflicts"], "7")
    self.assertEqual(rows["TP L1 misses"], "7")
    self.assertEqual(rows["Preemptions"], "2")
    core_rows = {r[0]: r[1] for r in unpack_pmc(make_event(make_deltas()))["rows"]}
    self.assertNotIn("TP L1 misses", core_rows)

if __name__ == "__main__": unittest.main()
