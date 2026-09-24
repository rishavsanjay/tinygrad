import ctypes, gc, os, platform, threading, unittest, weakref
from types import SimpleNamespace
from unittest import mock
from tinygrad import Device, Variable, dtypes
from tinygrad.helpers import getenv
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.runtime.autogen import kgsl, mesa
from tinygrad.runtime.ops_qcom import QCOMComputeQueue, QCOMDevice, QCOMSignal, _QCOMSubmission, qcom_last_local_size, qcom_validate_launch
from tinygrad.runtime.ops_qcom import QCOMRingAllocator, QCOMAllocator
from tinygrad.device import LRUAllocator
from tinygrad.runtime.support.hcq import HCQBuffer, HCQCompiled, HWQueue, MMIOInterface
from tinygrad.uop.ops import Ops, UOp

class TestQCOM(unittest.TestCase):
  def test_failed_cpu_mapping_releases_gpu_allocation(self):
    dev = SimpleNamespace(gen=8, fd=SimpleNamespace(mmap=mock.Mock(side_effect=OSError("mmap failed"))))
    allocation = SimpleNamespace(id=17, flags=kgsl.KGSL_MEMFLAGS_IOCOHERENT)
    with mock.patch.object(kgsl, 'IOCTL_KGSL_GPUOBJ_ALLOC', return_value=allocation), \
         mock.patch.object(kgsl, 'IOCTL_KGSL_GPUOBJ_FREE') as free:
      with self.assertRaisesRegex(OSError, 'mmap failed'): QCOMDevice._gpu_alloc(dev, 4096)
    free.assert_called_once_with(dev.fd, id=17)

  def test_scratch_growth_retains_existing_program_storage(self):
    free = mock.Mock()
    dev = SimpleNamespace(gen=8, allocator=SimpleNamespace(free=free), _gpu_alloc=lambda size: SimpleNamespace(size=size))
    QCOMDevice._ensure_stack_size(dev, 4096)
    program_scratch = dev._scratch
    QCOMDevice._ensure_stack_size(dev, 8192)
    self.assertIsNot(program_scratch, dev._scratch)
    free.assert_not_called()
    del program_scratch
    gc.collect()
    self.assertEqual(free.call_count, 1)
    self.assertEqual(free.call_args.args[1], 4096)

  def test_ring_wrap_retires_last_submitted_command(self):
    dev = SimpleNamespace(fd=1, ctx=2, _last_submit_timestamp=7)
    ring = QCOMRingAllocator(dev, 32)
    with mock.patch.object(kgsl, 'IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID') as wait:
      self.assertEqual(ring.alloc(24), 0)
      wait.assert_not_called()
      self.assertEqual(ring.alloc(16), 0)
      wait.assert_called_once_with(1, context_id=2, timestamp=7, timeout=30000)
      with self.assertRaises(ValueError): ring.alloc(33)

  def test_cached_buffer_reuse_retires_users(self):
    allocator = object.__new__(QCOMAllocator)
    dev = SimpleNamespace(gen=8, synchronize=mock.Mock())
    LRUAllocator.__init__(allocator, dev)
    allocator.cache[(16, None)].append('buffer')
    self.assertEqual(allocator.alloc(16), 'buffer')
    dev.synchronize.assert_called_once_with()

  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

  def test_a8xx_last_local_size_exact_groups(self):
    self.assertEqual(qcom_last_local_size((128, 64, 32), (16, 8, 4)), (16, 8, 4))

  def test_a8xx_last_local_size_partial_groups(self):
    self.assertEqual(qcom_last_local_size((50, 9, 18), (8, 4, 4)), (2, 1, 2))

  def test_launch_limits_and_partial_groups(self):
    qcom_validate_launch((6.25, 1, 1), (8, 1, 1))
    for gs,ls in (((1,1,1), (1024,2,1)), ((float('nan'),1,1), (1,1,1)), ((0,1,1), (1,1,1)),
                  ((2**32,1,1), (1,1,1)), ((0.1,1,1), (1,1,1))):
      with self.subTest(gs=gs, ls=ls), self.assertRaises(ValueError): qcom_validate_launch(gs, ls)

  def test_symbolic_launch_rechecked_before_packet_patch(self):
    queue = QCOMComputeQueue(SimpleNamespace(gen=8))
    count = Variable('count', 1, 2048)
    queue._launches.append(((1,1,1), (count,1,1)))
    queue._apply_var_vals({'count':64})
    with self.assertRaises(ValueError): queue._apply_var_vals({'count':2048})

class TestQCOMRetirement(unittest.TestCase):
  @staticmethod
  def signal(value=0, is_timeline=True):
    signal = object.__new__(QCOMSignal)
    signal.owner = SimpleNamespace(gen=8, fd=1, ctx=2, _submitted_retirement_signals=weakref.WeakSet())
    signal.is_timeline = is_timeline
    signal._command_timestamps, signal._retirement_lock, signal.should_return = {}, threading.Lock(), False
    signal._retired_value = None
    signal.owner._submitted_retirement_signals.add(signal)
    return signal, [value]

  @staticmethod
  def physical_signal(value=0, is_timeline=False):
    backing = (ctypes.c_uint64 * 2)()
    dev = SimpleNamespace(gen=8, fd=1, ctx=2, _retirement_signals=weakref.WeakValueDictionary(),
                          _submitted_retirement_signals=weakref.WeakSet())
    signal = QCOMSignal(HCQBuffer(ctypes.addressof(backing), ctypes.sizeof(backing), view=MMIOInterface(ctypes.addressof(backing),
                        ctypes.sizeof(backing))), value=value, owner=dev, is_timeline=is_timeline)
    signal.should_return, signal._test_backing = False, backing
    return signal, backing

  @staticmethod
  def queue(dev, signals):
    if not hasattr(dev, '_submitted_retirement_signals'): dev._submitted_retirement_signals = weakref.WeakSet()
    queue = object.__new__(QCOMComputeQueue)
    queue.dev, queue.binded_device, queue.submit_req, queue._signals = dev, dev, object(), signals
    dev.allocator, queue.hw_page = SimpleNamespace(free=lambda *args, **kwargs: None), SimpleNamespace(size=0)
    return queue

  def test_wait_blocks_until_submit_installs_timestamp(self):
    signal, value = self.signal()
    dev, entered, release = signal.owner, threading.Event(), threading.Event()
    queue = self.queue(dev, [(signal, 1)])
    errors:list[BaseException] = []

    def submit(*args, **kwargs):
      value[0] = 1
      entered.set()
      release.wait(1)
      return SimpleNamespace(timestamp=9)
    def run_submit():
      try: queue._submit(dev)
      except BaseException as e: errors.append(e)
    def run_wait(done):
      try: signal.wait(1, timeout=1000)
      except BaseException as e: errors.append(e)
      finally: done.set()

    with mock.patch.object(QCOMSignal, "value", new_callable=mock.PropertyMock, side_effect=lambda: value[0]), \
         mock.patch.object(kgsl, "IOCTL_KGSL_GPU_COMMAND", side_effect=submit), \
         mock.patch.object(kgsl, "IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID") as wait_timestamp:
      submit_thread = threading.Thread(target=run_submit)
      submit_thread.start()
      self.assertTrue(entered.wait(1))
      done = threading.Event()
      wait_thread = threading.Thread(target=run_wait, args=(done,))
      wait_thread.start()
      self.assertFalse(done.wait(0.02))
      release.set()
      submit_thread.join(1)
      wait_thread.join(1)

    self.assertFalse(errors)
    wait_timestamp.assert_called_once_with(dev.fd, context_id=dev.ctx, timestamp=9, timeout=mock.ANY)

  def test_wait_timeout_is_total_budget_despite_signal_progress(self):
    signal, value = self.signal()
    clock = [0.0]
    def progress(_):
      clock[0] += 0.006
      value[0] += 1
    with mock.patch.object(QCOMSignal, "value", new_callable=mock.PropertyMock, side_effect=lambda: value[0]), \
         mock.patch.object(QCOMSignal, "_sleep", side_effect=progress), \
         mock.patch("tinygrad.runtime.ops_qcom.time.perf_counter", side_effect=lambda: clock[0]):
      with self.assertRaisesRegex(RuntimeError, "Wait timeout: 10 ms"): signal.wait(4, timeout=10)
    self.assertEqual(value[0], 2)

  def test_wait_zero_timeout_is_nonblocking(self):
    signal, value = self.signal()
    clock = [0.0]
    with mock.patch.object(QCOMSignal, "value", new_callable=mock.PropertyMock, side_effect=lambda: value[0]), \
         mock.patch.object(QCOMSignal, "_sleep", side_effect=lambda _: clock.__setitem__(0, clock[0] + 0.008)), \
         mock.patch("tinygrad.runtime.ops_qcom.time.perf_counter", side_effect=lambda: clock[0]), \
         mock.patch("tinygrad.runtime.ops_qcom.getenv", return_value=7) as get_timeout:
      with self.assertRaisesRegex(RuntimeError, "Wait timeout: 0 ms"): signal.wait(1, timeout=0)
    get_timeout.assert_not_called()
    self.assertEqual(clock[0], 0)

  def test_ge_wait_retires_visible_satisfying_value(self):
    signal, value = self.signal(value=6)
    record = _QCOMSubmission(signal.owner)
    record.timestamp = 12
    signal._command_timestamps[6] = [record]
    with mock.patch.object(QCOMSignal, "value", new_callable=mock.PropertyMock, side_effect=lambda: value[0]), \
         mock.patch.object(kgsl, "IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID") as wait_timestamp:
      signal.wait(5, timeout=100)
    wait_timestamp.assert_called_once_with(signal.owner.fd, context_id=signal.owner.ctx, timestamp=12, timeout=mock.ANY)
    self.assertEqual(signal._command_timestamps, {})

  def test_visible_newer_value_ignores_older_pending_record(self):
    signal, value = self.signal(value=1)
    signal._command_timestamps[0] = [_QCOMSubmission(signal.owner)]
    with mock.patch.object(QCOMSignal, "value", new_callable=mock.PropertyMock, side_effect=lambda: value[0]), \
         mock.patch.object(kgsl, "IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID") as wait_timestamp:
      signal.wait(0, timeout=100)
    wait_timestamp.assert_not_called()

  def test_wait_stops_after_selected_submission_when_value_advances(self):
    signal, value = self.signal(value=5)
    records = [_QCOMSubmission(signal.owner), _QCOMSubmission(signal.owner)]
    records[0].timestamp, records[1].timestamp = 10, 11
    signal._command_timestamps = {5:[records[0]], 6:[records[1]]}
    with mock.patch.object(QCOMSignal, "value", new_callable=mock.PropertyMock, side_effect=lambda: value[0]), \
         mock.patch.object(kgsl, "IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID",
                           side_effect=lambda *args, **kwargs: value.__setitem__(0, 6)) as wait_timestamp:
      signal.wait(5, timeout=100)
    wait_timestamp.assert_called_once_with(signal.owner.fd, context_id=signal.owner.ctx, timestamp=10, timeout=mock.ANY)
    self.assertEqual(signal._command_timestamps, {6:[records[1]]})

  def test_symbolic_signal_metadata_binds_to_physical_owner(self):
    dev = SimpleNamespace(gen=8, _retirement_signals={0x1234:"physical"})
    signal_addr, signal_value = Variable("signal_addr", 0, 0xffffffffffffffff), Variable("signal_value", 0, 0xffffffff)
    virtual = SimpleNamespace(value_addr=signal_addr)
    queue = object.__new__(QCOMComputeQueue)
    queue.dev, queue.binded_device, queue._signals = dev, None, [(virtual, signal_value)]
    with mock.patch.object(HWQueue, "_apply_var_vals"):
      queue._apply_var_vals({signal_addr.expr:0x1234, signal_value.expr:7})
    self.assertEqual(queue._resolved_signals, [("physical", 7)])

  def test_submit_failure_removes_pending_record(self):
    signal, _ = self.signal()
    queue = self.queue(signal.owner, [(signal, 1)])
    queue._resolved_signals = [(signal, 1)]
    with mock.patch.object(kgsl, "IOCTL_KGSL_GPU_COMMAND", side_effect=OSError("submit failed")):
      with self.assertRaises(OSError): queue._submit(signal.owner)
    self.assertEqual(signal._command_timestamps, {})
    self.assertNotIn("_resolved_signals", queue.__dict__)

  def test_failed_duplicate_submission_preserves_older_record(self):
    signal, value = self.signal()
    queue = self.queue(signal.owner, [(signal, 5)])
    with mock.patch.object(kgsl, "IOCTL_KGSL_GPU_COMMAND", side_effect=(SimpleNamespace(timestamp=10), OSError("submit failed"))):
      queue._submit(signal.owner)
      with self.assertRaises(OSError): queue._submit(signal.owner)
    self.assertEqual(len(signal._command_timestamps[5]), 1)
    self.assertEqual(signal._command_timestamps[5][0].timestamp, 10)

    value[0] = 5
    with mock.patch.object(QCOMSignal, "value", new_callable=mock.PropertyMock, side_effect=lambda: value[0]), \
         mock.patch.object(kgsl, "IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID") as wait_timestamp:
      signal.wait(5, timeout=100)
    wait_timestamp.assert_called_once_with(signal.owner.fd, context_id=signal.owner.ctx, timestamp=10, timeout=mock.ANY)

  def test_successful_duplicate_second_wait_uses_retired_satisfaction(self):
    signal, value = self.signal(value=5)
    records = [_QCOMSubmission(signal.owner), _QCOMSubmission(signal.owner)]
    records[0].timestamp, records[1].timestamp = 10, 11
    signal._command_timestamps[5] = records
    with mock.patch.object(QCOMSignal, "value", new_callable=mock.PropertyMock, side_effect=lambda: value[0]), \
         mock.patch.object(kgsl, "IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID") as wait_timestamp:
      signal.wait(5, timeout=100)
      signal.wait(5, timeout=100)
    wait_timestamp.assert_called_once_with(signal.owner.fd, context_id=signal.owner.ctx, timestamp=10, timeout=mock.ANY)
    self.assertEqual(signal._command_timestamps, {5:[records[1]]})

  def test_successful_signal_reset_reuse_stays_bounded(self):
    signal, backing = self.physical_signal()
    queue = self.queue(signal.owner, [(signal, 5)])
    timestamps = iter(range(1, 1001))
    with mock.patch.object(kgsl, "IOCTL_KGSL_GPU_COMMAND", side_effect=lambda *args, **kwargs: SimpleNamespace(timestamp=next(timestamps))), \
         mock.patch.object(kgsl, "IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID") as wait_timestamp:
      for _ in range(1000):
        signal.value = 0
        queue._submit(signal.owner)
        signal.value = 0  # Host writes invalidate retired satisfaction, but must preserve the pending GPU record.
        self.assertEqual(sum(map(len, signal._command_timestamps.values())), 1)
        backing[0] = 5
        signal.wait(5, timeout=100)
        signal.wait(5, timeout=100)
        self.assertEqual(signal._command_timestamps, {})
    self.assertEqual(wait_timestamp.call_count, 1000)

  def test_timeline_retirement_clears_prior_dependency_records(self):
    timeline, value = self.signal(value=1)
    dependency, _ = self.signal(value=5, is_timeline=False)
    dependency.owner = timeline.owner
    timeline.owner._submitted_retirement_signals.add(dependency)
    dependency_record, timeline_record = _QCOMSubmission(timeline.owner), _QCOMSubmission(timeline.owner)
    dependency_record.timestamp, timeline_record.timestamp = 10, 11
    dependency._command_timestamps[5], timeline._command_timestamps[1] = [dependency_record], [timeline_record]
    with mock.patch.object(QCOMSignal, "value", new_callable=mock.PropertyMock, side_effect=lambda: value[0]), \
         mock.patch.object(kgsl, "IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID"):
      timeline.wait(1, timeout=100)
    self.assertEqual(dependency._command_timestamps, {})
    self.assertEqual(timeline._command_timestamps, {})

  def test_retirement_pruning_is_submitting_context_scoped(self):
    signal, value = self.signal(value=5)
    other_dev = SimpleNamespace(gen=8, fd=3, ctx=4, _submitted_retirement_signals=weakref.WeakSet())
    records = [_QCOMSubmission(signal.owner), _QCOMSubmission(other_dev)]
    records[0].timestamp, records[1].timestamp = 10, 1
    signal._command_timestamps = {5:[records[0]], 6:[records[1]]}
    with mock.patch.object(QCOMSignal, "value", new_callable=mock.PropertyMock, side_effect=lambda: value[0]), \
         mock.patch.object(kgsl, "IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID"):
      signal.wait(5, timeout=100)
    self.assertEqual(signal._command_timestamps, {6:[records[1]]})

  def test_retirement_uses_submitting_context(self):
    signal, value = self.signal()
    submitter = SimpleNamespace(gen=8, fd=3, ctx=4)
    queue = self.queue(submitter, [(signal, 5)])
    with mock.patch.object(kgsl, "IOCTL_KGSL_GPU_COMMAND", return_value=SimpleNamespace(timestamp=10)):
      queue._submit(submitter)
    value[0] = 5
    with mock.patch.object(QCOMSignal, "value", new_callable=mock.PropertyMock, side_effect=lambda: value[0]), \
         mock.patch.object(kgsl, "IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID") as wait_timestamp:
      signal.wait(5, timeout=100)
    wait_timestamp.assert_called_once_with(submitter.fd, context_id=submitter.ctx, timestamp=10, timeout=mock.ANY)

  def test_timeline_wrap_clears_reused_signal_metadata(self):
    dev = object.__new__(QCOMDevice)
    dev.gen = 8
    dev.timeline_signal, _ = self.signal()
    dev._shadow_timeline_signal, _ = self.signal()
    dev.timeline_signal._command_timestamps[5] = [_QCOMSubmission(dev.timeline_signal.owner)]
    dev._shadow_timeline_signal._command_timestamps[6] = [_QCOMSubmission(dev._shadow_timeline_signal.owner)]
    with mock.patch.object(HCQCompiled, "_wrap_timeline_signal"):
      dev._wrap_timeline_signal()
    self.assertEqual(dev.timeline_signal._command_timestamps, {})
    self.assertEqual(dev._shadow_timeline_signal._command_timestamps, {})

class TestQCOMDispatchSync(unittest.TestCase):
  @staticmethod
  def queue(gen):
    queue = object.__new__(QCOMComputeQueue)
    queue.dev, queue.cmd, queue.binded_device = SimpleNamespace(gen=gen), mock.Mock(), None
    return queue

  def test_a8xx_dispatch_sync_defaults_on(self):
    queue = self.queue(8)
    with mock.patch("tinygrad.runtime.ops_qcom.getenv", return_value=1):
      queue._dispatch_wait()
    queue.cmd.assert_called_once_with(mesa.CP_WAIT_FOR_IDLE)

  def test_a8xx_dispatch_wait_can_be_disabled(self):
    queue = self.queue(8)
    with mock.patch("tinygrad.runtime.ops_qcom.getenv", return_value=0):
      queue._dispatch_wait()
    queue.cmd.assert_not_called()

  def test_a6xx_dispatch_sync_is_not_affected_by_a8xx_knobs(self):
    queue = self.queue(6)
    with mock.patch("tinygrad.runtime.ops_qcom.getenv", return_value=0) as get_env:
      queue._dispatch_wait()
    get_env.assert_not_called()
    queue.cmd.assert_called_once_with(mesa.CP_WAIT_FOR_IDLE)

class TestQCOMPrecisionOptions(unittest.TestCase):
  @staticmethod
  def half_product_cast():
    return (UOp.const(1.001, dtypes.half) * UOp.const(1.003, dtypes.half)).cast(dtypes.float)

  def test_f16_mad_rewrite_defaults_off(self):
    with mock.patch.dict(os.environ, {"QCOM_F16_MAD": "0"}):
      getenv.cache_clear()
      self.assertIsNone(IR3Renderer.extra_matcher.rewrite(self.half_product_cast()))

  def test_f16_mad_rewrite_is_explicit_opt_in(self):
    with mock.patch.dict(os.environ, {"QCOM_F16_MAD": "1"}):
      getenv.cache_clear()
      rewritten = IR3Renderer.extra_matcher.rewrite(self.half_product_cast())
    self.assertIsNotNone(rewritten)
    assert rewritten is not None
    self.assertEqual(rewritten.op, Ops.MUL)
    self.assertEqual(rewritten.dtype, dtypes.float)
    self.assertTrue(all(x.op is Ops.CAST and x.src[0].dtype == dtypes.half for x in rewritten.src))

QCOM_HW = os.getenv("DEV", "").startswith("QCOM")

class TestQCOMPerfCounters(unittest.TestCase):
  # gen8 counters are kernel-managed (KGSL PERFCTR ioctls) and read back with CP_REG_TO_MEM.
  @unittest.skipUnless(QCOM_HW, "run with DEV=QCOM")
  def test_counters_discriminate_instruction_classes(self):
    from tinygrad import Tensor
    dev, perf = Device["QCOM"], Device["QCOM"].perf
    a = Tensor.randn(1 << 18).realize().to("QCOM")
    def fma_chain():
      x = a
      for _ in range(4): x = x * 1.0001 + 0.5
      return x.realize()
    fma_chain(), dev.synchronize()
    with perf:
      for _ in range(10): fma_chain()
    deltas = perf.results
    self.assertGreater(deltas["ALWAYS_ON_CYCLES"], 0)
    # an fp32 FMA chain must issue full MADs and no half or unfused mul/add instructions
    self.assertGreater(deltas["SP_FULL_ALU_MAD_INSTRUCTIONS"], 0)
    for unfused in ("SP_HALF_ALU_MAD_INSTRUCTIONS", "SP_FULL_ALU_MUL_INSTRUCTIONS", "SP_FULL_ALU_ADD_INSTRUCTIONS"):
      self.assertEqual(deltas[unfused], 0)
    self.assertGreater(deltas["SP_CS_INVOCATIONS"], 0)

if __name__ == '__main__':
  unittest.main()
