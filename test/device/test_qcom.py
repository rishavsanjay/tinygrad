import array, ctypes, errno, platform, unittest
from types import SimpleNamespace
from unittest import mock
from tinygrad import Device
from tinygrad.device import Compiled
from tinygrad.renderer.cstyle import ClangRenderer

class TestQCOM(unittest.TestCase):
  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

class TestQCOMRetirement(unittest.TestCase):
  @staticmethod
  def device(value:int, timestamp:int):
    from tinygrad.runtime.ops_qcom import QCOMDevice, QCOM_RETIREMENT_RING
    dev = object.__new__(QCOMDevice)
    dev.gen, dev.fd, dev.ctx, dev.wait_timeout_ms = 8, object(), 7, 30
    log = [0] * (QCOM_RETIREMENT_RING * 2)
    log[(value % QCOM_RETIREMENT_RING) * 2:(value % QCOM_RETIREMENT_RING) * 2 + 2] = [value, timestamp]
    dev.__dict__["qcom_retirement_log"] = SimpleNamespace(host=SimpleNamespace(view=lambda **_: log))
    return dev

  def test_visible_value_still_waits_for_exact_submission(self):
    from tinygrad.runtime.autogen import kgsl
    from tinygrad.runtime.ops_qcom import QCOMDevice
    sig = memoryview(array.array('Q', [5, 5]))
    with mock.patch.object(kgsl, 'IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID') as wait, \
         mock.patch.object(Compiled, '_wait_signal') as host_wait:
      QCOMDevice._wait_signal(self.device(5, 123), sig, 5, 20)
    wait.assert_called_once_with(mock.ANY, context_id=7, timestamp=123, timeout=mock.ANY)
    host_wait.assert_called_once_with(sig, 5, 20)

  def test_deadlock_is_retried_until_retirement(self):
    from tinygrad.runtime.autogen import kgsl
    from tinygrad.runtime.ops_qcom import QCOMDevice
    sig = memoryview(array.array('Q', [9, 9]))
    with mock.patch.object(kgsl, 'IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID', side_effect=[OSError(errno.EDEADLK, ''), None]) as wait, \
         mock.patch.object(Compiled, '_wait_signal'):
      QCOMDevice._wait_signal(self.device(9, 456), sig, 9, 20)
    self.assertEqual(wait.call_count, 2)

  def test_retirement_timeout_does_not_accept_visible_ram(self):
    from tinygrad.runtime.autogen import kgsl
    from tinygrad.runtime.ops_qcom import QCOMDevice
    sig = memoryview(array.array('Q', [11, 11]))
    with mock.patch.object(kgsl, 'IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID', side_effect=OSError(errno.ETIMEDOUT, '')), \
         mock.patch.object(Compiled, '_wait_signal'):
      with self.assertRaisesRegex(RuntimeError, 'did not retire'): QCOMDevice._wait_signal(self.device(11, 789), sig, 11, 20)

if __name__ == '__main__':
  unittest.main()
