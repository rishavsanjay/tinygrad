"""Pure-logic unit tests for Adreno 830 / A7xx+ support."""
import unittest
from tinygrad.runtime.ops_qcom import _decode_chip_id, pkt7_hdr
from tinygrad.runtime.autogen import mesa

class TestChipIdDecoder(unittest.TestCase):
  def test_old_format_a630(self): self.assertEqual(_decode_chip_id(0x06030001), (630, 6, "a630"))
  def test_old_format_a730(self): self.assertEqual(_decode_chip_id(0x07030001), (730, 7, "a730"))
  def test_new_format_a740(self): self.assertEqual(_decode_chip_id(0x43050a01), (740, 7, "a740"))
  def test_new_format_a830(self): self.assertEqual(_decode_chip_id(0x44050001), (830, 8, "a830"))

class TestPacketHeaders(unittest.TestCase):
  def test_pkt7_hdr(self):
    hdr = pkt7_hdr(mesa.CP_EVENT_WRITE, 4)
    self.assertEqual(hdr & mesa.CP_TYPE7_PKT, mesa.CP_TYPE7_PKT)
    self.assertEqual((hdr >> 16) & 0x7F, mesa.CP_EVENT_WRITE)

class TestA7xxRegisterDeltas(unittest.TestCase):
  def test_deltas(self):
    self.assertNotEqual(mesa.REG_A6XX_SP_UPDATE_CNTL, mesa.REG_A7XX_SP_UPDATE_CNTL)
    self.assertNotEqual(mesa.REG_A6XX_SP_CS_NDRANGE_0, mesa.REG_A7XX_SP_CS_NDRANGE_0)

class TestIR3CompilerArch(unittest.TestCase):
  def test_a630(self):
    from tinygrad.runtime.support.compiler_mesa import IR3Compiler
    c = IR3Compiler("a630")
    self.assertEqual(c.dev_id.gpu_id, 630)
    del c
  def test_a740_proxy(self):
    from tinygrad.runtime.support.compiler_mesa import IR3Compiler
    c = IR3Compiler("a740,chip_id=0x43050a01")
    self.assertEqual(c.cc.gen, 7)
    del c
  def test_a830_version_adaptive(self):
    from tinygrad.runtime.support.compiler_mesa import IR3Compiler
    info_raw = mesa.fd_dev_info_raw(mesa.struct_fd_dev_id(830, 0x44050001))
    a830_recognized = info_raw and info_raw.contents.chip != 0
    if a830_recognized:
      c = IR3Compiler("a830,chip_id=0x44050001")
      self.assertEqual(c.cc.gen, 8)
      del c
    else:
      with self.assertRaises(RuntimeError):
        IR3Compiler("a830,chip_id=0x44050001")

if __name__ == "__main__": unittest.main()
