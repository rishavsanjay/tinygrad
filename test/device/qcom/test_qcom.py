import ctypes, platform, struct, unittest
from types import SimpleNamespace
from tinygrad import Device
from tinygrad.renderer.cstyle import ClangRenderer

class TestQCOM(unittest.TestCase):
  def test_ir3_branchstack_encoding(self):
    from tinygrad.renderer.nir import IR3Renderer
    from tinygrad.runtime.autogen import mesa
    from tinygrad.runtime.support.mesa_abi import MESA_VERSION
    from tinygrad.runtime.ops_qcom import QCOMProgramData, qreg
    dev = SimpleNamespace(renderer=object.__new__(IR3Renderer), compiler=SimpleNamespace(cc=SimpleNamespace(max_branchstack=64)))
    cases = ((0, 0), (1, 0x4000), (3, 0x8000), (65, 0x80000)) if MESA_VERSION == "26.2.5" else ((0, 0), (1, 0x4000), (3, 0xc000))
    for branchstack, expected in cases:
      with self.subTest(branchstack=branchstack):
        variant = mesa.struct_ir3_shader_variant(branchstack=branchstack)
        obj = SimpleNamespace(signature=(), name="branchstack", lib=bytes(variant) + bytes(mesa.struct_ir3_const_state()))
        data = QCOMProgramData(dev, obj)
        self.assertEqual(qreg.a6xx_sp_cs_cntl_0(branchstack=data.brnchstck), expected)

  def test_qcomcl_branchstack_encoding(self):
    from tinygrad.runtime.support.mesa_abi import MESA_VERSION
    from tinygrad.runtime.ops_qcom import QCOMProgramData, qreg
    for branchstack, expected in ((2, 0x4000), (3, 0x8000 if MESA_VERSION == "26.2.5" else 0x4000), (6, 0xc000)):
      with self.subTest(branchstack=branchstack):
        lib = bytearray(1024)
        for offset, value in ((0xc0, 0x300), (0x100, 8), (0x110, 0x120), (0x228, branchstack)):
          struct.pack_into('<I', lib, offset, value)
        data = QCOMProgramData(SimpleNamespace(renderer=None), SimpleNamespace(signature=(), name='branchstack', lib=bytes(lib)))
        self.assertEqual(qreg.a6xx_sp_cs_cntl_0(branchstack=data.brnchstck), expected)

  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

if __name__ == '__main__':
  unittest.main()
