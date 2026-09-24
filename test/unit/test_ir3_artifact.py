import struct, unittest
from dataclasses import replace
from tinygrad.runtime.support.ir3 import IR3Shader

class TestIR3Artifact(unittest.TestCase):
  def shader(self):
    return IR3Shader('a800,chip_id=0x44050001', 'test-build', 0, 0, 0, False, False, False, 1, False, 4,
                     252, 252, 252, 0, 32, 1, (0,), 1, 0, False, ((1, 0, 4), (0, 8, 8)), (0,))

  def test_roundtrip(self):
    shader, imm, code = self.shader(), b'\x01\x00\x00\x00', b'\x00'*128
    self.assertEqual(IR3Shader.unpack(shader.pack(imm, code)), (shader, imm, code))

  def test_corrupt_and_truncated(self):
    data = self.shader().pack(b'', b'\x00'*128)
    for bad in (data[:10], data[:-1], data+b'x', b'OLD!'+data[4:], data[:-1]+b'x', data[:4]+struct.pack('<I', 0xffffffff)+data[8:]):
      with self.subTest(size=len(bad)), self.assertRaises(ValueError): IR3Shader.unpack(bad)

  def test_invalid_resources(self):
    for changes in ({'tex_to_image':(1,)}, {'params':((0, 1020, 8),)}, {'params':((0, 0, 8), (1, 4, 4))},
                    {'constlen':513}, {'writes':(-1,)}, {'fregs':True}):
      with self.subTest(changes=changes), self.assertRaises(ValueError): replace(self.shader(), **changes).pack(b'', b'code')

  def test_immediate_overflow(self):
    with self.assertRaisesRegex(ValueError, 'immediates'):
      IR3Shader.unpack(replace(self.shader(), imm_off=1024).pack(b'1234', b'code'))

if __name__ == '__main__': unittest.main()
