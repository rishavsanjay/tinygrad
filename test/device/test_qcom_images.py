import unittest
import numpy as np
from tinygrad.dtype import dtypes
from tinygrad.codegen.late.coalesce import image_valid_dims, transform_to_image
from tinygrad.helpers import Context, Target
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import qcom_image_layout, qcom_image_descriptor, qcom_sampler_descriptor

def linear_image_roundtrip(dtype, values):
  layout = qcom_image_layout(dtype, values.shape)
  raw = bytearray(layout.size)
  row_bytes = values.shape[1] * values.shape[2] * values.dtype.itemsize
  for y in range(values.shape[0]): raw[y*layout.row_pitch:y*layout.row_pitch+row_bytes] = values[y].tobytes()
  def read(x, y):
    if x < 0 or y < 0 or x >= layout.width or y >= layout.height: return np.zeros(4, dtype=values.dtype)
    off = y*layout.row_pitch + x*4*values.dtype.itemsize
    return np.frombuffer(raw, dtype=values.dtype, count=4, offset=off).copy()
  return layout, raw, read

class TestQCOMImageLayout(unittest.TestCase):
  def test_fp16_padding(self):
    layout = qcom_image_layout(dtypes.half, (7, 17, 4))
    self.assertEqual((layout.row_pitch, layout.array_pitch, layout.pitch_alignment, layout.size), (256, 2048, 128, 2048))

  def test_fp32_padding(self):
    layout = qcom_image_layout(dtypes.float, (3, 17, 4))
    self.assertEqual((layout.row_pitch, layout.array_pitch, layout.pitch_alignment, layout.size), (512, 2048, 256, 2048))

  def test_explicit_pitch(self):
    self.assertEqual(qcom_image_layout(dtypes.half, (5, 16, 4), 384).row_pitch, 384)
    with self.assertRaises(ValueError): qcom_image_layout(dtypes.half, (5, 17, 4), 136)
    with self.assertRaises(ValueError): qcom_image_layout(dtypes.float, (5, 17, 4), 256)

  def test_invalid_dtype_and_shape(self):
    with self.assertRaises(ValueError): qcom_image_layout(dtypes.int, (1, 16, 4))
    with self.assertRaises(ValueError): qcom_image_layout(dtypes.float, (1, 16, 3))

  def test_qcom_selection_requires_contiguous_rgba_and_aligned_rows(self):
    arch = "a830,QCOM_IMAGE_PITCH_ALIGNMENT=16"
    self.assertTrue(image_valid_dims(dtypes.float, 4*16*7, arch))
    self.assertEqual(image_valid_dims(dtypes.float, 4*17, arch), [])
    self.assertEqual(image_valid_dims(dtypes.float, 4*16+1, arch), [])

  def test_fp16_odd_size_roundtrip_and_padding(self):
    values = np.arange(3*17*4, dtype=np.float16).reshape(3, 17, 4)
    layout, raw, read = linear_image_roundtrip(dtypes.half, values)
    for y in range(3):
      for x in range(17): np.testing.assert_array_equal(read(x, y), values[y, x])
      self.assertEqual(raw[y*layout.row_pitch+17*8:(y+1)*layout.row_pitch], bytes(layout.row_pitch-17*8))

  def test_fp32_xy_rgba_and_border(self):
    values = np.arange(5*7*4, dtype=np.float32).reshape(5, 7, 4)
    _layout, _raw, read = linear_image_roundtrip(dtypes.float, values)
    np.testing.assert_array_equal(read(6, 4), values[4, 6])
    np.testing.assert_array_equal(read(2, 3), values[3, 2])
    for xy in ((-1, 0), (0, -1), (7, 0), (0, 5)): np.testing.assert_array_equal(read(*xy), np.zeros(4, dtype=np.float32))

  def test_image_modes_transform_eligible_operands(self):
    ren = Renderer(Target(device="QCOM", arch="a830,QCOM_IMAGE_PITCH_ALIGNMENT=16"))
    buf, idx = UOp.param(1, dtypes.float, (1024,)), UOp.const(dtypes.int, 0)
    for image in (1, 2):
      with Context(IMAGE=image): self.assertIsNotNone(transform_to_image(({}, ren), buf, idx))

class TestQCOMImageDescriptors(unittest.TestCase):
  addr = 0x12345678000

  def test_gen6_gen7_sampled_fp16_fp32(self):
    for gen in (6, 7):
      for dtype, pitch in ((dtypes.half, 128), (dtypes.float, 256)):
        desc = qcom_image_descriptor(gen, dtype, (9, 16, 4), self.addr, row_pitch=pitch)
        self.assertEqual(len(desc), 16)
        self.assertEqual((desc[1] & 0x7fff, (desc[1] >> 15) & 0x7fff), (16, 9))
        self.assertEqual((desc[2] >> 7) & 0x3fffff, pitch)
        self.assertEqual((desc[2] >> 29) & 7, mesa.A6XX_TEX_2D)
        self.assertEqual((desc[4] | (desc[5] << 32)) & ((1 << 49)-1), self.addr)

  def test_gen6_gen7_storage(self):
    for gen in (6, 7):
      sampled = qcom_image_descriptor(gen, dtypes.float, (3, 16, 4), self.addr, row_pitch=256)
      storage = qcom_image_descriptor(gen, dtypes.float, (3, 16, 4), self.addr, storage=True, row_pitch=256)
      self.assertNotEqual(sampled[0], storage[0])
      self.assertEqual(sampled[1:], storage[1:])

  def test_gen8_sampled_fp16_fp32(self):
    for dtype, pitch, fmt, pitchalign in (
      (dtypes.half, 128, mesa.FMT6_16_16_16_16_FLOAT, 1),
      (dtypes.float, 256, mesa.FMT6_32_32_32_32_FLOAT, 2),
    ):
      desc = qcom_image_descriptor(8, dtype, (9, 16, 4), self.addr, row_pitch=pitch)
      self.assertEqual(len(desc), 16)
      self.assertEqual(((desc[1] & 0x1ffff) << 32) | desc[0], self.addr)
      self.assertEqual(((desc[1] >> 17) & 7, (desc[1] >> 20) & 0xfff), (mesa.A6XX_TEX_2D, 1))
      self.assertEqual((desc[2] & 0x7fff, (desc[2] >> 15) & 0x7fff), (16, 9))
      self.assertEqual(desc[3] & 0xff, fmt)
      self.assertEqual(tuple((desc[3] >> shift) & 7 for shift in (10, 13, 16, 19)), (3, 4, 5, 6))
      self.assertEqual((desc[6] & 0xffffff, (desc[6] >> 24) & 0xf), (pitch * 8, pitchalign))

  def test_gen8_storage_2d_matches_sampled_memobj(self):
    sampled = qcom_image_descriptor(8, dtypes.float, (5, 16, 4), self.addr, row_pitch=256)
    storage = qcom_image_descriptor(8, dtypes.float, (5, 16, 4), self.addr, storage=True, row_pitch=256)
    self.assertEqual(sampled, storage)

  def test_gen8_odd_dimensions_and_padded_pitch(self):
    desc = qcom_image_descriptor(8, dtypes.half, (11, 17, 4), self.addr)
    self.assertEqual((desc[2] & 0x7fff, (desc[2] >> 15) & 0x7fff), (17, 11))
    self.assertEqual(desc[6] & 0xffffff, 256 * 8)

  def test_alignment_rejected(self):
    with self.assertRaises(ValueError): qcom_image_descriptor(8, dtypes.half, (1, 16, 4), self.addr + 32, row_pitch=128)

class TestQCOMSamplers(unittest.TestCase):
  def test_gen6_gen7(self):
    for gen in (6, 7):
      desc = qcom_sampler_descriptor(gen)
      self.assertEqual(tuple((desc[0] >> shift) & 7 for shift in (5, 8, 11)), (3, 3, 3))
      self.assertTrue(desc[1] & (1 << 5))

  def test_gen8(self):
    desc = qcom_sampler_descriptor(8)
    self.assertEqual(tuple((desc[0] >> shift) & 7 for shift in (6, 9, 12)), (3, 3, 3))
    self.assertTrue(desc[1] & (1 << 31))
    self.assertEqual(desc[2] & 1, 1)  # fast zero border

if __name__ == "__main__": unittest.main()
