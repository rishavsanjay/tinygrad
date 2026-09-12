import unittest
import numpy as np

from tinygrad.codegen.late.coalesce import image_valid_dims, transform_to_image
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, Target
from tinygrad.renderer import Renderer
from tinygrad.renderer.nir import _ir3_writable_images
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import qcom_image_descriptor, qcom_image_layout, qcom_sampler_descriptor
from tinygrad.runtime.ops_qcom import qcom_validate_image_counts
from tinygrad.uop.ops import UOp, sym_infer

class TestQCOMImageLayout(unittest.TestCase):
  def test_fp16_fp32_padding(self):
    h = qcom_image_layout(dtypes.half, (7, 17, 4))
    f = qcom_image_layout(dtypes.float, (3, 17, 4))
    self.assertEqual((h.row_pitch, h.array_pitch, h.pitch_alignment, h.size), (256, 4096, 128, 4096))
    self.assertEqual((f.row_pitch, f.array_pitch, f.pitch_alignment, f.size), (512, 4096, 256, 4096))

  def test_explicit_and_invalid_layouts(self):
    self.assertEqual(qcom_image_layout(dtypes.half, (5, 16, 4), 384).row_pitch, 384)
    self.assertEqual(qcom_image_layout(dtypes.float, (1, 252, 4), 4032).pitchalign, 0)
    for dtype, shape, pitch in ((dtypes.half, (5, 17, 4), 136), (dtypes.float, (5, 17, 4), 256)):
      with self.assertRaises(ValueError): qcom_image_layout(dtype, shape, pitch)
    for dtype, shape in ((dtypes.int, (1, 16, 4)), (dtypes.float, (1, 16, 3)), (dtypes.float, (0, 16, 4))):
      with self.assertRaises(ValueError): qcom_image_layout(dtype, shape)

  def test_odd_dimensions_and_padding(self):
    values = np.arange(3*17*4, dtype=np.float16).reshape(3, 17, 4)
    layout = qcom_image_layout(dtypes.half, values.shape)
    raw = bytearray(layout.size)
    row_bytes = values.shape[1] * 4 * values.dtype.itemsize
    for y in range(values.shape[0]): raw[y*layout.row_pitch:y*layout.row_pitch+row_bytes] = values[y].tobytes()
    for y in range(3):
      got = np.frombuffer(raw, dtype=values.dtype, count=17*4, offset=y*layout.row_pitch).reshape(17, 4)
      np.testing.assert_array_equal(got, values[y])
      self.assertEqual(raw[y*layout.row_pitch+row_bytes:(y+1)*layout.row_pitch], bytes(layout.row_pitch-row_bytes))

  def test_qcom_eligibility_and_modes(self):
    a8, a6 = "a830,IMAGE_PITCH_ALIGNMENT=16", "a630,IMAGE_PITCH_ALIGNMENT=64"
    self.assertTrue(image_valid_dims(dtypes.float, 4*16*7, a8))
    self.assertTrue(all(w % 16 == 0 for _,w in image_valid_dims(dtypes.float, 4096, a8)))
    self.assertTrue(all(w % 64 == 0 for _,w in image_valid_dims(dtypes.float, 4096, a6)))
    self.assertEqual(image_valid_dims(dtypes.float, 4*17, a8), [(1, 17)])
    self.assertEqual(image_valid_dims(dtypes.float, 4*16+1, a8), [])

    ren, buf, idx = Renderer(Target(device="QCOM", arch=a8)), UOp.param(1, dtypes.float, (1024,)), UOp.const(0, dtypes.int)
    with Context(IMAGE=0): self.assertIsNone(transform_to_image(({}, ren), buf, idx))
    with Context(IMAGE=2): self.assertIsNotNone(transform_to_image(({}, ren), buf, idx))

class TestQCOMImageDescriptors(unittest.TestCase):
  addr = 0x12345678000

  def test_gen6_gen7_sampled_and_storage(self):
    for gen in (6, 7):
      for dtype, pitch in ((dtypes.half, 128), (dtypes.float, 256)):
        sampled = qcom_image_descriptor(gen, dtype, (9, 16, 4), self.addr, row_pitch=pitch)
        storage = qcom_image_descriptor(gen, dtype, (9, 16, 4), self.addr, storage=True, row_pitch=pitch)
        self.assertEqual(len(sampled), 16)
        self.assertEqual((sampled[1] & 0x7fff, (sampled[1] >> 15) & 0x7fff), (16, 9))
        self.assertEqual(((sampled[2] >> 7) & 0x3fffff, (sampled[2] >> 29) & 7), (pitch, mesa.A6XX_TEX_2D))
        self.assertNotEqual(sampled[0], storage[0])

  def test_gen8_fp16_fp32_sampled_and_storage(self):
    for dtype, pitch, fmt, pitchalign in ((dtypes.half, 128, mesa.FMT6_16_16_16_16_FLOAT, 1),
                                          (dtypes.float, 256, mesa.FMT6_32_32_32_32_FLOAT, 2)):
      sampled = qcom_image_descriptor(8, dtype, (9, 16, 4), self.addr, row_pitch=pitch)
      self.assertEqual(sampled, qcom_image_descriptor(8, dtype, (9, 16, 4), self.addr, storage=True, row_pitch=pitch))
      self.assertEqual(((sampled[1] & 0x1ffff) << 32) | sampled[0], self.addr)
      self.assertEqual((sampled[2] & 0x7fff, (sampled[2] >> 15) & 0x7fff), (16, 9))
      self.assertEqual((sampled[3] & 0xff, tuple((sampled[3] >> x) & 7 for x in (10, 13, 16, 19))), (fmt, (3, 4, 5, 6)))
      self.assertEqual((sampled[6] & 0xffffff, (sampled[6] >> 24) & 0xf), (pitch * 8, pitchalign))

  def test_gen8_odd_shape_and_alignment(self):
    desc = qcom_image_descriptor(8, dtypes.half, (17, 17, 4), self.addr)
    self.assertEqual(desc[6] & 0xffffff, 256 * 8)
    self.assertEqual(desc[7], 2)
    with self.assertRaises(ValueError): qcom_image_descriptor(8, dtypes.half, (1, 16, 4), self.addr + 32, row_pitch=128)
    with self.assertRaises(ValueError): qcom_image_descriptor(8, dtypes.float, (1, 0x7fff, 4), self.addr, row_pitch=1 << 21)

  def test_symbolic_address_rebind(self):
    addr = UOp.variable("image_addr", 0, 0xffffffffffff, dtype=dtypes.uint64)
    symbolic = qcom_image_descriptor(8, dtypes.float, (7, 16, 4), addr, row_pitch=256)
    concrete = qcom_image_descriptor(8, dtypes.float, (7, 16, 4), self.addr, row_pitch=256)
    self.assertEqual([sym_infer(x, {addr.expr:self.addr}) for x in symbolic], concrete)

  def test_image_count_validation(self):
    for sampled,total in ((0, 0), (1, 1), (4, 7), (31, 32)): qcom_validate_image_counts(sampled, total)
    for sampled,total in ((2, 1), (32, 32), (0, 33), (-1, 1)):
      with self.assertRaises(RuntimeError): qcom_validate_image_counts(sampled, total)

class TestIR3ImageAccess(unittest.TestCase):
  def test_same_image_load_store_is_classified_writable(self):
    img = UOp.param(0, dtypes.float, (5, 16, 4))
    idx = img.index(UOp.const(0, dtypes.int), UOp.const(0, dtypes.int))
    value = UOp.const(1.0, dtypes.float).stack(UOp.const(2.0, dtypes.float), UOp.const(3.0, dtypes.float), UOp.const(4.0, dtypes.float))
    load, store = idx.load(), idx.store(value)
    self.assertEqual(_ir3_writable_images([img, idx, load, store]), {img})
    self.assertEqual(_ir3_writable_images([img, idx, load]), set())

class TestQCOMSamplers(unittest.TestCase):
  def test_nearest_unnormalized_zero_border(self):
    for gen in (6, 7):
      desc = qcom_sampler_descriptor(gen)
      self.assertEqual(tuple((desc[0] >> shift) & 7 for shift in (5, 8, 11)), (3, 3, 3))
      self.assertTrue(desc[1] & (1 << 5))
    desc = qcom_sampler_descriptor(8)
    self.assertEqual(tuple((desc[0] >> shift) & 7 for shift in (6, 9, 12)), (3, 3, 3))
    self.assertTrue(desc[1] & (1 << 31))
    self.assertEqual(desc[2] & 1, 1)

if __name__ == "__main__": unittest.main()
