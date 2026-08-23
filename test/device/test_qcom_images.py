import ctypes, unittest
from types import SimpleNamespace
import numpy as np

from tinygrad import Tensor
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.late.coalesce import image_valid_dims, transform_to_image
from tinygrad.codegen.opt.postrange import apply_opts, Scheduler
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, Target, prod
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import QCOMArgsState, QCOMComputeQueue, qcom_image_descriptor, qcom_image_layout, qcom_sampler_descriptor
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface
from tinygrad.uop.ops import AxisType, Ops, UOp, sym_infer

class TestQCOMImageLayout(unittest.TestCase):
  def test_fp16_fp32_padding(self):
    h = qcom_image_layout(dtypes.half, (7, 17, 4))
    f = qcom_image_layout(dtypes.float, (3, 17, 4))
    self.assertEqual((h.row_pitch, h.array_pitch, h.pitch_alignment, h.size), (256, 2048, 128, 2048))
    self.assertEqual((f.row_pitch, f.array_pitch, f.pitch_alignment, f.size), (512, 2048, 256, 2048))

  def test_explicit_and_invalid_layouts(self):
    self.assertEqual(qcom_image_layout(dtypes.half, (5, 16, 4), 384).row_pitch, 384)
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
    a8, a6 = "a830,QCOM_IMAGE_PITCH_ALIGNMENT=16", "a630,IMAGE_PITCH_ALIGNMENT=64"
    self.assertTrue(image_valid_dims(dtypes.float, 4*16*7, a8))
    self.assertTrue(all(w % 16 == 0 for _,w in image_valid_dims(dtypes.float, 4096, a8)))
    self.assertTrue(all(w % 64 == 0 for _,w in image_valid_dims(dtypes.float, 4096, a6)))
    self.assertNotIn((5, 80), image_valid_dims(dtypes.float, 4*5*80, a8))
    self.assertEqual(image_valid_dims(dtypes.float, 4*17, a8), [])
    self.assertEqual(image_valid_dims(dtypes.float, 4*16+1, a8), [])

    ren, buf, idx = Renderer(Target(device="QCOM", arch=a8)), UOp.param(1, dtypes.float, (1024,)), UOp.const(0, dtypes.int)
    with Context(IMAGE=0): self.assertIsNone(transform_to_image(({}, ren, ()), buf, idx))
    with Context(IMAGE=2): self.assertIsNotNone(transform_to_image(({}, ren, ()), buf, idx))

class TestQCOMAutomaticImageLowering(unittest.TestCase):
  @staticmethod
  def conv_ast(channels=16):
    with Context(IMAGE=0):
      x = Tensor.empty(1, 8, 8, channels, device="CPU").permute(0, 3, 1, 2)
      w, residual = Tensor.empty(channels, channels, 3, 3, device="CPU"), Tensor.empty(1, channels, 8, 8, device="CPU")
      return (x.conv2d(w, padding=1) + residual).schedule_linear().src[-1].src[0]

  @staticmethod
  def renderer(arch): return Renderer(Target(device="QCOM", arch=arch))

  def test_selected_reduction_slot_survives_lowering(self):
    ren, ast = self.renderer("a830,QCOM_IMAGE_PITCH_ALIGNMENT=16"), self.conv_ast()
    with Context(IMAGE=1): lowered = full_rewrite_to_sink(ast, ren)
    self.assertEqual(lowered.arg.image_slots, (1,))
    params = {u.arg.slot:u for u in lowered.backward_slice if u.op is Ops.PARAM}
    self.assertEqual(params[1].max_shape[-1], 4)
    self.assertTrue(all(len(params[slot].max_shape) == 1 for slot in (0, 2, 3)))

  def test_selected_slot_only_and_forced_mode(self):
    ren = self.renderer("a830,QCOM_IMAGE_PITCH_ALIGNMENT=16")
    selected, other, idx = UOp.param(1, dtypes.float, (1024,)), UOp.param(2, dtypes.float, (1024,)), UOp.const(0, dtypes.int)
    with Context(IMAGE=1):
      self.assertIsNotNone(transform_to_image(({}, ren, (1,)), selected, idx))
      self.assertIsNone(transform_to_image(({}, ren, (1,)), other, idx))
    with Context(IMAGE=2): self.assertIsNotNone(transform_to_image(({}, ren, (1,)), other, idx))

  def test_upcast_geometry(self):
    ren, ast = self.renderer("a830,QCOM_IMAGE_PITCH_ALIGNMENT=16"), self.conv_ast(16)
    with Context(IMAGE=1): automatic = apply_opts(ast, ren)
    self.assertEqual(automatic.arg.image_slots, (1,))
    self.assertEqual(automatic.arg.applied_opts[0].arg, 16)
    sched = Scheduler(automatic, ren)
    self.assertLessEqual(prod(sched.full_shape[i] for i in sched.axes_of(AxisType.LOCAL)), 64)
    with Context(IMAGE=1): fallback = apply_opts(self.conv_ast(12), ren)
    self.assertEqual(fallback.arg.applied_opts[0].arg, 4)

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
    self.assertEqual(desc[7], 1)
    with self.assertRaises(ValueError): qcom_image_descriptor(8, dtypes.half, (1, 16, 4), self.addr + 32, row_pitch=128)
    with self.assertRaises(ValueError): qcom_image_descriptor(8, dtypes.float, (1, 0x7fff, 4), self.addr, row_pitch=1 << 21)

  def test_symbolic_address_rebind(self):
    addr = UOp.variable("image_addr", 0, 0xffffffffffff, dtype=dtypes.uint64)
    symbolic = qcom_image_descriptor(8, dtypes.float, (7, 16, 4), addr, row_pitch=256)
    concrete = qcom_image_descriptor(8, dtypes.float, (7, 16, 4), self.addr, row_pitch=256)
    self.assertEqual([sym_infer(x, {addr.expr:self.addr}) for x in symbolic], concrete)

    raw = (ctypes.c_ubyte * 0x900)()
    kaddr = ctypes.addressof(raw)
    kbuf = HCQBuffer(kaddr, len(raw), view=MMIOInterface(kaddr, len(raw), fmt='B'))
    prg = SimpleNamespace(dev=SimpleNamespace(gen=8), signature=(("data0", 0, dtypes.float, (7, 16, 4)),), NIR=True,
      ibo_cnt=0, tex_cnt=1, tex_to_image=[0], consts_info=[], samp_cnt=0, samplers=[], buf_off=0, buf_offs=[],
      tex_off=0x800, ibo_off=0x800, samp_off=0x840, kernargs_alloc_size=len(raw))
    state = QCOMArgsState(kbuf, prg, (HCQBuffer(addr, 7*16*16),))
    queue = QCOMComputeQueue(SimpleNamespace(gen=8))
    queue.bind_args_state(state)
    queue._apply_var_vals({addr.expr:self.addr})
    got = list(kbuf.cpu_view().view(offset=0x800, size=0x40, fmt='I'))
    self.assertEqual(got, concrete)

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
