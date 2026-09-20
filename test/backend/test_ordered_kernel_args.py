import numpy as np
import pickle, unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Tensor, Context, dtypes
from tinygrad.device import Device, Buffer, ProgramArg, TinyELF
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.codegen import to_program
from tinygrad.engine.realize import run_linear

class TestOrderedKernelArgs(unittest.TestCase):
  def _program(self, out_slot:int, in_slot:int, variable_slots:tuple[int, ...]):
    output_uop = UOp.param(out_slot, dtypes.float32, shape=(4,))
    input_uop = UOp.param(in_slot, dtypes.float32, shape=(4,))
    variables = []
    for index, slot in enumerate(variable_slots):
      variable = UOp.variable(f"var_{index}", 0, 42, dtypes.int32, param=True)
      variables.append(variable.replace(arg=replace(variable.arg, slot=slot)))
    v_range = UOp.range(4, 0)
    value = input_uop.index(v_range).load() * variables[0].cast(dtypes.float32)
    for variable in variables[1:]: value = value + variable.cast(dtypes.float32)
    value = output_uop.index(v_range).store(value).end(v_range)
    return to_program(value.sink(arg=KernelInfo(name="ordered_kernel_args")), Device[Device.DEFAULT].renderer)

  def _test_template(self, out_slot:int, in_slot:int, variable_slots:tuple[int, ...], values:tuple[int, ...], expected_order):
    program = self._program(out_slot, in_slot, variable_slots)
    self.assertEqual(tuple((arg.addrspace, arg.slot) for arg in program.to_elf().signature), expected_order)
    self.assertEqual(program.arg.globals, tuple(sorted((in_slot, out_slot))))
    self.assertEqual(tuple(var.arg.slot for var in program.arg.vars), variable_slots)

    input_data = np.arange(4, dtype=np.float32)
    output_buf = Buffer(Device.DEFAULT, 4, dtypes.float32).allocate()
    input_buf = Buffer(Device.DEFAULT, 4, dtypes.float32, initial_value=input_data.tobytes())
    slot_to_buffer = {in_slot: input_buf, out_slot: output_buf}
    bufs = [UOp.from_buffer(slot_to_buffer.get(s, output_buf)) for s in range(max(program.arg.globals)+1)]
    run_linear(UOp(Ops.LINEAR, src=(program.call(*bufs),)), dict(zip((v.expr for v in program.arg.vars), values)), wait=True)
    np.testing.assert_equal(np.frombuffer(output_buf.as_memoryview(), dtype=np.float32), input_data * values[0] + sum(values[1:]))

  def test_interleaved_params(self):
    cases = [
      (0, 1, (2,), (3,), ((AddrSpace.GLOBAL,0),(AddrSpace.GLOBAL,1),(AddrSpace.ALU,0))),
      (1, 2, (0,), (3,), ((AddrSpace.ALU,0),(AddrSpace.GLOBAL,0),(AddrSpace.GLOBAL,1))),
      (0, 2, (1, 3), (4, 2), ((AddrSpace.GLOBAL,0),(AddrSpace.ALU,0),(AddrSpace.GLOBAL,1),(AddrSpace.ALU,1))),
    ]
    for case in cases:
      with self.subTest(case=case): self._test_template(*case)

  def test_params_beyond_abi_registers(self):
    buffer_uops = [UOp.param(i, dtypes.float32, shape=(4,)) for i in range(8)]
    v_range = UOp.range(4, 0)
    value = buffer_uops[1].index(v_range).load()
    for buffer_uop in buffer_uops[2:]: value = value + buffer_uop.index(v_range).load()
    value = buffer_uops[0].index(v_range).store(value).end(v_range)
    program = to_program(value.sink(arg=KernelInfo(name="params_beyond_abi_registers")), Device[Device.DEFAULT].renderer)
    input_data = [np.full(4, i, dtype=np.float32) for i in range(1, 8)]
    buffers = [Buffer(Device.DEFAULT, 4, dtypes.float32).allocate()]
    buffers += [Buffer(Device.DEFAULT, 4, dtypes.float32, initial_value=data.tobytes()) for data in input_data]
    run_linear(UOp(Ops.LINEAR, src=(program.call(*[UOp.from_buffer(b) for b in buffers]),)), wait=True)
    np.testing.assert_equal(np.frombuffer(buffers[0].as_memoryview(), dtype=np.float32), sum(input_data))

  @unittest.skipUnless(Device.DEFAULT == "METAL", "Metal graph support required")
  def test_metal_graph_rebinds_mixed_args(self):
    from tinygrad.runtime.graph.metal import MetalGraph
    program = self._program(0, 2, (1, 3))
    inputs = [UOp.placeholder((4,), dtypes.float32, i, device="METAL") for i in range(2)]
    unused = UOp.new_buffer("METAL", 4, dtypes.float32)
    linear = UOp(Ops.LINEAR, src=(program.call(inputs[0], unused, inputs[1]),))
    graph = None
    for scale, bias in ((3, 1), (7, 2)):
      data = np.arange(4, dtype=np.float32) + bias
      out = Buffer("METAL", 4, dtypes.float32).allocate()
      inp = Buffer("METAL", 4, dtypes.float32, initial_value=data.tobytes())
      bufs = tuple(UOp.from_buffer(b) for b in (out, inp))
      if graph is None: graph = MetalGraph(UOp(Ops.CUSTOM_FUNCTION, src=(linear,), arg="graph"), bufs)
      graph(bufs, {"var_0": scale, "var_1": bias}, wait=True)
      np.testing.assert_equal(np.frombuffer(out.as_memoryview(), dtype=np.float32), data * scale + bias)

  def test_program_info_stores_value_descriptors(self):
    info = self._program(0, 2, (1,)).arg
    self.assertTrue(all(type(arg) is ProgramArg for arg in info.args))
    self.assertEqual(pickle.loads(pickle.dumps(info)).args, info.args)

  def test_merge_args_reuses_shared_buffer_slot(self):
    sig = (ProgramArg(None,0,dtypes.float32,(4,),AddrSpace.GLOBAL), ProgramArg("v",0,dtypes.int32,(),AddrSpace.ALU),
           ProgramArg(None,0,dtypes.float32,(1,1,4),AddrSpace.GLOBAL), ProgramArg(None,1,dtypes.float32,(4,),AddrSpace.GLOBAL))
    self.assertEqual(TinyELF.merge_args(sig, [0x1111,0x2222], [7]), [0x1111,7,0x1111,0x2222])

  def test_iter_sig_alignment(self):
    sig = (ProgramArg("a",0,dtypes.int32,(),AddrSpace.ALU), ProgramArg(None,0,dtypes.float32,(4,),AddrSpace.GLOBAL),
           ProgramArg("b",1,dtypes.int16,(),AddrSpace.ALU), ProgramArg(None,1,dtypes.float32,(4,),AddrSpace.GLOBAL))
    self.assertEqual([off for off,_ in TinyELF.iter_sig(sig)], [0,8,16,24])
    self.assertEqual(TinyELF.packed_size(sig), 32)
    sig = (ProgramArg("a",0,dtypes.int16,(),AddrSpace.ALU), ProgramArg("b",1,dtypes.int64,(),AddrSpace.ALU))
    self.assertEqual([off for off,_ in TinyELF.iter_sig(sig)], [0,8])
    self.assertEqual(TinyELF.packed_size(sig), 16)

  def test_lvp_packed_size_matches_descriptor(self):
    from tinygrad.runtime.ops_cpu import lvp_pack_args
    sig = (ProgramArg(None,0,dtypes.float32,(4,),AddrSpace.GLOBAL), ProgramArg("v",0,dtypes.int32,(),AddrSpace.ALU))
    packed = lvp_pack_args(sig, [0x1111,7])
    self.assertEqual(len(packed), 24)
    self.assertEqual(np.frombuffer(packed, dtype=np.uint32, count=1, offset=8)[0], 3)
    narrow = (ProgramArg("a",0,dtypes.int16,(),AddrSpace.ALU), ProgramArg("b",1,dtypes.int16,(),AddrSpace.ALU))
    packed = lvp_pack_args(narrow, [2,3])
    self.assertEqual(len(packed), 16)
    self.assertEqual(np.frombuffer(packed, dtype=np.uint32, count=1, offset=8)[0], 1)

  def _sparse_program_and_call(self, device:str):
    program = self._program(0, 2, (1,))
    call_bufs = [UOp.placeholder((4,), dtypes.float32, i, device=device) for i in range(3)]
    call_var = UOp.variable("var_0", 0, 42, dtypes.int32)
    return program, call_bufs, program.call(*call_bufs, call_var.bind(7))

  def test_qcom_kernargs_mixed_order(self):
    program, call_bufs, call = self._sparse_program_and_call("QCOM")
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue
    fake_data = SimpleNamespace(signature=program.to_elf().signature, ibo_cnt=0, tex_cnt=0, tex_to_image=[], NIR=False,
      consts_info=[], samplers=[], samp_off=0, buf_offs=[0,8,16], tex_off=0, ibo_off=0, kernargs_alloc_size=64, wgsz=0xfc)
    captured = []
    with patch("tinygrad.runtime.ops_qcom.pack_args", side_effect=lambda xs,size: captured.extend(xs) or []):
      QCOMComputeQueue.kernargs(SimpleNamespace(devs=("QCOM",)), call, program, fake_data)
    self.assertEqual(captured, [(0,call_bufs[0].getaddr(("QCOM",))), (8,UOp.const(7).ccast(dtypes.int32)),
                                (16,call_bufs[2].getaddr(("QCOM",)))])

  def test_dsp_rpc_wrapper_uses_compact_buffer_ordinals(self):
    from tinygrad.runtime.ops_dsp import DSPRenderer
    renderer = object.__new__(DSPRenderer)
    b0, b2 = UOp.param(0,dtypes.float32,shape=(4,)), UOp.param(2,dtypes.float32,shape=(4,))
    v1 = UOp.variable("dsp_v",0,42,dtypes.int32,param=True).replace(arg=replace(UOp.variable("dsp_v",0,42,dtypes.int32,param=True).arg, slot=1))
    entry = renderer._render_entry("dsp_mixed", [("b0",(b0,False)), ("v1",(v1,False)), ("b2",(b2,False))])
    self.assertIn("off0 = ((int*)pra[1].buf.pv)[0]", entry)
    self.assertIn("off2 = ((int*)pra[1].buf.pv)[1]", entry)
    self.assertIn("pra[3].dma.fd", entry)
    self.assertIn("pra[4].dma.fd", entry)
    self.assertNotIn("pra[5].dma.fd", entry)
    self.assertIn("dsp_mixed(buf_0, sz_or_val_1, buf_2)", entry)

  def test_aliases_reuse_buffer_slot(self):
    out, alias = UOp.param(0, dtypes.float32, shape=(4,)), UOp.param(0, dtypes.float32, shape=(8,))
    variable = UOp.variable("scale", 0, 42, dtypes.int32, param=True)
    variable = variable.replace(arg=replace(variable.arg, slot=1))
    r = UOp.range(4, 0)
    sink = out.index(r).store(alias.index(r+4).load() * variable.cast(dtypes.float32)).end(r).sink(arg=KernelInfo(name="alias_args"))
    data = np.arange(8, dtype=np.float32)
    buf = Buffer(Device.DEFAULT, 8, dtypes.float32, initial_value=data.tobytes())
    run_linear(UOp(Ops.LINEAR, src=(sink.call(UOp.from_buffer(buf)),)), {"scale": 3}, wait=True)
    np.testing.assert_equal(np.frombuffer(buf.as_memoryview(), dtype=np.float32), np.concatenate((data[4:]*3, data[4:])))

  def test_export_preserves_mixed_argument_order(self):
    from extra.export_model import compile_net
    program, _, call = self._sparse_program_and_call("CPU")
    _, statements, _, _ = compile_net(UOp(Ops.LINEAR, src=(call,)), [])
    self.assertEqual(statements[0][1], ["input0", program.arg.vars[0], "input2"])

  @unittest.skipUnless(Device.DEFAULT == "CL", "images need CL")
  def test_image_and_flat_view_of_one_buffer(self):
    x, weight = Tensor.rand(16,10,27).realize(), Tensor.rand(10).realize()
    target = Tensor.uniform(16,27,low=0,high=10,dtype=dtypes.int32).realize()
    x_np, target_np, weight_np = x.numpy(), target.numpy(), weight.numpy()
    picked = np.take_along_axis(x_np, target_np[:,None,:], axis=1).squeeze(1)
    expected = -(picked * weight_np[target_np]).sum() / weight_np[target_np].sum()
    with Context(IMAGE=1): loss = x.nll_loss(target, weight=weight).numpy()
    np.testing.assert_allclose(loss, expected, atol=1e-5, rtol=1e-5)

if __name__ == "__main__": unittest.main()
