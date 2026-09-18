import numpy as np
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Tensor, Context, dtypes
from tinygrad.device import Device, Buffer, TinyELF
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime

class TestCommutativeParams(unittest.TestCase):
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
    return to_program(value.sink(arg=KernelInfo(name="commutative_params")), Device[Device.DEFAULT].renderer)

  def _test_template(self, out_slot:int, in_slot:int, variable_slots:tuple[int, ...], values:tuple[int, ...]):
    program = self._program(out_slot, in_slot, variable_slots)
    render_order = {u:i for i,u in enumerate(program.src[1].src) if u.op is Ops.PARAM}
    params = sorted(program.arg.parameters, key=lambda u: (u.arg.slot, render_order.get(u, 0)))
    gmap = {s:i for i,s in enumerate(program.arg.globals)}
    vmap = {v.arg.slot:i for i,v in enumerate(program.arg.vars)}
    expected_sig = tuple((u.arg.name, vmap[u.arg.slot] if u.addrspace is AddrSpace.ALU else gmap[u.arg.slot],
                          u.dtype, u._shape, u.addrspace) for u in params)
    self.assertEqual(program.to_elf().signature, expected_sig)
    self.assertEqual(program.arg.globals, tuple(sorted((in_slot, out_slot))))
    self.assertEqual(tuple(var.arg.slot for var in program.arg.vars), variable_slots)

    input_data = np.arange(4, dtype=np.float32)
    output_buf = Buffer(Device.DEFAULT, 4, dtypes.float32).allocate()
    input_buf = Buffer(Device.DEFAULT, 4, dtypes.float32, initial_value=input_data.tobytes())
    global_size, local_size = program.arg.launch_dims({})
    slot_to_buffer = {in_slot: input_buf, out_slot: output_buf}
    get_runtime(Device.DEFAULT, program)(*[slot_to_buffer[s]._buf for s in program.arg.globals],
      global_size=global_size, local_size=local_size, vals=values, wait=True)
    np.testing.assert_equal(np.frombuffer(output_buf.as_memoryview(), dtype=np.float32), input_data * values[0] + sum(values[1:]))

  def test_commutative_params(self):
    cases = [
      (0, 1, (2,), (3,)),       # B,B,S
      (1, 0, (2,), (2,)),       # B,B,S, construction reversed
      (1, 2, (0,), (3,)),       # S,B,B
      (2, 1, (0,), (3,)),       # S,B,B, construction reversed
      (0, 2, (1, 3), (4, 2)),   # B,S,B,S, sparse globals (0,2)
      (3, 2, (0, 1), (2, 4)),   # S,S,B,B
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
    global_size, local_size = program.arg.launch_dims({})
    get_runtime(Device.DEFAULT, program)(*[b._buf for b in buffers], global_size=global_size, local_size=local_size, wait=True)
    np.testing.assert_equal(np.frombuffer(buffers[0].as_memoryview(), dtype=np.float32), sum(input_data))

  def test_merge_args_reuses_shared_buffer_slot(self):
    sig = ((None,0,dtypes.float32,(4,),AddrSpace.GLOBAL), ("v",0,dtypes.int32,(),AddrSpace.ALU),
           (None,0,dtypes.float32,(1,1,4),AddrSpace.GLOBAL), (None,1,dtypes.float32,(4,),AddrSpace.GLOBAL))
    self.assertEqual(TinyELF.merge_args(sig, [0x1111,0x2222], [7]), [0x1111,7,0x1111,0x2222])

  def test_iter_sig_mixed_alignment(self):
    sig = (("a",0,dtypes.int32,(),AddrSpace.ALU), (None,0,dtypes.float32,(4,),AddrSpace.GLOBAL),
           ("b",1,dtypes.int16,(),AddrSpace.ALU), (None,1,dtypes.float32,(4,),AddrSpace.GLOBAL))
    self.assertEqual(list(TinyELF.iter_sig(sig)), [(0,dtypes.int32,AddrSpace.ALU), (8,dtypes.float32,AddrSpace.GLOBAL),
                                                   (16,dtypes.int16,AddrSpace.ALU), (24,dtypes.float32,AddrSpace.GLOBAL)])

  def _sparse_program_and_call(self, device:str):
    program = self._program(0, 2, (1,))
    call_bufs = [UOp.placeholder((4,), dtypes.float32, i, device=device) for i in range(3)]
    call_var = UOp.variable("var_0", 0, 42, dtypes.int32)
    return program, call_bufs, program.call(*call_bufs, call_var.bind(7))

  def test_amd_kernargs_mixed_order(self):
    program, call_bufs, call = self._sparse_program_and_call("AMD")
    from tinygrad.runtime.ops_amd import AMDComputeQueue
    fake_data = SimpleNamespace(kernargs_segment_size=128, enable_dispatch_ptr=0)
    with patch("tinygrad.runtime.ops_amd.pack_args", side_effect=lambda xs,size:list(xs)):
      packed = AMDComputeQueue.kernargs(SimpleNamespace(devs=("AMD",)), call, program, fake_data)
    self.assertEqual(packed, [(0,call_bufs[0].getaddr(("AMD",))), (8,UOp.const(7).ccast(dtypes.int32)),
                              (16,call_bufs[2].getaddr(("AMD",)))])

  def test_cuda_hcq2_mixed_order(self):
    program, call_bufs, call = self._sparse_program_and_call("CUDA")
    from tinygrad.runtime.ops_cuda import CUDAQueue
    captured = []
    fake = SimpleNamespace(devs=("CUDA",), extern=lambda tag: UOp.const(0,dtypes.uint64),
                           launch=lambda func,gs,ls,args: captured.extend(args))
    CUDAQueue.exec(fake, call, program)
    self.assertEqual(captured, [call_bufs[0].getaddr(("CUDA",)), UOp.const(7).ccast(dtypes.int32), call_bufs[2].getaddr(("CUDA",))])

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
