import os, sys, unittest
import numpy as np
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Buffer, TinyELF
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context, IMAGE
from tinygrad.uop.ops import UOp, KernelInfo, Ops, AxisType
from tinygrad.runtime.support.compiler_adreno import AdrenoCompiler
from tinygrad.runtime.support.hcq import HCQBuffer

@unittest.skipUnless(os.getenv('DEV','').split(':')[0]=='ADRENO','requires physical A830 native backend')
class TestAdrenoExecution(unittest.TestCase):
  def test_spills_and_long_shader(self):
    dev=Device['ADRENO']
    out=UOp.param(0,dtypes.float,shape=(200,))
    acc=UOp.placeholder((200,),dtypes.float,-1,AddrSpace.REG)
    three=UOp.const(3.0)
    values=UOp(Ops.STACK,src=(three,)*200)
    stored=acc.store(values)
    loaded=acc.load()
    src=dev.renderer.render([out,acc,three,values,stored,loaded,out.store(loaded)])
    lib=dev.renderer.compiler.compile(src)
    resource,_,_=AdrenoCompiler.unpack(lib)
    self.assertGreater(resource.pvtmem_size,0)
    self.assertGreater(resource.instrlen,dev.dev_info.props.instr_cache_size)
    obj=TinyELF(lib,'native_spill',dev.renderer.target,((None,0,dtypes.float,(200,)),))
    dst=Buffer('ADRENO',200,dtypes.float).allocate()
    dev.runtime(obj)(dst._buf,wait=True)
    np.testing.assert_array_equal(dst.numpy(),np.full(200,3,dtype=np.float32))

  def test_shared_memory_barrier(self):
    dev=Device['ADRENO']
    out=UOp.param(0,dtypes.int,shape=(128,))
    shared=UOp.placeholder((128,),dtypes.int,-1,AddrSpace.LOCAL)
    x=UOp.special(128,'lidx0')
    other=x^127
    first=shared.index(x).store(x.cast(dtypes.int))
    barrier=UOp(Ops.BARRIER,src=(first,))
    read=shared.after(barrier).index(other).load()
    write=out.index(x).store(read)
    ast=UOp.sink(write,arg=KernelInfo(name='native_shared')).rtag('manual')
    program=to_program(ast,dev.renderer)
    resources,_,_=AdrenoCompiler.unpack(program.to_elf().lib)
    self.assertGreaterEqual(resources.shared_size,512)
    dst=Buffer('ADRENO',128,dtypes.int).allocate()
    dev.runtime(program.to_elf())(dst._buf,global_size=(1,1,1),local_size=(128,1,1),wait=True)
    np.testing.assert_array_equal(dst.numpy(),np.arange(128,dtype=np.int32)^127)

  def test_register_slice_and_cast_index(self):
    dev=Device['ADRENO']
    out=UOp.param(0,dtypes.float,shape=(4,))
    acc=UOp.placeholder((8,),dtypes.float,-1,AddrSpace.REG)
    values=[UOp.const(float(i)) for i in range(8)]
    packed=UOp(Ops.STACK,src=tuple(values))
    stored=acc.store(packed)
    view=UOp(Ops.SHRINK,src=(acc,UOp.const(3).cast(dtypes.int),UOp.const(4).cast(dtypes.int)))
    loaded=view.load()
    src=dev.renderer.render([out,acc,*values,packed,stored,view.src[1].src[0],view.src[1],view.src[2].src[0],view.src[2],view,
                             loaded,out.store(loaded)])
    lib=dev.renderer.compiler.compile(src)
    obj=TinyELF(lib,'native_register_slice',dev.renderer.target,((None,0,dtypes.float,(4,)),))
    dst=Buffer('ADRENO',4,dtypes.float).allocate()
    dev.runtime(obj)(dst._buf,wait=True)
    np.testing.assert_array_equal(dst.numpy(),np.arange(3,7,dtype=np.float32))

  def test_interleaved_scalar_and_aliased_buffers(self):
    dev=Device['ADRENO']
    out=UOp.param(0,dtypes.float,shape=(1,),name='out')
    scalar=UOp.param(5,dtypes.uint,name='scale',addrspace=AddrSpace.ALU)
    a=UOp.param(1,dtypes.float,shape=(1,),name='a')
    b=UOp.param(1,dtypes.float,shape=(1,),name='alias')
    av,bv,sv=a.load(),b.load(),scalar.cast(dtypes.float)
    total=av+bv
    result=total+sv
    code=dev.renderer.render([out,scalar,a,b,av,bv,sv,total,result,out.store(result)])
    obj=TinyELF(dev.renderer.compiler.compile(code),'native_mixed_abi',dev.renderer.target,
                (('out',0,dtypes.float,(1,)),('a',1,dtypes.float,(1,)),('alias',1,dtypes.float,(1,)),('scale',2,dtypes.uint,())))
    dst=Buffer('ADRENO',1,dtypes.float).allocate()
    src=Buffer('ADRENO',1,dtypes.float).allocate()
    dev.allocator._copyin(src._buf,memoryview(np.array([1.25],dtype=np.float32)).cast('B'))
    dev.runtime(obj)(dst._buf,src._buf,vals=(7,),wait=True)
    np.testing.assert_array_equal(dst.numpy(),[9.5])
    np.testing.assert_array_equal(src.numpy(),[1.25])

  def test_large_signed_pointer_offsets(self):
    dev=Device['ADRENO']
    dst=Buffer('ADRENO',1,dtypes.float).allocate()
    src=Buffer('ADRENO',1,dtypes.float).allocate()
    dev.allocator._copyin(src._buf,memoryview(np.array([1.25],dtype=np.float32)).cast('B'))
    for offset in (2**33+7,-2**32-3):
      with self.subTest(offset=offset):
        out,inp=UOp.param(0,dtypes.float,shape=(1,)),UOp.param(1,dtypes.float,shape=(1,))
        raw=UOp.const(offset)
        index=raw.cast(dtypes.long)
        pointer=inp.index(index)
        value=pointer.load()
        code=dev.renderer.render([out,inp,raw,index,pointer,value,out.store(value)])
        lib=dev.renderer.compiler.compile(code)
        obj=TinyELF(lib,'native_large_offset',dev.renderer.target,((None,0,dtypes.float,(1,)),(None,1,dtypes.float,(1,))))
        # The synthetic base plus the large signed byte offset resolves to the
        # same valid allocation; no giant allocation or out-of-bounds access.
        base=HCQBuffer(src._buf.va_addr-offset*4,4,_base=src._buf,owner=src._buf.owner)
        dev.runtime(obj)(dst._buf,base,wait=True)
        np.testing.assert_array_equal(dst.numpy(),[1.25])
    np.testing.assert_array_equal(src.numpy(),[1.25])

  def test_workgroup_and_local_ids(self):
    out=UOp.param(0,dtypes.int,shape=(96,))
    g,x,y=UOp.special(4,'gidx0'),UOp.special(8,'lidx0'),UOp.special(3,'lidx1')
    i=(g*3+y)*8+x
    ast=UOp.sink(out.index(i).store(i.cast(dtypes.int)),arg=KernelInfo(name='native_ids')).rtag('manual')
    dev=Device['ADRENO']
    program=to_program(ast,dev.renderer)
    dst=Buffer('ADRENO',96,dtypes.int).allocate()
    dev.runtime(program.to_elf())(dst._buf,global_size=(4,1,1),local_size=(8,3,1),wait=True)
    np.testing.assert_array_equal(dst.numpy(),np.arange(96,dtype=np.int32))

  def test_affine_jit_and_input_integrity(self):
    @TinyJit
    def run(x): return (x*3+2).realize()
    for phase in range(22):
      inp=np.arange(257,dtype=np.float32)+phase
      x=Tensor(inp,device='ADRENO').realize()
      np.testing.assert_array_equal(run(x).numpy(),inp*3+2)
      np.testing.assert_array_equal(x.numpy(),inp)

  def test_integer_mul_and_shifts(self):
    inp=np.array([0,1,-1,65537,0x1234567,-0x1234567],dtype=np.int32)
    x=Tensor(inp,device='ADRENO')
    np.testing.assert_array_equal((x*65537).numpy(),inp*np.int32(65537))
    np.testing.assert_array_equal((x>>7).numpy(),inp>>7)
    np.testing.assert_array_equal((x^12345).numpy(),inp^12345)

  def test_storage_widths_and_half_rounding(self):
    for dt in (dtypes.int8,dtypes.uint8,dtypes.int16,dtypes.uint16,dtypes.int64,dtypes.uint64):
      with self.subTest(dtype=dt):
        x=Tensor([0,1,7,15],device='ADRENO',dtype=dt)
        self.assertEqual((x*2+1).tolist(),[1,3,15,31])
    x=Tensor([2**33+7,2**34+3],device='ADRENO',dtype=dtypes.int64)
    self.assertEqual((x+5).tolist(),[2**33+12,2**34+8])
    a=np.array([-0.3333,0.1,1.125,11.3],dtype=np.float16)
    x=Tensor(a,device='ADRENO')
    expected=(a*np.float16(3)).astype(np.float16)+np.float16(0.5)
    np.testing.assert_array_equal((x*3+0.5).numpy(),expected)
    np.testing.assert_array_equal(x.bitcast(dtypes.uint16).numpy(),a.view(np.uint16))

  def test_integer_division_and_remainder(self):
    for dt,nums,divs in ((np.int32,[-2147483648,-100,-17,-1,0,17,100,2147483647],[3,-7,5,-2,3,-5,7,31]),
                        (np.uint32,[0,17,2**31,2**32-1,2**32-1],[3,5,2**31-1,2**31+1,2**32-1])):
      a,b=np.array(nums,dtype=dt),np.array(divs,dtype=dt)
      x,y=Tensor(a,device='ADRENO'),Tensor(b,device='ADRENO')
      np.testing.assert_array_equal((x//y).numpy(),a//b)
      np.testing.assert_array_equal((x%y).numpy(),a%b)

  def test_comparison_where_and_cast(self):
    inp=np.array([-3,-0.5,0,1.5,7],dtype=np.float32)
    x=Tensor(inp,device='ADRENO')
    np.testing.assert_array_equal((x<0).numpy(),inp<0)
    np.testing.assert_array_equal((x<0).where(x,-x).numpy(),np.where(inp<0,inp,-inp))
    np.testing.assert_array_equal(x.cast(dtypes.int).numpy(),inp.astype(np.int32))

  def test_masked_view(self):
    inp=np.arange(15,dtype=np.float32).reshape(3,5)
    x=Tensor(inp,device='ADRENO')
    np.testing.assert_array_equal(x.pad(((1,2),(2,1))).contiguous().numpy(),np.pad(inp,((1,2),(2,1))))
    np.testing.assert_array_equal((x.T+2).numpy(),inp.T+2)

  def test_reduction_and_matmul(self):
    a=np.arange(35,dtype=np.float32).reshape(5,7)/16
    b=np.arange(21,dtype=np.float32).reshape(7,3)/8
    x,y=Tensor(a,device='ADRENO'),Tensor(b,device='ADRENO')
    np.testing.assert_allclose(x.sum(axis=1).numpy(),a.sum(axis=1),rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose((x@y).numpy(),a@b,rtol=1e-6,atol=1e-6)

  def test_softmax_and_gradient(self):
    a=np.array([[-1,0,1],[2,0,-2]],dtype=np.float32)
    x=Tensor(a,device='ADRENO')
    loss=(x*x).sum()
    loss.backward()
    np.testing.assert_allclose(x.grad.numpy(),2*a,rtol=1e-6,atol=1e-6)
    exp=np.exp(a-a.max(axis=1,keepdims=True))
    np.testing.assert_allclose(x.softmax(axis=1).numpy(),exp/exp.sum(axis=1,keepdims=True),rtol=1e-5,atol=1e-6)

  def test_symbolic_jit(self):
    @TinyJit
    def run(x): return (x+1).sum().contiguous().realize()
    host=np.arange(30,dtype=np.float32).reshape(3,10)
    x=Tensor(host,device='ADRENO').realize()
    for size in (1,2,3,7,10,4):
      np.testing.assert_allclose(run(x[:,:Variable('n',1,10).bind(size)]).item(),(host[:,:size]+1).sum(),rtol=1e-6)
    np.testing.assert_array_equal(x.numpy(),host)

  def test_convolution_normalization_and_attention(self):
    a=np.arange(25,dtype=np.float32).reshape(1,1,5,5)/16
    w=np.array([[[[1,-2],[3,0.5]]]],dtype=np.float32)
    expected=np.empty((1,1,4,4),dtype=np.float32)
    for y in range(4):
      for x in range(4): expected[0,0,y,x]=(a[0,0,y:y+2,x:x+2]*w[0,0]).sum()
    np.testing.assert_allclose(Tensor(a,device='ADRENO').conv2d(Tensor(w,device='ADRENO')).numpy(),expected,rtol=1e-5,atol=1e-5)
    q=np.array([[1,2,3],[-1,0,1]],dtype=np.float32)/4
    x=Tensor(q,device='ADRENO')
    normalized=(q-q.mean(-1,keepdims=True))/np.sqrt(q.var(-1,keepdims=True)+1e-5)
    np.testing.assert_allclose(x.layernorm().numpy(),normalized,rtol=1e-5,atol=1e-5)
    scores=q@q.T/np.sqrt(np.float32(3))
    weights=np.exp(scores-scores.max(-1,keepdims=True))
    weights/=weights.sum(-1,keepdims=True)
    np.testing.assert_allclose(((x@x.T/np.sqrt(3)).softmax()@x).numpy(),weights@q,rtol=1e-5,atol=1e-5)

  def test_no_mesa_dependency(self):
    Device['ADRENO'].synchronize()
    self.assertFalse([m for m in sys.modules if m in ('tinygrad.renderer.nir','tinygrad.runtime.support.compiler_mesa',
                                                    'tinygrad.runtime.autogen.mesa')])

@unittest.skipUnless(os.getenv('DEV','').split(':')[0]=='ADRENO' and IMAGE.value==2, 'requires A830 with IMAGE=2')
class TestAdrenoImages(unittest.TestCase):
  def test_fp32_fp16_image_arithmetic_and_buffer_bias(self):
    for dt,npdt,tol in ((dtypes.float,np.float32,1e-6),(dtypes.half,np.float16,3e-3)):
      with self.subTest(dtype=dt):
        a=(np.arange(7*16*4,dtype=np.float32).reshape(7,16,4)/17).astype(npdt)
        x=Tensor(a,device='ADRENO',dtype=dt).contiguous().realize()
        bias=Tensor([1.5],device='ADRENO',dtype=dtypes.float).realize()
        out=((x+2).contiguous().realize()*bias).contiguous().numpy()
        np.testing.assert_allclose(out,(a+2)*1.5,rtol=tol,atol=tol)
        np.testing.assert_array_equal(x.numpy(),a)

  def test_interleaved_images_and_coherent_update(self):
    a=np.arange(5*16*4,dtype=np.float32).reshape(5,16,4)/11
    b=np.flip(a,axis=1).copy()+0.5
    def dual(out_add,in_a,out_mul,in_b):
      y,x,c=UOp.range(5,0),UOp.range(16,1),UOp.range(4,2,AxisType.UPCAST)
      add=out_add[y,x,c].store(in_a[y,x,c]+in_b[y,x,c])
      mul=out_mul[y,x,c].store(in_a[y,x,c]*in_b[y,x,c])
      return UOp.group(add,mul).end(y,x,c).sink(arg=KernelInfo(name='native_image_dual'))
    out_add,in_a,out_mul,in_b=(Tensor.empty(*a.shape,device='ADRENO'),Tensor(a,device='ADRENO').contiguous().realize(),
                               Tensor.empty(*a.shape,device='ADRENO'),Tensor(b,device='ADRENO').contiguous().realize())
    result=Tensor.custom_kernel(out_add,in_a,out_mul,in_b,fxn=dual)
    Tensor.realize(result[0],result[2])
    np.testing.assert_allclose(result[0].numpy(),a+b,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(result[2].numpy(),a*b,rtol=1e-6,atol=1e-6)
    def update(image):
      y,x,c=UOp.range(5,0),UOp.range(16,1),UOp.range(4,2,AxisType.UPCAST)
      return image[y,x,c].store(image[y,x,c]*2+1).end(y,x,c).sink(arg=KernelInfo(name='native_image_update'))
    same=Tensor.custom_kernel(Tensor(a,device='ADRENO').contiguous().realize(),fxn=update)[0]
    np.testing.assert_allclose(same.numpy(),a*2+1,rtol=1e-6,atol=1e-6)

  def test_jit_image_rebinding_and_alignment(self):
    @TinyJit
    def transform(x): return ((x+1).contiguous().realize()*2).contiguous().realize()
    addresses=[]
    for val in range(1,11,2):
      x=Tensor.full((7,16,4),val,device='ADRENO').contiguous().realize()
      addresses.append(int(x.uop.buffer._buf.va_addr))
      np.testing.assert_array_equal(transform(x).numpy(),np.full((7,16,4),(val+1)*2,dtype=np.float32))
    self.assertGreater(len(set(addresses)),1)
    base=Buffer('ADRENO',1040,dtypes.float).allocate()
    aligned=base.view(5*16*4,dtypes.float,64).allocate()
    img=Tensor(UOp.from_buffer(aligned).reshape((5,16,4)))
    with Context(IMAGE=0): img.assign(Tensor.ones(5,16,4,device='ADRENO')).realize()
    np.testing.assert_array_equal((img+1).contiguous().numpy(),np.full((5,16,4),2,dtype=np.float32))
    bad=Tensor(UOp.from_buffer(base.view(5*16*4,dtypes.float,16).allocate()).reshape((5,16,4)))
    with self.assertRaisesRegex(ValueError,'unaligned QCOM image address'): (bad+1).contiguous().realize()

  def test_asymmetric_padding_zero_border(self):
    x=np.arange(1*1*4*4,dtype=np.float32).reshape(1,1,4,4)/7
    w=np.array([[[[1,-2],[3,0.5]]]],dtype=np.float32)
    for pad in ((0,1,0,1),(2,1,2,1),(2,0,2,1)):
      with self.subTest(pad=pad):
        left,right,top,bottom=pad
        padded=np.pad(x,((0,0),(0,0),(top,bottom),(left,right)))
        expected=np.empty((1,1,padded.shape[2]-1,padded.shape[3]-1),dtype=np.float32)
        for y in range(expected.shape[2]):
          for xidx in range(expected.shape[3]): expected[0,0,y,xidx]=(padded[0,0,y:y+2,xidx:xidx+2]*w[0,0]).sum()
        out=Tensor(x,device='ADRENO').conv2d(Tensor(w,device='ADRENO'),padding=pad).numpy()
        np.testing.assert_allclose(out,expected,rtol=1e-5,atol=1e-5)

if __name__=='__main__': unittest.main()
