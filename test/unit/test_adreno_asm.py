import json, struct, unittest
from tinygrad.codegen import to_program
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.helpers import Target
from tinygrad.renderer.adreno import AdrenoRenderer
from tinygrad.runtime.support.compiler_adreno import ARCH, AdrenoCompiler, assemble, encode, field, disassemble_word
from tinygrad.uop.ops import AxisType, KernelInfo, UOp, Ops

class TestAdrenoEncoding(unittest.TestCase):
  def test_independent_machine_words(self):
    # Mesa 26.2.1 ir3/tests/disasm.c: words observed from vendor shaders, not
    # roundtrips through this assembler. These layouts are valid on A830.
    cases = [
      ({'op':'end'}, 0x0300000000000000),
      ({'op':'br','offset':-4}, 0x00800000fffffffc),
      ({'op':'br','offset':3,'inv':True}, 0x0090000000000003),
      ({'op':'mov','dst':0,'src':('c',32),'src_type':'f32','dst_type':'f32'}, 0x2024400000000020),
      ({'op':'mov','dst':5,'src':('i',0xffff),'src_type':'s16','dst_type':'s16'}, 0x205100050000ffff),
      ({'op':'mad.u24','dst':9,'src1':('c',0),'src2':('r',10),'src3':('r',9)}, 0x6205000900091000),
      ({'op':'rcp','dst':10,'src':('r',3)}, 0x8010000a00000003),
      ({'op':'ldg','dst':6,'addr':6,'type':'u32'}, 0xc006000601818001),
      ({'op':'ldg','dst':3,'addr':3,'type':'u32','offset':308}, 0xc00600030180c269),
      ({'op':'stg','addr':2,'src':33,'type':'s32','size':3,'offset':305}, 0xc0ca053103800242),
      ({'op':'stl','addr':2,'src':4,'type':'u32'}, 0xc106050001800008),
      ({'op':'ldl','dst':1,'addr':1,'type':'u32'}, 0xc046000101804001),
      ({'op':'bar','global':True,'sy':True}, 0xf042000000000000),
      ({'op':'ldp','dst':24,'addr':8,'type':'u32','size':3}, 0xc086001803820001),
      ({'op':'stp','addr':45,'src':1,'type':'f32','offset':-176}, 0xc1425b5001803e02),
    ]
    for instruction, word in cases:
      with self.subTest(instruction=instruction): self.assertEqual(encode(instruction),word)

  def test_field_bounds(self):
    self.assertEqual(field(-4096,1,13,True),0x2000)
    for val,width,signed in ((4096,13,True),(-4097,13,True),(256,8,False),(-1,8,False)):
      with self.subTest(val=val), self.assertRaises(ValueError): field(val,0,width,signed)
    with self.assertRaises(ValueError): encode({'op':'unknown'})

  def test_disassembly(self):
    self.assertEqual(disassemble_word(0x2024400000000020),'mov.f32f32 r0.x, c8.x')
    self.assertEqual(disassemble_word(0x00800000fffffffc),'br p0.x, #-4')
    self.assertEqual(disassemble_word(0xc00600030180c269),'ldg.u32 r0.w, g[r0.w+308], 1')
    with self.assertRaises(ValueError): disassemble_word(0xe000000000000000)

  def test_branch_resolution(self):
    binary=assemble([{'op':'label','name':'again'},{'op':'nop'}, {'op':'jump','target':'again'},{'op':'end'}])
    self.assertEqual(len(binary),128)
    self.assertEqual(struct.unpack_from('<Q',binary,8)[0],0x01000000ffffffff)
    self.assertTrue(struct.unpack_from('<Q',binary)[0] & (1<<59))
    with self.assertRaises(ValueError): assemble([{'op':'nop'}])
    with self.assertRaises(ValueError): assemble([{'op':'max.u','dst':0,'src1':('r',0),'src2':('r',1)}])

class TestAdrenoCompiler(unittest.TestCase):
  def program(self):
    dst,src=UOp.param(0,dtypes.float,shape=(257,)),UOp.param(1,dtypes.float,shape=(257,))
    i=UOp.range(257,0,AxisType.GLOBAL)
    return to_program(UOp.sink(dst.index(i).store(src.index(i)*3+2).end(i),arg=KernelInfo(name='affine')),
                      AdrenoRenderer(Target('ADRENO',arch=ARCH)))

  def test_independent_lowering_and_artifact(self):
    p=self.program()
    resources,signature,binary=AdrenoCompiler.unpack(p.to_elf().lib)
    self.assertEqual(resources.arch,ARCH)
    self.assertEqual(resources.params,((0,0,8),(1,8,8)))
    self.assertEqual(signature,[[None,0,'float',[257]],[None,1,'float',[257]]])
    ops={ins['op'] for ins in json.loads(p.src[2].arg)['instructions']}
    self.assertTrue({'ldg','stg','mul.f','add.f'} <= ops)
    self.assertEqual(len(binary),resources.instrlen*128)
    # Deterministic encoding and ABI; no serialized C pointers.
    self.assertEqual(AdrenoCompiler().compile(p.src[2].arg),p.to_elf().lib)

  def test_shared_index_decomposition(self):
    out=UOp.param(0,dtypes.int,shape=(128,))
    shared=UOp.placeholder((128,),dtypes.int,-1,AddrSpace.LOCAL)
    i=UOp.special(128,'lidx0')
    barrier=UOp(Ops.BARRIER,src=(shared.index(i).store(i.cast(dtypes.int)),))
    load=shared.after(barrier).index(i^127).load()
    p=to_program(UOp.sink(out.index(i).store(load),arg=KernelInfo(name='shared')).rtag('manual'),
                 AdrenoRenderer(Target('ADRENO',arch=ARCH)))
    resources,_,_=AdrenoCompiler.unpack(p.to_elf().lib)
    self.assertEqual(resources.shared_size,512)
    ops={ins['op'] for ins in json.loads(p.src[2].arg)['instructions']}
    self.assertTrue({'ldl','stl','bar','fence','xor.b'}<=ops)

  def test_mixed_width_compare_and_cast(self):
    src=UOp.param(1,dtypes.ulong,shape=(4,))
    out=UOp.param(0,dtypes.bool,shape=(4,))
    i=UOp.range(4,0,AxisType.GLOBAL)
    expr=src.index(i).load().ne(UOp.cconst(0,dtypes.uint))
    p=to_program(UOp.sink(out.index(i).store(expr).end(i),arg=KernelInfo(name='wide_compare')),
                 AdrenoRenderer(Target('ADRENO',arch=ARCH)))
    AdrenoCompiler.unpack(p.to_elf().lib)
    float_out=UOp.param(0,dtypes.float,shape=(4,))
    p=to_program(UOp.sink(float_out.index(i).store(src.index(i).load().cast(dtypes.float)).end(i),arg=KernelInfo(name='wide_to_float')),
                 AdrenoRenderer(Target('ADRENO',arch=ARCH)))
    AdrenoCompiler.unpack(p.to_elf().lib)

  def test_sine_expands_before_wide_dtype_lowering(self):
    src=UOp.param(1,dtypes.float,shape=(6,))
    out=UOp.param(0,dtypes.float,shape=(6,))
    i=UOp.range(6,0,AxisType.GLOBAL)
    p=to_program(UOp.sink(out.index(i).store(src.index(i).load().alu(Ops.SIN)).end(i),arg=KernelInfo(name='large_sine')),
                 AdrenoRenderer(Target('ADRENO',arch=ARCH)))
    AdrenoCompiler.unpack(p.to_elf().lib)

  def test_interleaved_and_aliased_argument_layout(self):
    out=UOp.param(0,dtypes.float,shape=(1,),name='out')
    scalar=UOp.param(5,dtypes.uint,name='scale',addrspace=AddrSpace.ALU)
    a=UOp.param(1,dtypes.float,shape=(1,),name='a')
    b=UOp.param(1,dtypes.float,shape=(1,),name='alias')
    renderer=AdrenoRenderer(Target('ADRENO',arch=ARCH))
    src=renderer.render([out,scalar,a,b])
    resources,signature,_=AdrenoCompiler.unpack(renderer.compiler.compile(src))
    self.assertEqual(resources.params,((0,0,8),(2,8,4),(1,16,8),(1,24,8)))
    self.assertEqual(signature,[['out',0,'float',[1]],['a',1,'float',[1]],['alias',1,'float',[1]],['scale',2,dtypes.uint.name,[]]])

  def test_corrupt_artifacts(self):
    lib=self.program().to_elf().lib
    for data in (lib[:10],lib[:-1],b'IR3\x01'+lib[4:],lib[:60]+bytes([lib[60]^1])+lib[61:]):
      with self.subTest(size=len(data)),self.assertRaises(ValueError): AdrenoCompiler.unpack(data)
    with self.assertRaises(ValueError): AdrenoCompiler('a840')

if __name__=='__main__': unittest.main()
