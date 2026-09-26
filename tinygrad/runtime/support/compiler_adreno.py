from __future__ import annotations
import hashlib, json, struct, pathlib
from dataclasses import asdict
from typing import Any
from tinygrad.device import Compiler
from tinygrad.helpers import round_up
from tinygrad.helpers import is_image_shape
from tinygrad.runtime.support.ir3 import IR3Shader

# Reference notices: extra/qcom_gpu_driver/ADRENO_REFERENCE_LICENSE.
# Field definitions: Mesa 26.2.1 src/freedreno/isa/ir3-cat{0,1,2,3,4,6}.xml.
# This encoder is Python code; no Mesa functions or compiled shader templates are used.
_source_root = pathlib.Path(__file__).resolve().parents[2]
_build_sources = ('runtime/support/compiler_adreno.py', 'renderer/adreno.py', 'runtime/autogen/adreno.py', 'runtime/ops_adreno.py',
                  'codegen/__init__.py', 'codegen/late/coalesce.py', 'codegen/decomp/dtype.py', 'codegen/decomp/transcendental.py')
BUILD = 'tinygrad-adreno-a830-v1_' + hashlib.sha256(b''.join((_source_root/path).read_bytes() for path in _build_sources)).hexdigest()
ARCH = 'a830,chip_id=0x44050001'
ARCH_IMAGE = 'a830,QCOM_IMAGE_PITCH_ALIGNMENT=16,chip_id=0x44050001'
TYPES = {'f16':0, 'f32':1, 'u16':2, 'u32':3, 's16':4, 's32':5, 'u8':6}
CAT2 = {'add.f':0, 'min.f':1, 'max.f':2, 'mul.f':3, 'cmps.f':5, 'absneg.f':6, 'floor.f':9, 'ceil.f':10,
        'trunc.f':13, 'add.u':16, 'add.s':17, 'sub.u':18, 'sub.s':19, 'cmps.u':20, 'cmps.s':21,
        'min.u':22, 'min.s':23, 'max.u':24, 'max.s':25, 'absneg.s':26, 'and.b':28, 'or.b':29,
        'not.b':30, 'xor.b':31, 'mull.u':50, 'shl.b':54, 'shr.b':55, 'ashr.b':56}
CAT3 = {'madsh.m16':3, 'mad.u24':4, 'mad.f32':7, 'sel.b32':9}
CAT4 = {'rcp':0, 'rsq':1, 'log2':2, 'exp2':3, 'sqrt':6}

def field(value:int, shift:int, width:int, signed=False) -> int:
  if type(value) is not int or not (-(1 << (width-1)) if signed else 0) <= value < 1 << (width-int(signed)):
    raise ValueError(f'Adreno field {value!r} does not fit {width} bits')
  return (value & ((1 << width)-1)) << shift

def operand(src, cat:int=2) -> int:
  kind, value = src
  if kind == 'r': return field(value, 0, 8)
  if kind == 'c': return field(value, 0, 11) | (1 << 12)
  if kind == 'i' and cat == 2: return field(value, 0, 11, signed=True) | (1 << 13)
  raise ValueError(f'unsupported Adreno source {src}')

def encode(ins:dict[str, Any]) -> int:
  op = ins['op']
  flags = (int(ins.get('ss', False)) << 44) | (int(ins.get('sy', False)) << 60) | (int(ins.get('jp', False)) << 59)
  if op in ('nop', 'end', 'jump', 'br', 'predt', 'prede'):
    opc = {'nop':0, 'end':6, 'jump':2, 'br':1, 'predt':13, 'prede':15}[op]
    return flags | (opc << 55) | ((1 << 49) if op in ('predt', 'prede') else 0) | \
      field(ins.get('offset', 0), 0, 32, signed=True) | (int(ins.get('inv', False)) << 52) | field(ins.get('repeat',0), 40, 3)
  if op == 'mov':
    kind, val = ins['src']
    mode = {'r':0, 'c':1, 'i':2}[kind]
    width = {'r':8, 'c':11, 'i':32}[kind]
    return flags | (1 << 61) | field(ins['dst'],32,8) | (TYPES[ins.get('dst_type','u32')] << 46) | \
      (TYPES[ins.get('src_type','u32')] << 50) | (mode << 53) | field(ins.get('round',0),55,2) | field(val,0,width)
  if op in CAT2:
    return flags | (2 << 61) | field(ins['dst'],32,8) | (CAT2[op] << 53) | (int(ins.get('full',True)) << 52) | \
      field(ins.get('cond',0),48,3) | operand(ins['src1']) | (operand(ins.get('src2',('r',0))) << 16)
  if op in CAT3:
    # cat3 src2 is a GPR; src1/src3 can also address the constant file.
    if ins['src2'][0] != 'r': raise ValueError('Adreno cat3 source 2 must be a register')
    return flags | (3 << 61) | (CAT3[op] << 55) | field(ins['dst'],32,8) | operand(ins['src1'],3) | \
      field(ins['src2'][1],47,8) | (operand(ins['src3'],3) << 16)
  if op in CAT4:
    return flags | (4 << 61) | (CAT4[op] << 53) | (1 << 52) | field(ins['dst'],32,8) | operand(ins['src'])
  if op in ('ldib','stib'):
    if ins.get('type','f32') != 'f32' or ins.get('size',4) != 4 or not 0 <= ins['uav'] < 32 or \
       not 0 <= ins['data'] <= 252 or not 0 <= ins['coord'] <= 254:
      raise ValueError('native Adreno image instructions require typed f32 RGBA and a valid UAV')
    # Mesa 26.2.1 ir3-cat6.xml: typed 2D, four components, immediate UAV.
    # A830 IR3 witnesses: ldib=0xc02200050361ba00, stib=0xc022000903677a00.
    return flags | (6 << 61) | (2 << 52) | (TYPES['f32'] << 49) | field(ins['uav'],41,8) | \
      field(ins['data'],32,8) | field(ins['coord'],24,8) | (6 << 20) | ({'ldib':6,'stib':29}[op] << 14) | \
      (3 << 12) | (1 << 11) | (1 << 9)
  if op in ('ldg','ldp','ldl'):
    if ins.get('ss',False): raise ValueError('Adreno memory instructions do not encode SS')
    return flags | (6 << 61) | ({'ldg':0,'ldl':1,'ldp':2}[op] << 54) | (TYPES[ins.get('type','u32')] << 49) | field(ins['dst'],32,8) | \
      field(ins['addr'],14,8) | field(ins.get('size',1),24,3) | (1 << 23) | 1 | field(ins.get('offset',0),1,13,signed=True)
  if op in ('stg','stp','stl'):
    if ins.get('ss',False): raise ValueError('Adreno memory instructions do not encode SS')
    offset = ins.get('offset',0)
    field(offset,0,13,signed=True)
    return flags | (6 << 61) | ({'stg':3,'stl':4,'stp':5}[op] << 54) | (TYPES[ins.get('type','u32')] << 49) | field(ins['addr'],41,8) | \
      (1 << 40) | field(offset & 255,32,8) | field((offset >> 8) & 31,9,5) | field(ins['src'],1,8) | \
      field(ins.get('size',1),24,3) | (1 << 23)
  if op in ('bar','fence'):
    return flags | (7 << 61) | (1 << 49) | (int(op=='fence') << 55) | \
      (int(ins.get('global',False)) << 54) | (int(ins.get('local',False)) << 53) | \
      (int(ins.get('read',False)) << 52) | (int(ins.get('write',False)) << 51)
  raise ValueError(f'unsupported Adreno instruction {op!r}')

def assemble(instructions:list[dict[str, Any]]) -> bytes:
  labels, pc = {}, 0
  for ins in instructions:
    if ins['op'] == 'label':
      if ins['name'] in labels: raise ValueError('duplicate Adreno label')
      labels[ins['name']] = pc
    else: pc += 1
  result, pc = [], 0
  targets = {labels[ins['target']] for ins in instructions if 'target' in ins}
  for ins in instructions:
    if ins['op'] == 'label': continue
    ins = dict(ins)
    if 'target' in ins: ins['offset'] = labels[ins['target']] - pc
    if pc in targets: ins['jp'] = True
    result.append(encode(ins))
    pc += 1
  if not result or result[-1] >> 61 != 0 or (result[-1] >> 55) & 15 != 6: raise ValueError('Adreno shader must terminate with end')
  return struct.pack(f'<{len(result)}Q', *result).ljust(round_up(len(result)*8,128), b'\0')

def disassemble_word(word:int) -> str:
  def reg(n): return f'r{n//4}.{ "xyzw"[n%4] }'
  def signed(n,bits): return n-(1<<bits) if n&(1<<(bits-1)) else n
  def src(n):
    if n&0x2000: return str(signed(n&0x7ff,11))
    if n&0x1000: return f'c{(n&0x7ff)//4}.{ "xyzw"[n%4] }'
    if n&0x0800: raise ValueError('relative register decoding is not supported')
    return reg(n&255)
  cat,dst = word>>61,(word>>32)&255
  flags = ('(sy)' if word&(1<<60) else '')+('(ss)' if cat!=6 and word&(1<<44) else '')+('(jp)' if word&(1<<59) else '')
  if cat==0:
    opc=(word>>55)&15
    names={0:'nop',1:'br',2:'jump',6:'end',13:'predt',15:'prede'}
    if opc not in names: raise ValueError('unknown native Adreno flow instruction')
    if opc in (1,2): return flags+names[opc]+(' !p0.x,' if opc==1 and word&(1<<52) else ' p0.x,' if opc==1 else '')+f' #{signed(word&0xffffffff,32)}'
    return flags+(f'(rpt{(word>>40)&7})' if (word>>40)&7 else '')+names[opc]
  if cat==1:
    mode=(word>>53)&3
    types={v:k for k,v in TYPES.items()}
    it,ot=types[(word>>50)&7],types[(word>>46)&7]
    val = reg(word&255) if mode==0 else f'c{(word&2047)//4}.{ "xyzw"[word%4] }' if mode==1 else hex(word&0xffffffff)
    rounding=('','(even)','(pos_infinity)','(neg_infinity)')[(word>>55)&3]
    return flags+f'mov.{it}{ot} {rounding}{reg(dst)}, {val}'
  if cat==2:
    op={v:k for k,v in CAT2.items()}[(word>>53)&63]
    cond='.'+('lt','le','gt','ge','eq','ne')[(word>>48)&7] if op.startswith('cmps.') else ''
    unary=op in ('absneg.f','absneg.s','floor.f','ceil.f','trunc.f','not.b')
    return flags+op+cond+f' {reg(dst)}, {src(word&0xffff)}'+('' if unary else f', {src((word>>16)&0xffff)}')
  if cat==3:
    op={v:k for k,v in CAT3.items()}[(word>>55)&15]
    return flags+f'{op} {reg(dst)}, {src(word&0x1fff)}, {reg((word>>47)&255)}, {src((word>>16)&0x1fff)}'
  if cat==4:
    return flags+f'{ {v:k for k,v in CAT4.items()}[(word>>53)&63] } {reg(dst)}, {src(word&0xffff)}'
  if cat==6:
    if (word>>52)&3==2 and (word>>14)&63 in (6,29):
      op='ldib' if (word>>14)&63==6 else 'stib'
      return flags+f'{op}.b.typed.2d.f32.4.imm {reg((word>>32)&255)}, {reg((word>>24)&255)}, {(word>>41)&255}'
    typ={v:k for k,v in TYPES.items()}[(word>>49)&7]
    size=(word>>24)&7
    if (word>>54)&31 in (0,1,2):
      offset=signed((word>>1)&8191,13)
      op,space={0:('ldg','g'),1:('ldl','l'),2:('ldp','p')}[(word>>54)&31]
      return flags+f'{op}.{typ} {reg(dst)}, {space}[{reg((word>>14)&255)}{offset:+d}], {size}'
    if (word>>54)&31 in (3,4,5):
      offset=signed(dst|(((word>>9)&31)<<8),13)
      op,space={3:('stg','g'),4:('stl','l'),5:('stp','p')}[(word>>54)&31]
      return flags+f'{op}.{typ} {space}[{reg((word>>41)&255)}{offset:+d}], {reg((word>>1)&255)}, {size}'
  if cat==7 and word&(1<<49) and (word>>55)&15 in (0,1):
    return flags+('fence' if (word>>55)&15 else 'bar')+''.join('.'+name for bit,name in ((54,'g'),(53,'l'),(52,'r'),(51,'w')) if word&(1<<bit))
  raise ValueError(f'unsupported native Adreno instruction word {word:#018x}')

class AdrenoCompiler(Compiler):
  def __init__(self, arch:str=ARCH):
    if arch not in (ARCH,ARCH_IMAGE): raise ValueError(f'unsupported native Adreno target {arch!r}')
    self.arch = arch
    super().__init__(f'compile_{BUILD}_{arch}')

  def compile(self, src:str) -> bytes:
    source = json.loads(src)
    binary = assemble(source['instructions'])
    resources = IR3Shader(self.arch, BUILD, source['branchstack'], source['private_size'], source.get('shared_size',0),
                          False, False, True, len(binary)//128, False,
                          source['constlen'], 0xfc, 192, 0, 16, 0, source.get('num_uavs',0), (), source['fregs'], 0, False,
                          tuple(tuple(p) for p in source['params']), tuple(source['writes']))
    resources.validate()
    meta = json.dumps({'resources':asdict(resources), 'signature':source['signature']}, sort_keys=True, separators=(',',':')).encode()
    payload = meta + binary
    return struct.pack('<4sII', b'ADN\x01', len(meta), len(binary)) + hashlib.sha256(payload).digest() + payload

  @staticmethod
  def unpack(data:bytes):
    if len(data) < 44: raise ValueError('truncated native Adreno artifact')
    magic, nmeta, nbin = struct.unpack_from('<4sII',data)
    if magic != b'ADN\x01' or not 0 < nmeta <= 65536 or not nbin or nbin % 128 or len(data) != 44+nmeta+nbin:
      raise ValueError('invalid native Adreno artifact header')
    if hashlib.sha256(data[44:]).digest() != data[12:44]: raise ValueError('corrupt native Adreno artifact')
    try:
      meta = json.loads(data[44:44+nmeta])
      fields = meta['resources']
      for key in ('params','tex_to_image','writes'):
        fields[key] = tuple(tuple(p) if isinstance(p,list) else p for p in fields[key])
      resources = IR3Shader(**fields)
      resources.validate()
      if resources.arch not in (ARCH,ARCH_IMAGE) or resources.build != BUILD or resources.instrlen != nbin//128:
        raise ValueError('native Adreno artifact target/build/length mismatch')
      if resources.pvtmem_size > 4096 or resources.shared_size > 32768 or resources.num_uavs > 32:
        raise ValueError('native Adreno artifact requests unsupported resources')
      if not 3 <= resources.fregs <= 48 or resources.hregs or resources.wgid != 192 or resources.lid != 0 or resources.buf_off != 16:
        raise ValueError('native Adreno artifact register/argument layout mismatch')
      if resources.pvtmem_per_wave or resources.early_preamble or not resources.mergedregs or resources.double_threadsize or \
         resources.round_robin_mode or resources.wgsz!=0xfc or resources.imm_off or resources.tex_to_image:
        raise ValueError('native Adreno artifact execution mode mismatch')
      if not 1<=resources.constlen<=64 or resources.buf_off+max((offset+size for _,offset,size in resources.params),default=0)>resources.constlen*16:
        raise ValueError('native Adreno artifact constants do not cover arguments')
      signature = meta['signature']
      if not isinstance(signature,list) or len(signature)!=len(resources.params)+resources.num_uavs:
        raise ValueError('native Adreno artifact ABI length mismatch')
      for arg in signature:
        if not isinstance(arg,list) or len(arg)!=4 or (arg[0] is not None and not isinstance(arg[0],str)) or \
           type(arg[1]) is not int or arg[1]<0 or not isinstance(arg[2],str) or not isinstance(arg[3],list) or \
           any(type(dim) is not int or dim<0 for dim in arg[3]): raise ValueError('native Adreno artifact malformed signature')
      image_args=[arg for arg in signature if is_image_shape(tuple(arg[3]))]
      if len(image_args)!=resources.num_uavs or any(arg[2] not in ('float','half') for arg in image_args):
        raise ValueError('native Adreno artifact image signature mismatch')
      if sorted(arg[1] for arg in signature if not is_image_shape(tuple(arg[3])))!=sorted(slot for slot,_,_ in resources.params):
        raise ValueError('native Adreno artifact argument slots mismatch')
      return resources, signature, data[44+nmeta:]
    except (KeyError, TypeError, UnicodeError) as e: raise ValueError('invalid native Adreno metadata') from e

  def disassemble(self, lib:bytes):
    _,_,binary=self.unpack(lib)
    for pc,(word,) in enumerate(struct.iter_unpack('<Q',binary)):
      print(f'{pc:04d}: {word>>32:08x}_{word&0xffffffff:08x} {disassemble_word(word)}')
