from __future__ import annotations
import json, struct
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Target, round_up, getenv, is_image_shape, IMAGE
from tinygrad.renderer import Renderer
from tinygrad.runtime.support.compiler_adreno import AdrenoCompiler, ARCH
from tinygrad.uop.ops import Ops, UOp, GroupOp, UPat, PatternMatcher
from tinygrad.codegen.decomp.dtype import l2i
from tinygrad.codegen.decomp.transcendental import get_transcendental_patterns

def lower_wide_index(x:UOp):
  if not any(s.dtype in dtypes.int64s for s in x.src): return None
  wide_dt = next(s.dtype for s in x.src if s.dtype in dtypes.int64s)
  dt = dtypes.int if wide_dt==dtypes.long else dtypes.uint
  words:list[UOp] = []
  for i,s in enumerate(x.src):
    # Shifts take a single-word count and WHERE takes a single-word predicate.
    # Every other binary operand must contribute both words, even when an
    # optimization leaves a 32-bit constant opposite a 64-bit index.
    if s.dtype in dtypes.int64s:
      view = s.bitcast(dt)
      words.extend(view.index(UOp.cconst(lane,dtypes.int)) for lane in (0,1))
    elif x.op not in (Ops.SHL,Ops.SHR,Ops.WHERE) or x.op is Ops.WHERE and i>0:
      # Build the extension as 32-bit UOps so the final rewrite never sees an
      # unsupported 64-to-32 cast introduced by this renderer's own rule.
      low = s.cast(dt)
      high = (s < s.const_like(0)).where(UOp.cconst(-1,dt),UOp.cconst(0,dt)) if s.dtype in dtypes.ints else UOp.cconst(0,dt)
      words.extend((low,high))
    else: words.append(s)
  result = l2i(x.op,dt,*words)
  return UOp(Ops.STACK,src=result).bitcast(x.dtype) if isinstance(result,tuple) else result

def lower_wide_cast(x:UOp):
  if x.src[0].dtype in dtypes.int64s:
    words=x.src[0].bitcast(dtypes.int if x.src[0].dtype==dtypes.long else dtypes.uint)
    lo,hi=(words.index(UOp.cconst(i,dtypes.int)) for i in (0,1))
    if x.dtype in (dtypes.int,dtypes.uint): return lo.cast(x.dtype)
    if x.dtype in dtypes.floats: return l2i(Ops.CAST,x.dtype,lo,hi)
  return None

class AdrenoRenderer(Renderer):
  supports_float4 = False
  has_local = True
  has_shared = True
  extra_matcher = PatternMatcher([(UPat(GroupOp.ALU,name="x"),lower_wide_index),
                                  (UPat(Ops.CAST,name="x"),lower_wide_cast)])
  global_max = (0x7fffffff,)*3
  local_max = (1024,)*3
  code_for_op = {op:lambda:None for op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX, Ops.NEG, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE,
                                         Ops.WHERE, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR, Ops.RECIPROCAL,
                                         Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.TRUNC, Ops.CDIV, Ops.CMOD)}
  early_decomp_matcher = get_transcendental_patterns(tuple(code_for_op), False)

  def __init__(self, target:Target):
    super().__init__(target)
    self.supports_float4 = bool(IMAGE)
    self.compiler = AdrenoCompiler(target.arch or ARCH)

  def supported_dtypes(self): return {dtypes.float, dtypes.half, dtypes.int, dtypes.uint, dtypes.bool, *dtypes.int8s, *dtypes.int16s}

  def render(self, uops:list[UOp]) -> str:
    def strong(dt):
      if dt in (dtypes.weakfloat,dtypes.half): return dtypes.float
      if dt in (dtypes.weakint,*dtypes.int8s,*dtypes.int16s): return dtypes.uint if dt in dtypes.uints else dtypes.int
      return dt
    instructions:list[dict] = []
    regs:dict[UOp,list[int]] = {}
    pointers:dict[UOp,tuple[UOp,UOp|None]] = {}
    image_pointers:dict[UOp,tuple[UOp,UOp,UOp]] = {}
    image_params=[u for u in uops if u.op is Ops.PARAM and is_image_shape(u._shape)]
    if len(image_params)>32: raise ValueError('native Adreno image resource limit exceeded (32 UAVs)')
    image_indices={u:i for i,u in enumerate(image_params)}
    positions = {u:i for i,u in enumerate(uops)}
    last = dict(positions)
    for i,u in enumerate(uops):
      for s in u.src: last[s] = max(last.get(s,0),i)
    # Address expressions and register views retain their operands until use.
    for u in reversed(uops):
      if u.op in (Ops.AFTER,Ops.RESHAPE,Ops.SHRINK,Ops.STACK,Ops.INDEX):
        for s in (u.src[:1] if u.op in (Ops.AFTER,Ops.RESHAPE) else u.src): last[s] = max(last.get(s,0),last[u])
    for u in uops:
      if u.op is Ops.END:
        for rng in u.src[1:]:
          if rng.op is not Ops.RANGE: continue
          loop_begin,loop_finish = positions[rng],positions[u]
          for v in uops[:loop_begin+1]:
            if last[v]>=loop_begin: last[v] = max(last[v],loop_finish)
    # r4-r11 are address/constant scratch. Image shaders also reserve r12-r15
    # for RGBA data and r17-r18 for the adjacent 2D coordinate pair; r16 is
    # narrow load/store and FP16 conversion scratch.
    reg_limit = getenv('ADRENO_MAX_REGS',176)
    if not (20 if image_params else 16) <= reg_limit <= 176: raise ValueError('ADRENO_MAX_REGS is outside the native register range')
    available, allocated = set(range(12,reg_limit))-(set(range(12,19)) if image_params else {16}), {}
    free_spills:set[int] = set()
    private_size = 0
    local_offsets:dict[UOp,int] = {}
    shared_size = 0
    current:UOp
    max_reg = 20 if image_params else 12
    def alloc(n=1):
      nonlocal max_reg, private_size
      ret = sorted(available)[:n]
      available.difference_update(ret)
      while len(ret)<n:
        if free_spills: ret.append(free_spills.pop())
        else:
          private_size += 4
          if private_size>4096: raise RuntimeError('native Adreno private-memory spill budget exceeded (4096 bytes/thread)')
          ret.append(-private_size)
      allocated[current] = ret
      max_reg = max(max_reg,max((r for r in ret if r>=0),default=0)+1,182 if private_size else 0)
      return ret
    def raw(op, **kw):
      instructions.append({'op':op, **kw})
      # A830's longest fixed ALU -> non-ALU dependency is six cycles. Drain
      # asynchronous memory/SFU before reuse; this conservative schedule is
      # intentionally the first correctness baseline, not a speed claim.
      if op not in ('label','br','jump','predt','prede','nop','end'):
        instructions.append({'op':'nop','repeat':5,'ss':True,'sy':True})
    def emit(op, **kw):
      reloads = 0
      for key,value in list(kw.items()):
        reg = value[1] if isinstance(value,tuple) and value[0]=='r' else value if key=='src' and isinstance(value,int) else 0
        if isinstance(reg,int) and reg<0:
          tmp = 176+reloads
          reloads += 1
          raw('mov',dst=181,src=('i',-reg-4))
          raw('ldp',dst=tmp,addr=181)
          kw[key] = ('r',tmp) if isinstance(value,tuple) else tmp
      spill = kw.get('dst',0)
      if isinstance(spill,int) and spill<0: kw['dst'] = 180
      raw(op,**kw)
      if isinstance(spill,int) and spill<0:
        raw('mov',dst=181,src=('i',-spill-4))
        raw('stp',src=180,addr=181)
    def mov(dst,src,dt=dtypes.uint):
      dt = strong(dt)
      typ = 'f32' if dt == dtypes.float else 's32' if dt == dtypes.int else 'u32'
      emit('mov',dst=dst,src=src,src_type=typ,dst_type=typ)
    def finish_dtype(dst,dt):
      if dt == dtypes.half:
        emit('mov',dst=16,src=('r',dst),src_type='f32',dst_type='f16',round=1)
        emit('mov',dst=dst,src=('r',16),src_type='f16',dst_type='f32')
      elif dt in dtypes.int8s+dtypes.int16s:
        shift = 32-dt.bitsize
        emit('shl.b',dst=dst,src1=('r',dst),src2=('i',shift))
        emit('shr.b' if dt in dtypes.uints else 'ashr.b',dst=dst,src1=('r',dst),src2=('i',shift))
    def constant_index(u):
      while u.op is Ops.CAST: u = u.src[0]
      if u.op is not Ops.CONST or not isinstance(u.val,int): raise NotImplementedError('native Adreno dynamic register indexing')
      return u.val
    def rs(u, lane=0):
      rr = regs[u]
      return ('r',rr[0 if len(rr)==1 else lane])
    params = [u for u in uops if u.op is Ops.PARAM]
    buffers = sorted({u.arg.slot for u in params if u.addrspace != AddrSpace.ALU})
    scalars = sorted({u.arg.slot for u in params if u.addrspace == AddrSpace.ALU})
    layout, offsets, end = [], {}, 0
    for u in params:
      if u in image_indices: continue
      size = u.dtype.itemsize if u.addrspace == AddrSpace.ALU else 8
      if u.addrspace == AddrSpace.ALU and size != 4: raise NotImplementedError(f'native Adreno scalar width {size}')
      end = round_up(end,size)
      offsets[u] = (16+end)//4
      slot = len(buffers)+scalars.index(u.arg.slot) if u.addrspace == AddrSpace.ALU else buffers.index(u.arg.slot)
      layout.append((slot,end,size))
      end += size
    if end+16 > 1024: raise ValueError('native Adreno arguments exceed constant upload')
    # Match TinyELF's compact buffer projection followed by its scalar projection.
    signature = [(u.arg.name,buffers.index(u.arg.slot),u.dtype.name,u._shape) for u in params if u.addrspace != AddrSpace.ALU]
    signature += [(u.arg.name,len(buffers)+scalars.index(u.arg.slot),u.dtype.name,u._shape)
                  for u in sorted(params,key=lambda x:x.arg.slot) if u.addrspace == AddrSpace.ALU]
    writes = sorted({buffers.index(u.src[0].buf_uop.arg.slot) for u in uops
                     if u.op is Ops.STORE and u.src[0].buf_uop.op is Ops.PARAM})
    def address(ptr, lane=0):
      root, idx = pointers[ptr]
      if root.addrspace not in (AddrSpace.GLOBAL,AddrSpace.LOCAL): raise NotImplementedError('native Adreno memory address space')
      c = offsets.get(root,0)
      shift = root.dtype.itemsize.bit_length()-1
      if idx is None:
        mov(6,('i',lane*root.dtype.itemsize))
        mov(8,('i',0))
      else:
        mov(6,rs(idx))
        if idx.dtype in dtypes.int64s:
          if len(regs[idx])!=2: raise NotImplementedError('native Adreno 64-bit index shape')
          mov(8,rs(idx,1))
          if shift:
            emit('shl.b',dst=8,src1=('r',8),src2=('i',shift))
            emit('shr.b',dst=7,src1=rs(idx),src2=('i',32-shift))
            emit('or.b',dst=8,src1=('r',8),src2=('r',7))
            emit('shl.b',dst=6,src1=('r',6),src2=('i',shift))
        elif shift:
          emit('shr.b' if strong(idx.dtype)==dtypes.uint else 'ashr.b',dst=8,src1=rs(idx),src2=('i',32-shift))
          emit('shl.b',dst=6,src1=('r',6),src2=('i',shift))
        elif strong(idx.dtype)==dtypes.uint: mov(8,('i',0))
        else: emit('ashr.b',dst=8,src1=rs(idx),src2=('i',31))
        if lane:
          mov(10,('r',6))
          emit('add.u',dst=6,src1=('r',6),src2=('i',lane*root.dtype.itemsize))
          emit('cmps.u',dst=7,src1=('r',6),src2=('r',10),cond=0)
          emit('add.u',dst=8,src1=('r',8),src2=('r',7))
      if root.addrspace == AddrSpace.LOCAL:
        mov(9,('i',local_offsets[root]))
        emit('add.u',dst=4,src1=('r',9),src2=('r',6))
        return 4
      # Full 64-bit pointer addition, including signed offsets and low-word carry.
      mov(9,('c',c))
      emit('add.u',dst=4,src1=('r',9),src2=('r',6))
      emit('cmps.u',dst=7,src1=('r',4),src2=('r',9),cond=0)
      mov(5,('c',c+1))
      emit('add.u',dst=5,src1=('r',5),src2=('r',7))
      emit('add.u',dst=5,src1=('r',5),src2=('r',8))
      return 4
    def compare(dst, a,b,dt,cond):
      emit('cmps.f' if dt==dtypes.float else 'cmps.u' if dt in (dtypes.uint,dtypes.bool) else 'cmps.s',
           dst=dst,src1=a,src2=b,cond=cond)
    loops:dict[UOp,tuple[str,str]] = {}
    ifs:list[UOp] = []
    max_branch, branch_depth = 0, 0
    for u in uops:
      current = u
      for owner in list(allocated):
        if last[owner] < positions[u]:
          for reg in allocated.pop(owner):
            if reg<0: free_spills.add(reg)
            else: available.add(reg)
      if u.op is Ops.CONST:
        regs[u] = alloc(u.max_numel())
        val = struct.unpack('<I',struct.pack('<f',u.arg))[0] if strong(u.dtype)==dtypes.float else int(u.arg)&0xffffffff
        for rr in regs[u]: mov(rr,('i',val),u.dtype)
        for rr in regs[u]: finish_dtype(rr,u.dtype)
      elif u.op is Ops.PARAM:
        if u in image_indices: pass
        elif u.addrspace == AddrSpace.GLOBAL: pointers[u] = (u,None)
        elif u.addrspace == AddrSpace.ALU:
          regs[u] = alloc()
          mov(regs[u][0],('c',offsets[u]),u.dtype)
        else: raise NotImplementedError('native Adreno parameter address space')
      elif u.op is Ops.BUFFER:
        if u.addrspace == AddrSpace.LOCAL:
          local_offsets[u] = shared_size
          shared_size = round_up(shared_size+u.max_numel()*u.dtype.itemsize,16)
          if shared_size>32768: raise ValueError('native Adreno shared memory exceeds 32KiB')
        elif u.addrspace == AddrSpace.REG: regs[u] = alloc(u.max_numel())
        else: raise NotImplementedError(f'native Adreno buffer address space {u.addrspace}')
        pointers[u] = (u,None)
      elif u.op in (Ops.AFTER,Ops.RESHAPE):
        if u.src[0] in regs: regs[u] = regs[u.src[0]]
        if u.src[0] in pointers: pointers[u] = pointers[u.src[0]]
      elif u.op is Ops.SPECIAL:
        regs[u] = alloc()
        # WIE writes workgroup IDs to the shared r48 register file on A830,
        # and local invocation IDs to the ordinary r0 register file.
        mov(regs[u][0],('r',int(u.arg[-1])+(0 if u.arg[0]=='l' else 192)))
      elif u.op in (Ops.INDEX,Ops.SHRINK):
        if u.src[0] in image_indices:
          if u.op is not Ops.INDEX or len(u.src)!=3 or u.max_numel()!=4:
            raise NotImplementedError('native Adreno images require four-component 2D indexing')
          image_pointers[u]=(u.src[0],u.src[1],u.src[2])
        elif u.src[0] in pointers:
          root,_ = pointers[u.src[0]]
          pointers[u] = (root,u.src[1])
        else:
          offset = constant_index(u.src[1])
          size = u.max_numel() if u.op is Ops.SHRINK else 1
          regs[u] = regs[u.src[0]][offset:offset+size]
          if len(regs[u])!=size: raise ValueError('native Adreno register index out of bounds')
      elif u.op is Ops.LOAD:
        regs[u] = alloc(u.max_numel())
        if u.src[0] in image_pointers:
          root,y,x=image_pointers[u.src[0]]
          if u.max_numel()!=4: raise NotImplementedError('native Adreno image load requires RGBA')
          mov(17,rs(x))
          mov(18,rs(y))
          emit('ldib',data=12,coord=17,uav=image_indices[root])
          for lane,dst in enumerate(regs[u]):
            if len(u.src)>1:
              mov(dst,rs(u.src[1],lane),u.dtype)
              compare(248,rs(u.src[2],lane),('i',0),dtypes.uint,5)
              emit('predt')
              max_branch = max(max_branch,branch_depth+1)
            mov(dst,('r',12+lane),u.dtype)
            if len(u.src)>1: emit('prede')
          continue
        root, idx = pointers[u.src[0]]
        if root.addrspace == AddrSpace.REG:
          for lane,dst in enumerate(regs[u]): mov(dst,('r',regs[root][lane+(constant_index(idx) if idx is not None else 0)]),u.dtype)
        else:
          if root.dtype.itemsize not in (1,2,4): raise NotImplementedError(f'native Adreno buffer dtype {root.dtype}')
          for lane,dst in enumerate(regs[u]):
            if len(u.src)>1:
              mov(dst,rs(u.src[1],lane),u.dtype)
              compare(248,rs(u.src[2],lane),('i',0),dtypes.uint,5)
              emit('predt')
              max_branch = max(max_branch,branch_depth+1)
            addr = address(u.src[0],lane)
            if root.dtype.itemsize<4:
              typ = 'f16' if root.dtype==dtypes.half else 'u8' if root.dtype.itemsize==1 else 'u16'
              emit('ldl' if root.addrspace==AddrSpace.LOCAL else 'ldg',dst=16,addr=addr,type=typ)
              emit('mov',dst=dst,src=('r',16),src_type='f16' if typ=='f16' else 'u16',dst_type='f32' if typ=='f16' else 'u32')
              if root.dtype in dtypes.int8s+dtypes.int16s: finish_dtype(dst,root.dtype)
            else: emit('ldl' if root.addrspace==AddrSpace.LOCAL else 'ldg',dst=dst,addr=addr,type='f32' if u.dtype==dtypes.float else 'u32')
            if len(u.src)>1: emit('prede')
      elif u.op is Ops.STORE:
        if u.src[0] in image_pointers:
          root,y,x=image_pointers[u.src[0]]
          if u.src[1].max_numel()!=4: raise NotImplementedError('native Adreno image store requires RGBA')
          mov(17,rs(x))
          mov(18,rs(y))
          for lane,src in enumerate(regs[u.src[1]]): mov(12+lane,('r',src),u.src[1].dtype)
          emit('stib',data=12,coord=17,uav=image_indices[root])
          continue
        root, idx = pointers[u.src[0]]
        for lane,src in enumerate(regs[u.src[1]]):
          if root.addrspace == AddrSpace.REG:
            mov(regs[root][lane+(constant_index(idx) if idx is not None else 0)],('r',src),root.dtype)
          else:
            if root.dtype.itemsize not in (1,2,4): raise NotImplementedError(f'native Adreno buffer dtype {root.dtype}')
            addr = address(u.src[0],lane)
            if root.dtype.itemsize<4:
              typ = 'f16' if root.dtype==dtypes.half else 'u8' if root.dtype.itemsize==1 else 'u16'
              emit('mov',dst=16,src=('r',src),src_type='f32' if typ=='f16' else 'u32',dst_type='f16' if typ=='f16' else 'u16',
                   round=1 if typ=='f16' else 0)
              emit('stl' if root.addrspace==AddrSpace.LOCAL else 'stg',src=16,addr=addr,type=typ)
            else: emit('stl' if root.addrspace==AddrSpace.LOCAL else 'stg',src=src,addr=addr,type='f32' if root.dtype==dtypes.float else 'u32')
      elif u.op is Ops.RANGE:
        regs[u] = alloc()
        mov(regs[u][0],('i',0))
        start, finish = f'loop_{len(loops)}', f'loop_end_{len(loops)}'
        loops[u] = start,finish
        emit('label',name=start)
        compare(248,rs(u),rs(u.src[0]),dtypes.int,0)
        emit('br',target=finish,inv=True)
        branch_depth += 1
        max_branch = max(max_branch,branch_depth)
      elif u.op is Ops.END:
        for rng in reversed(u.src[1:]):
          if rng not in loops: raise NotImplementedError('native Adreno loop end')
          start,finish = loops[rng]
          emit('add.u',dst=regs[rng][0],src1=rs(rng),src2=('i',1))
          emit('jump',target=start)
          emit('label',name=finish)
          branch_depth -= 1
      elif u.op is Ops.BARRIER:
        if ifs: raise NotImplementedError('native Adreno divergent barrier')
        emit('fence',local=True,read=True,write=True,ss=True,sy=True)
        emit('bar',local=True,ss=True,sy=True)
      elif u.op is Ops.IF:
        compare(248,rs(u.src[0]),('i',0),dtypes.uint,5)
        emit('predt')
        ifs.append(u)
        branch_depth += 1
        max_branch = max(max_branch,branch_depth)
      elif u.op is Ops.ENDIF:
        ifs.pop()
        emit('prede')
        branch_depth -= 1
      elif u.op is Ops.STACK: regs[u] = [rr for s in u.src for rr in regs[s]]
      elif u.op in (Ops.CAST,Ops.BITCAST):
        if u.op is Ops.BITCAST and u.src[0].dtype in dtypes.int64s and u.dtype.bitsize==32:
          regs[u] = alloc(2)
          for lane,dst in enumerate(regs[u]): mov(dst,rs(u.src[0],lane))
          continue
        if u.dtype in dtypes.int64s:
          if u.op is Ops.BITCAST and u.src[0].dtype.bitsize==32 and u.src[0].max_numel()==2:
            regs[u] = alloc(2)
            for lane,dst in enumerate(regs[u]): mov(dst,rs(u.src[0],lane))
            continue
          if u.op is not Ops.CAST or (u.src[0].dtype!=dtypes.weakint and u.src[0].dtype.bitsize!=32):
            raise NotImplementedError(f'native Adreno 64-bit address cast {u.src[0].dtype} -> {u.dtype}')
          regs[u] = alloc(2)
          mov(regs[u][0],rs(u.src[0]))
          if u.src[0].op is Ops.CONST: mov(regs[u][1],('i',(int(u.src[0].val)>>32)&0xffffffff))
          elif u.src[0].dtype in dtypes.uints: mov(regs[u][1],('i',0))
          else: emit('ashr.b',dst=regs[u][1],src1=rs(u.src[0]),src2=('i',31))
          continue
        regs[u] = alloc(u.max_numel())
        input_type, output_type = strong(u.src[0].dtype), strong(u.dtype)
        for lane,dst in enumerate(regs[u]):
          src = rs(u.src[0],lane)
          if u.op is Ops.BITCAST and (u.dtype==dtypes.half or u.src[0].dtype==dtypes.half):
            it,ot = ('f32','f16') if u.src[0].dtype==dtypes.half else ('u32','u16')
            emit('mov',dst=16,src=src,src_type=it,dst_type=ot)
            it,ot = ('f16','f32') if u.dtype==dtypes.half else ('u16','u32')
            emit('mov',dst=dst,src=('r',16),src_type=it,dst_type=ot)
          elif u.op is Ops.BITCAST or output_type==input_type: mov(dst,src)
          elif u.dtype==dtypes.bool: compare(dst,src,('i',0),u.src[0].dtype,5)
          elif output_type in (dtypes.float,dtypes.int,dtypes.uint) and input_type in (dtypes.float,dtypes.int,dtypes.uint,dtypes.bool):
            it = 'f32' if input_type==dtypes.float else 's32' if input_type==dtypes.int else 'u32'
            ot = 'f32' if output_type==dtypes.float else 's32' if output_type==dtypes.int else 'u32'
            emit('mov',dst=dst,src=src,src_type=it,dst_type=ot)
          else: raise NotImplementedError(f'native Adreno cast {u.src[0].dtype} -> {u.dtype}')
          finish_dtype(dst,u.dtype)
      elif u.op in GroupOp.ALU:
        if u.dtype in dtypes.int64s: raise NotImplementedError('native Adreno wide ALU requires UOp decomposition')
        regs[u] = alloc(u.max_numel())
        for lane,dst in enumerate(regs[u]):
          ss = [rs(s,lane) for s in u.src]
          dt = strong(u.src[0].dtype)
          typ = 'f' if dt==dtypes.float else 'u' if dt in (dtypes.uint,dtypes.bool) else 's'
          if u.op in (Ops.ADD,Ops.SUB,Ops.MAX):
            if u.op is Ops.SUB and typ=='f':
              emit('mul.f',dst=10,src1=ss[1],src2=('r',11))
              emit('add.f',dst=dst,src1=ss[0],src2=('r',10))
            else: emit({Ops.ADD:'add',Ops.SUB:'sub',Ops.MAX:'max'}[u.op]+'.'+typ,dst=dst,src1=ss[0],src2=ss[1])
          elif u.op is Ops.MUL:
            if typ=='f': emit('mul.f',dst=dst,src1=ss[0],src2=ss[1])
            else:
              emit('mull.u',dst=dst,src1=ss[0],src2=ss[1])
              emit('madsh.m16',dst=dst,src1=ss[0],src2=ss[1],src3=('r',dst))
              emit('madsh.m16',dst=dst,src1=ss[1],src2=ss[0],src3=('r',dst))
          elif u.op in (Ops.CMPLT,Ops.CMPEQ,Ops.CMPNE):
            compare(dst,ss[0],ss[1],dt,{Ops.CMPLT:0,Ops.CMPEQ:4,Ops.CMPNE:5}[u.op])
          elif u.op is Ops.WHERE: emit('sel.b32',dst=dst,src1=ss[1],src2=ss[0],src3=ss[2])
          elif u.op in (Ops.AND,Ops.OR,Ops.XOR,Ops.SHL,Ops.SHR):
            emit({Ops.AND:'and.b',Ops.OR:'or.b',Ops.XOR:'xor.b',Ops.SHL:'shl.b',Ops.SHR:'ashr.b' if typ=='s' else 'shr.b'}[u.op],
                 dst=dst,src1=ss[0],src2=ss[1])
          elif u.op in (Ops.CDIV,Ops.CMOD):
            # Exact restoring division. Carry preserves the 33rd remainder bit
            # when the unsigned divisor is above 2**31.
            max_reg = max(max_reg,192)
            mov(182,ss[0])
            mov(183,ss[1])
            if typ=='s':
              compare(189,('r',182),('i',0),dtypes.int,0)
              compare(190,('r',183),('i',0),dtypes.int,0)
              emit('xor.b',dst=191,src1=('r',189),src2=('r',190))
              emit('sub.u',dst=188,src1=('i',0),src2=('r',182))
              emit('sel.b32',dst=182,src1=('r',188),src2=('r',189),src3=('r',182))
              emit('sub.u',dst=188,src1=('i',0),src2=('r',183))
              emit('sel.b32',dst=183,src1=('r',188),src2=('r',190),src3=('r',183))
            mov(184,('i',0))
            mov(185,('i',0))
            for bit in range(31,-1,-1):
              emit('shr.b',dst=186,src1=('r',184),src2=('i',31))
              emit('shl.b',dst=184,src1=('r',184),src2=('i',1))
              emit('shr.b',dst=188,src1=('r',182),src2=('i',bit))
              emit('and.b',dst=188,src1=('r',188),src2=('i',1))
              emit('or.b',dst=184,src1=('r',184),src2=('r',188))
              compare(187,('r',184),('r',183),dtypes.uint,3)
              emit('or.b',dst=187,src1=('r',187),src2=('r',186))
              emit('sub.u',dst=188,src1=('r',184),src2=('r',183))
              emit('sel.b32',dst=184,src1=('r',188),src2=('r',187),src3=('r',184))
              emit('shl.b',dst=188,src1=('r',187),src2=('i',bit))
              emit('or.b',dst=185,src1=('r',185),src2=('r',188))
            result=184 if u.op is Ops.CMOD else 185
            if typ=='s':
              emit('sub.u',dst=188,src1=('i',0),src2=('r',result))
              emit('sel.b32',dst=dst,src1=('r',188),src2=('r',189 if u.op is Ops.CMOD else 191),src3=('r',result))
            else: mov(dst,('r',result))
          elif u.op is Ops.NEG: emit('sub.s',dst=dst,src1=('i',0),src2=ss[0]) if typ!='f' else \
            emit('mul.f',dst=dst,src1=ss[0],src2=('r',11))
          elif u.op in (Ops.RECIPROCAL,Ops.SQRT,Ops.LOG2,Ops.EXP2):
            emit({Ops.RECIPROCAL:'rcp',Ops.SQRT:'sqrt',Ops.LOG2:'log2',Ops.EXP2:'exp2'}[u.op],dst=dst,src=ss[0])
          elif u.op is Ops.TRUNC: emit('trunc.f',dst=dst,src1=ss[0])
          else: raise NotImplementedError(f'native Adreno ALU {u.op}')
          finish_dtype(dst,u.dtype)
      elif u.op not in (Ops.SINK,Ops.NOOP,Ops.GROUP): raise NotImplementedError(f'native Adreno lowering {u.op}')
    if ifs or branch_depth: raise ValueError('unbalanced native Adreno control flow')
    # Reserve -1.0 in r2.w for float negation, separate from allocated UOp registers.
    instructions[:0] = [{'op':'mov','dst':11,'src':('i',0xbf800000),'src_type':'f32','dst_type':'f32'},
                        {'op':'nop','repeat':5,'ss':True,'sy':True}]
    emit('nop',ss=True,sy=True)
    emit('end')
    return json.dumps({'instructions':instructions,'fregs':(max_reg+3)//4,'constlen':(end+16+15)//16,
                       'private_size':private_size,'shared_size':shared_size,
                       'branchstack':max_branch,'params':layout,'writes':writes,'signature':signature,
                       'num_uavs':len(image_params)},separators=(',',':'))
