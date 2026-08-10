import base64, ctypes, pathlib, tempfile, hashlib
from tinygrad.device import Compiler
from tinygrad.helpers import cpu_objdump, system, data64
from tinygrad.runtime.autogen import mesa, llvm, libc
from tinygrad.runtime.support.compiler_llvm import CPULLVMCompiler, expect, cerr

# NB: compilers assume mesa's glsl type cache is managed externally with mesa.glsl_type_singleton_init_or_ref() and mesa.glsl_type_singleton_decref()

def rzalloc(typ, ctx=None, **kwargs):
  s = ctypes.cast(mesa.rzalloc_size(ctypes.cast(ctx, ctypes.c_void_p), ctypes.sizeof(typ)), ctypes.POINTER(typ))
  for k,v in kwargs.items(): setattr(s.contents, k, v)
  return s

def deserialize(enc_src, opts):
  blobreader = mesa.struct_blob_reader()
  mesa.blob_reader_init(blobreader, src:=base64.b64decode(enc_src), len(src))
  return mesa.nir_deserialize(None, ctypes.cast(opts, ctypes.POINTER(mesa.nir_shader_compiler_options)), blobreader)

class LVPCompiler(CPULLVMCompiler):
  def __init__(self, arch): CPULLVMCompiler.__init__(self, arch.split(","), cache_key="compile_lvp")

  def compile(self, src) -> bytes:
    shader, ctx = deserialize(src, mesa.lvp_nir_options), llvm.LLVMGetGlobalContext()
    gallivm = mesa.gallivm_create(None, mesa.lp_context_ref(ctypes.cast(ctx, ctypes.POINTER(mesa.struct_LLVMOpaqueContext)), True), None).contents
    module, builder = ctypes.cast(gallivm.module, llvm.LLVMModuleRef), ctypes.cast(gallivm.builder, llvm.LLVMBuilderRef)

    params = mesa.struct_lp_build_tgsi_params(mesa.struct_lp_type(floating=True, sign=True, width=32, length=4),
      resources_type=mesa.lp_build_jit_resources_type(gallivm), mask=ctypes.pointer(mesa.struct_lp_build_mask_context()))

    pt = llvm.LLVMPointerType(ctypes.cast(params.resources_type, llvm.LLVMTypeRef), 0)
    fn = llvm.LLVMAddFunction(module, shader.contents.info.name, llvm.LLVMFunctionType(llvm.LLVMVoidTypeInContext(ctx), pt, 1, 0))
    llvm.LLVMPositionBuilderAtEnd(builder, llvm.LLVMAppendBasicBlockInContext(ctx, fn, b"entry"))

    params.consts_ptr = mesa.lp_build_struct_get_ptr2(gallivm, params.resources_type,
      ctypes.cast(llvm.LLVMGetParam(fn, 0), mesa.LLVMValueRef), mesa.LP_JIT_RES_CONSTANTS, b"constants")
    mesa.lp_build_mask_begin(params.mask, gallivm, params.type, mesa.lp_build_one(gallivm, params.type))
    mesa.lp_build_mask_end(params.mask)

    mesa.lp_build_nir_soa(gallivm, shader, params, None)
    llvm.LLVMBuildRetVoid(builder)
    mesa.gallivm_verify_function(gallivm, ctypes.cast(fn, mesa.LLVMValueRef))
    mesa.lp_passmgr_run(gallivm.passmgr, gallivm.module, ctypes.cast(self.target_machine, mesa.LLVMTargetMachineRef), gallivm.module_name)
    obj_buf = expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(self.target_machine, module, llvm.LLVMObjectFile, err:=cerr(),
                                                              ctypes.pointer(buf:=llvm.LLVMMemoryBufferRef())), err, buf)
    obj = ctypes.string_at(llvm.LLVMGetBufferStart(obj_buf), llvm.LLVMGetBufferSize(obj_buf))

    mesa.gallivm_destroy(gallivm)
    mesa.ralloc_free(shader)
    return obj

  def disassemble(self, lib: bytes): cpu_objdump(lib)

class NAKCompiler(Compiler):
  # simplified from https://elixir.bootlin.com/mesa/mesa-26.0.3/source/src/nouveau/winsys/nouveau_device.c#L118
  @staticmethod
  def warps_per_sm(arch): return 48 if arch in ("sm_86", "sm_87", "sm_89", "sm_120") else 64
  def __init__(self, arch):
    self.arch = arch
    self.cc = mesa.nak_compiler_create(mesa.struct_nv_device_info(sm=int(arch[3:]), max_warps_per_mp=self.warps_per_sm(arch)))
    self.nir_options = bytes(mesa.nak_nir_options(self.cc).contents)
    super().__init__(f"compile_nak_{arch}")

  def __del__(self): mesa.nak_compiler_destroy(self.cc)

  def __reduce__(self): return NAKCompiler, (self.arch,)

  def compile(self, src) -> bytes:
    shader = deserialize(src, self.nir_options)
    mesa.nak_preprocess_nir(shader, self.cc)
    ret = bytes((out:=mesa.nak_compile_shader(shader, False, self.cc, 0, None).contents).info) + ctypes.string_at(out.code, out.code_size)
    mesa.nak_shader_bin_destroy(out)
    mesa.ralloc_free(shader)
    return ret

  def disassemble(self, lib: bytes):
    try:
      fn = (pathlib.Path(tempfile.gettempdir()) / f"tinynak_{hashlib.md5(lib).hexdigest()}").as_posix()
      with open(fn, "wb") as f: f.write(lib[ctypes.sizeof(mesa.struct_nak_shader_info):])
      print(system(f"nvdisasm -b SM{self.arch[3:]} {fn}"))
    except Exception as e: print("Failed to generate SASS", str(e), "Make sure your PATH contains nvdisasm binary of compatible version.")

def disas_adreno(lib:bytes, gpu_id=630):
  with tempfile.TemporaryFile('w+') as tf:
    mesa_fp = ctypes.cast(fp:=libc.fdopen(tf.fileno(), b"w"), ctypes.POINTER(mesa.struct__IO_FILE))
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
    def hd(data, n, instr):
      fst, snd = data64(ctypes.cast(instr, ctypes.POINTER(ctypes.c_uint64)).contents.value)
      # print with libc so that output interleaves properly
      libc.fputs(f"{n:04} [{fst:08x}_{snd:08x}] ".encode(), fp)

    mesa.ir3_isa_disasm(lib, len(lib), mesa_fp, mesa.struct_isa_decode_options(gpu_id, True, 0, True, pre_instr_cb=hd))
    libc.fflush(fp)
    tf.seek(0)
    print(tf.read())

class IR3Compiler(Compiler):
  def __init__(self, arch):
    self.arch = arch
    arch_name, *opts = arch.split(',')
    gpu_id = int(arch_name[1:])
    chip_id = 0x06030001 if gpu_id == 630 else int(next(x.split('=', 1)[1] for x in opts if x.startswith('chip_id=')), 0)
    self.dev_id = mesa.struct_fd_dev_id(gpu_id, chip_id)
    dev_info = mesa.fd_dev_info_raw(self.dev_id)
    if not dev_info or dev_info.contents.chip == 0:
      raise RuntimeError(f"unsupported Adreno chip_id={chip_id:#x} for {arch_name!r}")
    self.cc = mesa.ir3_compiler_create(None, self.dev_id, dev_info, mesa.struct_ir3_compiler_options(disable_cache=True))
    self.cc.contents.has_preamble = False
    self.nir_options = bytes(mesa.ir3_get_compiler_options(self.cc).contents)
    super().__init__(f"compile_ir3_mesa_26_1_4_{arch}")

  def __del__(self):
    if getattr(self, 'cc', None): mesa.ir3_compiler_destroy(self.cc)

  def __reduce__(self): return IR3Compiler, (self.arch,)

  def compile(self, src) -> bytes:
    nir_shader = deserialize(src, self.nir_options)
    mesa.ir3_nir_lower_io_vars_to_temporaries(nir_shader)
    mesa.ir3_finalize_nir(self.cc, mesa.struct_ir3_shader_nir_options(), nir_shader)
    # Variants own their constant state in Mesa's ralloc tree.
    # https://gitlab.freedesktop.org/mesa/mesa/-/blob/6dfbc555b4128ee51139c5f78c5aba2594c9701b/src/freedreno/ir3/ir3_shader.c#L496-562
    ir3_shader = rzalloc(mesa.struct_ir3_shader, compiler=self.cc, type=mesa.MESA_SHADER_COMPUTE, nir=nir_shader)
    shader = ir3_shader.contents
    mesa.ir3_nir_post_finalize(ir3_shader)
    variant = rzalloc(mesa.struct_ir3_shader_variant, ir3_shader, shader=ir3_shader, type=shader.type,
                      compiler=self.cc, key=mesa.struct_ir3_shader_key())
    v = variant.contents
    v.const_state = rzalloc(mesa.struct_ir3_const_state, variant)
    shader.variants, shader.variant_count = variant, 1
    v.num_uavs = (info:=nir_shader.contents.info).num_ssbos + info.num_images
    if mesa.ir3_compile_shader_nir(self.cc, ir3_shader, variant): raise RuntimeError(f"IR3 compilation failed for {self.arch!r}")
    binary = ctypes.cast(mesa.ir3_shader_assemble(variant), ctypes.POINTER(ctypes.c_uint32))
    ret = bytes(v) + bytes(v.const_state.contents) + ctypes.string_at(v.imm_state.values, v.imm_state.count * 4) + \
      ctypes.string_at(binary, v.info.size)
    mesa.ralloc_free(ir3_shader)
    mesa.ralloc_free(nir_shader)
    return ret

  @staticmethod
  def unpack_lib(lib: bytes) -> tuple[mesa.struct_ir3_shader_variant, mesa.struct_ir3_const_state, bytes, bytes]:
    shifted = lib[ctypes.sizeof(v:=mesa.struct_ir3_shader_variant.from_buffer_copy(lib)):]
    shifted = shifted[ctypes.sizeof(cs:=mesa.struct_ir3_const_state.from_buffer_copy(shifted)):]
    return v, cs, shifted[:v.imm_state.count * 4], shifted[v.imm_state.count * 4:]

  def disassemble(self, lib: bytes): disas_adreno(self.unpack_lib(lib)[3], self.dev_id.gpu_id)
