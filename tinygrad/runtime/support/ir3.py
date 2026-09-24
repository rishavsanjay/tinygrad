import hashlib, json, struct
from dataclasses import asdict, dataclass

# Only values consumed by the runtime cross this boundary, never Mesa's C pointers.
@dataclass(frozen=True)
class IR3Shader:
  arch: str
  build: str
  branchstack: int
  pvtmem_size: int
  shared_size: int
  pvtmem_per_wave: bool
  early_preamble: bool
  mergedregs: bool
  instrlen: int
  double_threadsize: bool
  constlen: int
  wgsz: int
  wgid: int
  lid: int
  buf_off: int
  imm_off: int
  num_uavs: int
  tex_to_image: tuple[int, ...]
  fregs: int
  hregs: int
  round_robin_mode: bool
  # (compact call argument slot, byte offset, byte size), in NIR parameter order, excluding images.
  params: tuple[tuple[int, int, int], ...]
  writes: tuple[int, ...]

  def validate(self):
    for name, value in asdict(self).items():
      if name in ('arch', 'build'):
        if type(value) is not str or not value: raise ValueError(f"invalid IR3 {name}")
      elif name in ('pvtmem_per_wave', 'early_preamble', 'mergedregs', 'double_threadsize', 'round_robin_mode'):
        if type(value) is not bool: raise ValueError(f"invalid IR3 {name}")
      elif name not in ('tex_to_image', 'params', 'writes'):
        if type(value) is not int or not 0 <= value <= 0xffffffff: raise ValueError(f"invalid IR3 {name}")
    if self.constlen > 512 or self.branchstack > 64 or self.fregs > 256 or self.hregs > 256:
      raise ValueError("IR3 shader resource limits exceeded")
    if not 0 <= len(self.tex_to_image) <= min(self.num_uavs, 31) or self.num_uavs > 32:
      raise ValueError("IR3 image resource limits exceeded")
    if any(type(i) is not int or not 0 <= i < self.num_uavs for i in self.tex_to_image):
      raise ValueError("IR3 texture mapping outside image table")
    if any(type(i) is not int or i < 0 for i in self.writes): raise ValueError("invalid IR3 written arguments")
    end = 0
    for slot, offset, size in self.params:
      if any(type(v) is not int for v in (slot, offset, size)) or slot < 0 or size not in (1, 2, 4, 8):
        raise ValueError("invalid IR3 argument layout")
      if offset < end or offset % size: raise ValueError("overlapping or unaligned IR3 arguments")
      end = offset + size
    if self.buf_off + end > 1024: raise ValueError("IR3 arguments exceed uploaded constant data")

  def pack(self, immediates:bytes, binary:bytes) -> bytes:
    self.validate()
    meta = json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()
    payload = meta + immediates + binary
    return struct.pack('<4sIII', b'IR3\x01', len(meta), len(immediates), len(binary)) + hashlib.sha256(payload).digest() + payload

  @staticmethod
  def unpack(data:bytes) -> tuple['IR3Shader', bytes, bytes]:
    if len(data) < 48: raise ValueError("truncated IR3 artifact")
    magic, nmeta, nimm, nbin = struct.unpack_from('<4sIII', data)
    if magic != b'IR3\x01': raise ValueError("unsupported IR3 artifact version; recompile shader")
    if not 0 < nmeta <= 65536 or not nbin or nimm % 4 or len(data) != 48 + nmeta + nimm + nbin:
      raise ValueError("invalid IR3 artifact lengths")
    if hashlib.sha256(data[48:]).digest() != data[16:48]: raise ValueError("corrupt IR3 artifact")
    try:
      fields = json.loads(data[48:48+nmeta])
      fields['tex_to_image'] = tuple(fields['tex_to_image'])
      fields['params'] = tuple(tuple(p) for p in fields['params'])
      fields['writes'] = tuple(fields['writes'])
      shader = IR3Shader(**fields)
      shader.validate()
    except (TypeError, KeyError, UnicodeError) as e: raise ValueError("invalid IR3 metadata") from e
    if shader.imm_off + nimm > 1024: raise ValueError("IR3 immediates exceed uploaded constant data")
    return shader, data[48+nmeta:48+nmeta+nimm], data[48+nmeta+nimm:]
