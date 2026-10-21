import ctypes, importlib.metadata, os, pathlib, sysconfig
from tinygrad.helpers import DEV
from tinygrad.runtime.support.c import DLL

SUPPORTED_VERSIONS = ("25.2.7", "26.2.5")
dll = DLL("mesa", "tinymesa_cpu" if DEV.renderer == "LVP" else "tinymesa",
          os.path.join(sysconfig.get_paths()["platlib"], "tinymesa"), emsg="pip install tinymesa==25.2.7.2")

def mesa_version() -> str:
  if not getattr(dll, "_name", None): return "25.2.7" # register definitions remain usable without the compiler library
  version = None
  if hasattr(dll, "tinymesa_mesa_version"):
    dll.tinymesa_mesa_version.restype = ctypes.c_char_p
    version = dll.tinymesa_mesa_version().decode()
  else:
    try:
      package = importlib.metadata.distribution("tinymesa")
      if pathlib.Path(dll._name).resolve().parent == pathlib.Path(str(package.locate_file("tinymesa"))).resolve():
        version = ".".join(package.version.split("+")[0].split(".")[:3])
    except importlib.metadata.PackageNotFoundError: pass
  explicit = os.getenv("MESA_VERSION")
  if version is not None and explicit is not None and version != explicit:
    raise RuntimeError(f"MESA_VERSION={explicit} disagrees with loaded Mesa {version}: {dll._name}")
  # Legacy standalone Tinymesa downloads use the original NIR ABI without version metadata.
  if version is None and explicit is None and hasattr(dll, "nir_builder_init_simple_shader") and not hasattr(dll, "nir_def_instr_noninline"):
    version = "25.2.7"
  if (version := version or explicit) is None or version not in SUPPORTED_VERSIONS:
    raise RuntimeError(f"Unsupported or unidentified Mesa {version!r}: {dll._name}; expected {SUPPORTED_VERSIONS}. "
                       "Custom libraries without version metadata require MESA_VERSION.")
  return version

MESA_VERSION = mesa_version()
