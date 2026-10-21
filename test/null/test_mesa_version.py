import os, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad.runtime.support import mesa_abi

class TestMesaVersion(unittest.TestCase):
  def embedded(self, version):
    return SimpleNamespace(_name="/custom/mesa.so", tinymesa_mesa_version=lambda:version.encode())

  def select(self, library, explicit=None, package=None):
    env = {} if explicit is None else {"MESA_VERSION":explicit}
    with patch.object(mesa_abi, "dll", library), patch.dict(os.environ, env, clear=True):
      with patch.object(mesa_abi.importlib.metadata, "distribution", return_value=package,
                        side_effect=mesa_abi.importlib.metadata.PackageNotFoundError if package is None else None):
        return mesa_abi.mesa_version()

  def test_embedded_versions(self):
    for version in ("25.2.7", "26.2.5"):
      with self.subTest(version=version): self.assertEqual(self.select(self.embedded(version)), version)

  def test_reject_unknown_embedded_version(self):
    with self.assertRaisesRegex(RuntimeError, "Unsupported"): self.select(self.embedded("27.0.0"))

  def test_reject_conflicting_override(self):
    with self.assertRaisesRegex(RuntimeError, "disagrees"): self.select(self.embedded("26.2.5"), "25.2.7")

  def test_old_wheel_metadata(self):
    package = SimpleNamespace(version="25.2.7.2", locate_file=lambda _:Path("/installed/tinymesa"))
    self.assertEqual(self.select(SimpleNamespace(_name="/installed/tinymesa/libtinymesa.so"), package=package), "25.2.7")

  def test_legacy_standalone_library(self):
    library = SimpleNamespace(_name="/custom/mesa.so", nir_builder_init_simple_shader=object())
    self.assertEqual(self.select(library), "25.2.7")

  def test_unmarked_new_nir_requires_a_version(self):
    library = SimpleNamespace(_name="/custom/mesa.so", nir_builder_init_simple_shader=object(), nir_def_instr_noninline=object())
    with self.assertRaisesRegex(RuntimeError, "unidentified"): self.select(library)
    self.assertEqual(self.select(library, "26.2.5"), "26.2.5")

  def test_custom_library_cannot_borrow_wheel_version(self):
    package = SimpleNamespace(version="25.2.7.2", locate_file=lambda _:Path("/installed/tinymesa"))
    library = SimpleNamespace(_name="/custom/mesa.so")
    with self.assertRaisesRegex(RuntimeError, "unidentified"): self.select(library, package=package)
    self.assertEqual(self.select(library, "26.2.5", package), "26.2.5")

  def test_reject_unknown_wheel_version(self):
    package = SimpleNamespace(version="26.3.0.1", locate_file=lambda _:Path("/installed/tinymesa"))
    with self.assertRaisesRegex(RuntimeError, "Unsupported"):
      self.select(SimpleNamespace(_name="/installed/tinymesa/libtinymesa.so"), package=package)

  def test_registers_without_a_library(self): self.assertEqual(self.select(SimpleNamespace(_name=None)), "25.2.7")

if __name__ == "__main__": unittest.main()
