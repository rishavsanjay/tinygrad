import importlib
from typing import TYPE_CHECKING
from tinygrad.runtime.support.mesa_abi import MESA_VERSION

if TYPE_CHECKING:
  from tinygrad.runtime.autogen.mesa_26_2_5 import *  # noqa: F403
else:
  _bindings = importlib.import_module("tinygrad.runtime.autogen.mesa_" + MESA_VERSION.replace(".", "_"))
  globals().update({name:value for name,value in vars(_bindings).items() if not name.startswith("__")})
