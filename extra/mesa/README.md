# Mesa compiler versions

tinygrad accepts Mesa 25.2.7 and 26.2.5 during the compiler-library transition.
The default optional dependency remains `tinymesa==25.2.7.2` until the newer
wheels are published. The same tinygrad source also works with an installed
`tinymesa==26.2.5.3` wheel. Do not widen the dependency to arbitrary Mesa versions:
NIR structures and compiler APIs are private interfaces.

The loader selects a complete matching binding module before calling Mesa. New
Tinymesa libraries report their version through `tinymesa_mesa_version()`.
Legacy wheels use their installed package metadata, only when the resolved
library belongs to that package. Unsupported declared versions and conflicting overrides
are rejected. The generated modules also reject selection with the wrong ABI.

`MESA_PATH=/absolute/path/to/library.so` selects a custom compiler. Existing
standalone Tinymesa downloads retain the legacy Mesa 25.2.7 default when they
expose the old NIR API and have no version metadata. This preserves the old
Android download flow; the API probe is not a guarantee for arbitrary Mesa
builds. An unmarked build exposing the newer NIR API requires
`MESA_VERSION=26.2.5`. That variable explicitly declares the custom library's ABI;
its filename is not used to infer compatibility. New Tinymesa downloads report
their version themselves.

LVP, NAK and IR3 compiler caches include the selected Mesa version. Restart the
process after changing the compiler library; do not mix bindings, serialized
NIR or IR3 metadata between versions.

To regenerate one binding set, select its matching library and run with
`REGEN=1`. `mesa_25_2_7.py` and `mesa_26_2_5.py` retain separate pinned source
archives and generation arguments; `mesa.py` only selects the correct set.
