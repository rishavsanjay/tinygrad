# Runtime Autogen Implementation Details

`tinygrad/runtime/autogen/` manages the automatic generation of ctypes bindings for various hardware libraries. This allows tinygrad to interface with drivers (CUDA, OpenCL, HIP, etc.) without external C++ dependencies or heavy wrapper libraries.

## 1. `__init__.py`

The core logic for fetching, parsing, and generating bindings.

*   **`load(name, dll, files, ...)`**:
    *   Checks if the binding file (`name.py`) exists.
    *   If not (or `REGEN=1`), it:
        *   Fetches sources (tarballs from GitHub/official sites).
        *   Runs preprocessing (e.g., generating headers for Mesa/Freedreno).
        *   Calls `tinygrad.runtime.support.autogen.gen` to parse C headers and generate Python code.
        *   Writes the file.
    *   Imports and returns the module.

## 2. Supported Bindings

*   **`cuda` / `nvrtc`**: NVIDIA CUDA driver and runtime compiler.
*   **`opencl`**: Standard OpenCL headers.
*   **`hip` / `comgr` / `hsa`**: AMD ROCm stack.
*   **`metal`**: macOS Metal framework (using `objc` bridges).
*   **`io_uring`**: Linux asynchronous IO (for disk).
*   **`llvm` / `libclang`**: LLVM infrastructure.
*   **`mesa`**: Mesa 3D graphics library (extracted for NIR/IR3 support).

## 3. How it works

The generator (`support/autogen.py`, utilized here) uses `clang` to parse the C header files.
It extracts:
*   **Structs**: Converted to `ctypes.Structure`.
*   **Enums**: Converted to Python constants.
*   **Functions**: Converted to `ctypes` function pointers attached to the DLL.

This system ensures tinygrad is self-contained and can bootstrap its own driver interfaces.
