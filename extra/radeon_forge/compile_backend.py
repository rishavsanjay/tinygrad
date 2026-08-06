from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tinygrad.runtime.support.compiler_amd import HIPCCCompiler

from .amd_metadata import resource_report_from_text
from .contracts import Candidate, ResourceReport
from .families import compiler_defines


@dataclass(frozen=True)
class CompiledCandidate:
  candidate: Candidate
  library: bytes
  resources: ResourceReport
  metadata_text: str


def _read_metadata(library: bytes) -> str:
  tools_and_args = (
    ("/opt/rocm/llvm/bin/llvm-readobj", "--notes"),
    ("/opt/rocm/llvm/bin/llvm-objdump", "--notes"),
    ("llvm-readobj", "--notes"),
    ("llvm-objdump", "--notes"),
  )
  with tempfile.NamedTemporaryFile(suffix=".hsaco") as f:
    f.write(library)
    f.flush()
    outputs: list[str] = []
    for tool, flag in tools_and_args:
      executable = tool if Path(tool).exists() else shutil.which(tool)
      if executable is None: continue
      proc = subprocess.run([executable, flag, f.name], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
      if proc.stdout: outputs.append(proc.stdout)
      if proc.returncode == 0 and proc.stdout: break
    return "\n".join(outputs)


def compile_candidate(candidate: Candidate, target: str = "gfx1100") -> CompiledCandidate:
  if candidate.source_path is None: raise ValueError(f"candidate {candidate.candidate_id} has no source_path")
  source = Path(candidate.source_path).read_text(encoding="utf-8")
  options = ["-std=c++20", "-ffast-math", *compiler_defines(candidate.parameters)]
  library = HIPCCCompiler(target, options).compile_cached(source)
  metadata_text = _read_metadata(library)
  return CompiledCandidate(candidate, library, resource_report_from_text(metadata_text), metadata_text)
