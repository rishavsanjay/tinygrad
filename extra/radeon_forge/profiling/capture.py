from __future__ import annotations

import csv, hashlib, json, os, shutil, subprocess, time, uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class CaptureKind(str, Enum):
  COUNTERS = "counters"
  ATT = "att"


@dataclass(frozen=True)
class CaptureRequest:
  kind: CaptureKind
  command: tuple[str, ...]
  stage: str
  session_id: str = ""
  trace_id: str = ""
  agent_step: int = 0
  kernel_regex: str = ""
  counters: tuple[str, ...] = ()
  timeout_seconds: int = 900
  metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptureArtifact:
  path: str
  size_bytes: int
  sha256: str
  suffix: str


@dataclass(frozen=True)
class CaptureResult:
  capture_id: str
  kind: str
  passed: bool
  output_directory: str
  profiler_command: tuple[str, ...]
  target_command: tuple[str, ...]
  returncode: int | None
  elapsed_ms: float
  stdout_tail: str
  stderr_tail: str
  artifacts: tuple[CaptureArtifact, ...]
  summary: Mapping[str, Any]
  error: str = ""

  def to_dict(self) -> dict[str, Any]: return asdict(self)


class Rocprofv3Adapter:
  """Evidence-preserving rocprofv3 adapter for counters and ATT/SQTT.

  CLI options are detected from the installed profiler help rather than assumed
  from a particular ROCm release. Performance is always measured by the target
  hardware; this adapter only captures and normalizes evidence.
  """
  def __init__(self, project_root: str | Path, evidence_root: str | Path):
    self.project_root = Path(project_root).resolve()
    self.evidence_root = Path(evidence_root).resolve()
    self.evidence_root.mkdir(parents=True, exist_ok=True)

  @staticmethod
  def executable() -> str | None: return shutil.which("rocprofv3")

  def probe(self) -> dict[str, Any]:
    executable = self.executable()
    if executable is None: return {"available":False, "reason":"rocprofv3 is not on PATH"}
    version = subprocess.run([executable, "--version"], text=True, capture_output=True, timeout=20)
    help_run = subprocess.run([executable, "--help"], text=True, capture_output=True, timeout=20)
    help_text = help_run.stdout + "\n" + help_run.stderr
    avail_exe = shutil.which("rocprofv3-avail")
    available = None
    if avail_exe:
      proc = subprocess.run([avail_exe, "list"], text=True, capture_output=True, timeout=60)
      available = {"returncode":proc.returncode, "stdout":proc.stdout[-50000:], "stderr":proc.stderr[-10000:]}
    return {"available":True, "executable":executable, "version_returncode":version.returncode,
      "version":(version.stdout+version.stderr).strip(), "supports_att":"--att" in help_text,
      "supports_kernel_regex":"--kernel-include-regex" in help_text, "supports_pmc":"--pmc" in help_text,
      "supports_output_directory":"--output-directory" in help_text or " -d" in help_text,
      "available_components":available}

  def _validate_target(self, command: Sequence[str]) -> tuple[str, ...]:
    argv = tuple(str(x) for x in command)
    if not argv: raise ValueError("profile target command must not be empty")
    executable = Path(argv[0]).name
    allowed = {"python", "python3", "pytest", "tinygrad", "radeon-forge"}
    if executable not in allowed: raise ValueError(f"profile target executable {executable!r} is not allowlisted")
    if any("\x00" in part for part in argv): raise ValueError("profile command contains NUL")
    return argv

  def _profiler_command(self, request: CaptureRequest, output: Path, help_text: str) -> tuple[str, ...]:
    executable = self.executable()
    if executable is None: raise RuntimeError("rocprofv3 is not installed")
    args: list[str] = [executable]
    if request.kind is CaptureKind.ATT:
      if "--att" not in help_text: raise RuntimeError("installed rocprofv3 does not advertise ATT support")
      args.append("--att")
      if "--att-simd-select" in help_text: args += ["--att-simd-select", "0x0"]
      if request.kernel_regex:
        if "--kernel-include-regex" not in help_text: raise RuntimeError("installed rocprofv3 cannot filter ATT by kernel regex")
        args += ["--kernel-include-regex", request.kernel_regex]
    elif request.kind is CaptureKind.COUNTERS:
      if not request.counters: raise ValueError("counter capture requires at least one counter")
      if "--pmc" not in help_text: raise RuntimeError("installed rocprofv3 does not advertise --pmc counter collection")
      args += ["--pmc", ",".join(request.counters)]
    else: raise ValueError(request.kind)
    if "--output-directory" in help_text: args += ["--output-directory", str(output)]
    else: args += ["-d", str(output)]
    return tuple(args + ["--"] + list(request.command))

  @staticmethod
  def _artifacts(root: Path) -> tuple[CaptureArtifact, ...]:
    ret = []
    for path in sorted(root.rglob("*")):
      if not path.is_file(): continue
      data = path.read_bytes()
      ret.append(CaptureArtifact(str(path), len(data), hashlib.sha256(data).hexdigest(), path.suffix.lower()))
    return tuple(ret)

  @staticmethod
  def _csv_summary(root: Path) -> dict[str, Any]:
    files, rows, columns = 0, 0, set()
    numeric: dict[str, list[float]] = {}
    for path in root.rglob("*.csv"):
      files += 1
      try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
          reader = csv.DictReader(handle)
          columns.update(reader.fieldnames or ())
          for row in reader:
            rows += 1
            for key, value in row.items():
              try: numeric.setdefault(str(key), []).append(float(value))
              except (TypeError, ValueError): pass
      except OSError: continue
    aggregate = {key:{"count":len(values), "min":min(values), "max":max(values), "mean":sum(values)/len(values)}
                 for key, values in numeric.items() if values}
    return {"csv_files":files, "csv_rows":rows, "columns":sorted(columns), "numeric_columns":aggregate}

  def capture(self, request: CaptureRequest) -> CaptureResult:
    target = self._validate_target(request.command)
    capture_id = f"{request.kind.value}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:10]}"
    output = self.evidence_root / capture_id
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "forge_capture_manifest.json"
    started = time.perf_counter_ns()
    returncode: int | None = None
    stdout = stderr = error = ""
    profiler_command: tuple[str, ...] = ()
    try:
      executable = self.executable()
      if executable is None: raise RuntimeError("rocprofv3 is not installed")
      help_run = subprocess.run([executable, "--help"], text=True, capture_output=True, timeout=20)
      help_text = help_run.stdout + "\n" + help_run.stderr
      profiler_command = self._profiler_command(request, output, help_text)
      env = {**os.environ, "PYTHONUNBUFFERED":"1", "RADEON_FORGE_CAPTURE_ID":capture_id,
             "RADEON_FORGE_EXECUTION_STAGE":request.stage, "RADEON_FORGE_TRACE_ID":request.trace_id}
      proc = subprocess.run(profiler_command, cwd=self.project_root, env=env, text=True, capture_output=True,
                            timeout=max(1, min(request.timeout_seconds, 3600)))
      returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except Exception as exc: error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    artifacts = tuple(x for x in self._artifacts(output) if Path(x.path) != manifest_path)
    summary = {"stage":request.stage, "session_id":request.session_id, "trace_id":request.trace_id,
      "agent_step":request.agent_step, "kernel_regex":request.kernel_regex, "counters":list(request.counters),
      "capture_metadata":dict(request.metadata), "parsed_csv":self._csv_summary(output)}
    passed = returncode == 0 and not error and bool(artifacts)
    result = CaptureResult(capture_id, request.kind.value, passed, str(output), profiler_command, target, returncode,
                           elapsed_ms, stdout[-50000:], stderr[-50000:], artifacts, summary, error)
    manifest_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return result
