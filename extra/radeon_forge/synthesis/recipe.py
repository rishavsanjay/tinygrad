from __future__ import annotations

import base64, hashlib, json, shutil, tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .workspace import CandidateRecord, CandidateWorkspace, KernelSpec


FORMAT_VERSION = 1
MAX_RECIPE_BYTES = 8_000_000
MAX_ARTIFACT_BYTES = 4_000_000
_ALLOWED_ROLES = {"oracle", "benchmark", "heldout", "knowledge", "reference", "seed", "implementation_cache", "support"}


def _canonical_hash(payload: Any) -> str:
  return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
  if value is None: return ()
  if not isinstance(value, list) or not all(isinstance(x, str) for x in value): raise ValueError(f"{field_name} must be an array of strings")
  return tuple(value)


def _safe_relative_path(value: str) -> str:
  path = PurePosixPath(value)
  if not value or path.is_absolute() or ".." in path.parts or "." in path.parts or "\x00" in value:
    raise ValueError(f"unsafe recipe artifact path: {value!r}")
  return path.as_posix()


def _command(value: Any, field_name: str) -> tuple[str, ...]:
  command = _string_list(value, field_name)
  if any("\x00" in part for part in command): raise ValueError(f"{field_name} contains NUL")
  return command


@dataclass(frozen=True)
class RecipeArtifact:
  path: str
  role: str
  content: str
  executable: bool = False
  description: str = ""

  @property
  def sha256(self) -> str: return hashlib.sha256(self.content.encode()).hexdigest()


@dataclass(frozen=True)
class ForgeRecipe:
  """Portable optimization knowledge.

  A recipe deliberately captures intent, invariants, oracles and optional seed
  material without defining a kernel DSL. An intelligent executor is expected
  to fill implementation gaps and the independent oracles decide what survives.
  """
  name: str
  description: str
  operation: str
  target: str
  objective: str
  agent_brief: str
  invariants: tuple[str, ...]
  unknowns: tuple[str, ...] = ()
  compatibility: Mapping[str, Any] = field(default_factory=dict)
  acceptance: Mapping[str, Any] = field(default_factory=dict)
  shapes: Mapping[str, int | str] = field(default_factory=dict)
  dtypes: Mapping[str, str] = field(default_factory=dict)
  language: str = "python"
  extension: str = ".py"
  entrypoint: str = "build_kernel"
  mockgpu_command: tuple[str, ...] = ()
  hardware_command: tuple[str, ...] = ()
  heldout_command: tuple[str, ...] = ()
  artifacts: tuple[RecipeArtifact, ...] = ()
  seed_artifact: str | None = None
  seed_hypothesis: str = "Imported implementation cache; revalidate and specialize locally."
  metadata: Mapping[str, Any] = field(default_factory=dict)
  source_text: str = field(default="", repr=False, compare=False)

  @property
  def recipe_id(self) -> str:
    payload = asdict(self)
    payload.pop("source_text", None)
    return f"{self.name}-{_canonical_hash(payload)[:16]}"

  @classmethod
  def load(cls, path: str | Path) -> ForgeRecipe:
    source_path = Path(path)
    raw = source_path.read_bytes()
    if len(raw) > MAX_RECIPE_BYTES: raise ValueError(f"recipe exceeds {MAX_RECIPE_BYTES} bytes")
    payload = tomllib.loads(raw.decode("utf-8"))
    if int(payload.get("format_version", 0)) != FORMAT_VERSION: raise ValueError(f"unsupported recipe format_version; expected {FORMAT_VERSION}")

    required = ("name", "description", "operation", "target", "objective", "agent_brief", "invariants")
    missing = [key for key in required if key not in payload]
    if missing: raise ValueError(f"recipe missing required fields: {missing}")

    artifacts: list[RecipeArtifact] = []
    seen_paths: set[str] = set()
    for item in payload.get("artifact", []):
      if not isinstance(item, dict): raise ValueError("each [[artifact]] entry must be a table")
      artifact_path = _safe_relative_path(str(item.get("path", "")))
      if artifact_path in seen_paths: raise ValueError(f"duplicate artifact path: {artifact_path}")
      seen_paths.add(artifact_path)
      role = str(item.get("role", "support"))
      if role not in _ALLOWED_ROLES: raise ValueError(f"unsupported artifact role {role!r}")
      has_text, has_b64 = "content" in item, "content_base64" in item
      if has_text == has_b64: raise ValueError(f"artifact {artifact_path} must define exactly one of content or content_base64")
      try: content = str(item["content"]) if has_text else base64.b64decode(str(item["content_base64"]), validate=True).decode("utf-8")
      except Exception as exc: raise ValueError(f"artifact {artifact_path} has invalid UTF-8/base64 content") from exc
      if len(content.encode()) > MAX_ARTIFACT_BYTES: raise ValueError(f"artifact {artifact_path} exceeds {MAX_ARTIFACT_BYTES} bytes")
      expected_sha = item.get("sha256")
      actual_sha = hashlib.sha256(content.encode()).hexdigest()
      if expected_sha is not None and str(expected_sha) != actual_sha: raise ValueError(f"artifact {artifact_path} sha256 mismatch")
      artifacts.append(RecipeArtifact(artifact_path, role, content, bool(item.get("executable", False)), str(item.get("description", ""))))

    seed_artifact = payload.get("seed_artifact")
    if seed_artifact is not None:
      seed_artifact = _safe_relative_path(str(seed_artifact))
      if seed_artifact not in seen_paths: raise ValueError("seed_artifact must reference an embedded artifact")

    oracle = payload.get("oracle", {})
    if not isinstance(oracle, dict): raise ValueError("[oracle] must be a table")
    reserved = {"format_version", "name", "description", "operation", "target", "objective", "agent_brief", "invariants",
                "unknowns", "compatibility", "acceptance", "shapes", "dtypes", "language", "extension", "entrypoint", "oracle",
                "artifact", "seed_artifact", "seed_hypothesis", "metadata"}
    extra = {key: value for key, value in payload.items() if key not in reserved}
    metadata = dict(payload.get("metadata", {}))
    if extra: metadata["unrecognized_recipe_fields"] = extra

    recipe = cls(
      name=str(payload["name"]).strip(), description=str(payload["description"]).strip(), operation=str(payload["operation"]).strip(),
      target=str(payload["target"]).strip(), objective=str(payload["objective"]).strip(), agent_brief=str(payload["agent_brief"]).strip(),
      invariants=_string_list(payload["invariants"], "invariants"), unknowns=_string_list(payload.get("unknowns", []), "unknowns"),
      compatibility=dict(payload.get("compatibility", {})), acceptance=dict(payload.get("acceptance", {})),
      shapes=dict(payload.get("shapes", {})), dtypes={str(k): str(v) for k, v in dict(payload.get("dtypes", {})).items()},
      language=str(payload.get("language", "python")), extension=str(payload.get("extension", ".py")),
      entrypoint=str(payload.get("entrypoint", "build_kernel")),
      mockgpu_command=_command(oracle.get("mockgpu_command", []), "oracle.mockgpu_command"),
      hardware_command=_command(oracle.get("hardware_command", []), "oracle.hardware_command"),
      heldout_command=_command(oracle.get("heldout_command", []), "oracle.heldout_command"), artifacts=tuple(artifacts),
      seed_artifact=seed_artifact, seed_hypothesis=str(payload.get("seed_hypothesis", "Imported implementation cache; revalidate and specialize locally.")),
      metadata=metadata, source_text=raw.decode("utf-8"),
    )
    if not recipe.name or not recipe.agent_brief or not recipe.invariants: raise ValueError("name, agent_brief and at least one invariant are required")
    if not recipe.extension.startswith(".") or "/" in recipe.extension: raise ValueError("extension must be a simple suffix such as .py or .cpp")
    return recipe


@dataclass(frozen=True)
class InstalledRecipe:
  recipe_id: str
  bundle_root: str
  spec_id: str
  seed_candidate_id: str | None
  artifact_hashes: Mapping[str, str]


class RecipeLibrary:
  def __init__(self, workspace: CandidateWorkspace):
    self.workspace = workspace
    self.root = workspace.root / "recipes"
    self.root.mkdir(parents=True, exist_ok=True)

  def install(self, recipe: ForgeRecipe) -> InstalledRecipe:
    bundle = self.root / recipe.recipe_id
    if bundle.exists(): shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    if recipe.source_text: (bundle / "recipe.forge.toml").write_text(recipe.source_text, encoding="utf-8")
    artifact_hashes: dict[str, str] = {}
    for artifact in recipe.artifacts:
      path = (bundle / artifact.path).resolve()
      if bundle.resolve() not in path.parents: raise ValueError(f"artifact escaped bundle: {artifact.path}")
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(artifact.content, encoding="utf-8")
      if artifact.executable: path.chmod(path.stat().st_mode | 0o100)
      artifact_hashes[artifact.path] = artifact.sha256

    metadata = {
      **dict(recipe.metadata), "recipe_id": recipe.recipe_id, "recipe_bundle": str(bundle), "recipe_agent_brief": recipe.agent_brief,
      "recipe_unknowns": list(recipe.unknowns), "recipe_compatibility": dict(recipe.compatibility), "recipe_acceptance": dict(recipe.acceptance),
      "recipe_artifacts": [{"path": x.path, "role": x.role, "sha256": x.sha256, "description": x.description} for x in recipe.artifacts],
      "heldout_command": list(recipe.heldout_command),
    }
    spec = KernelSpec(recipe.name, recipe.operation, recipe.target, recipe.language, recipe.extension, recipe.entrypoint,
                      recipe.shapes, recipe.dtypes, recipe.invariants, recipe.objective, recipe.mockgpu_command,
                      recipe.hardware_command, metadata)
    self.workspace.save_spec(spec)

    seed_candidate: CandidateRecord | None = None
    if recipe.seed_artifact is not None:
      source = (bundle / recipe.seed_artifact).read_text(encoding="utf-8")
      seed_candidate = self.workspace.create_candidate(spec.spec_id, source, recipe.seed_hypothesis)
      seed_candidate = self.workspace.update(seed_candidate.candidate_id, "imported_unverified", {
        "recipe_id": recipe.recipe_id, "seed_artifact": recipe.seed_artifact, "portable_cache": True,
      })

    manifest = {"recipe_id": recipe.recipe_id, "spec_id": spec.spec_id, "bundle_root": str(bundle),
                "seed_candidate_id": seed_candidate.candidate_id if seed_candidate else None, "artifact_hashes": artifact_hashes}
    (bundle / "installed.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return InstalledRecipe(**manifest)

  def install_file(self, path: str | Path) -> InstalledRecipe: return self.install(ForgeRecipe.load(path))

  def installed(self) -> list[InstalledRecipe]:
    ret: list[InstalledRecipe] = []
    for path in sorted(self.root.glob("*/installed.json")):
      try: ret.append(InstalledRecipe(**json.loads(path.read_text(encoding="utf-8"))))
      except Exception: continue
    return ret

  def inspect(self, recipe_id: str) -> dict[str, Any]:
    path = self.root / recipe_id / "installed.json"
    if not path.is_file(): raise KeyError(f"unknown recipe {recipe_id}")
    installed = InstalledRecipe(**json.loads(path.read_text(encoding="utf-8")))
    spec = self.workspace.load_spec(installed.spec_id)
    return {"installed": asdict(installed), "spec": asdict(spec) | {"spec_id": spec.spec_id},
            "recipe_source": str(self.root / recipe_id / "recipe.forge.toml")}


def _toml_string(value: str) -> str: return json.dumps(value, ensure_ascii=False)
def _toml_array(values: Sequence[str]) -> str: return "[" + ", ".join(_toml_string(x) for x in values) + "]"


def export_recipe(workspace: CandidateWorkspace, spec_id: str, output: str | Path, candidate_id: str | None = None) -> Path:
  """Export a spec and optional implementation cache as one portable TOML file.

  Exact generated code is base64-encoded because it is a disposable cache. The
  intent, invariants and agent brief stay human-readable and reviewable.
  """
  spec = workspace.load_spec(spec_id)
  metadata = dict(spec.metadata)
  agent_brief = str(metadata.get("recipe_agent_brief") or
                    "Use the supplied intent, invariants and oracle as the contract. Inspect the local hardware and workload, then freely regenerate or restructure the implementation. Do not trust the cached implementation without rerunning every oracle.")
  lines = [f"format_version = {FORMAT_VERSION}", f"name = {_toml_string(spec.name)}",
           f"description = {_toml_string(str(metadata.get('description', spec.operation)))}", f"operation = {_toml_string(spec.operation)}",
           f"target = {_toml_string(spec.target)}", f"objective = {_toml_string(spec.objective)}", f"agent_brief = {_toml_string(agent_brief)}",
           f"invariants = {_toml_array(spec.invariants)}", f"language = {_toml_string(spec.language)}",
           f"extension = {_toml_string(spec.extension)}", f"entrypoint = {_toml_string(spec.entrypoint)}"]
  unknowns = tuple(str(x) for x in metadata.get("recipe_unknowns", ()))
  if unknowns: lines.append(f"unknowns = {_toml_array(unknowns)}")
  if candidate_id is not None:
    candidate = workspace.load_candidate(candidate_id)
    if candidate.spec_id != spec_id: raise ValueError("candidate does not belong to spec")
    source = Path(candidate.source_path).read_text(encoding="utf-8")
    seed_name = f"implementation-cache{spec.extension}"
    lines += [f"seed_artifact = {_toml_string(seed_name)}", f"seed_hypothesis = {_toml_string(candidate.hypothesis)}"]
  else: candidate, source, seed_name = None, "", ""

  if spec.shapes:
    lines.append("\n[shapes]")
    lines += [f"{json.dumps(str(k))} = {json.dumps(v)}" for k, v in spec.shapes.items()]
  if spec.dtypes:
    lines.append("\n[dtypes]")
    lines += [f"{json.dumps(str(k))} = {_toml_string(str(v))}" for k, v in spec.dtypes.items()]
  compatibility = dict(metadata.get("recipe_compatibility", {}))
  if compatibility:
    lines.append("\n[compatibility]")
    lines += [f"{json.dumps(str(k))} = {json.dumps(v)}" for k, v in compatibility.items()]
  acceptance = dict(metadata.get("recipe_acceptance", {}))
  if acceptance:
    lines.append("\n[acceptance]")
    lines += [f"{json.dumps(str(k))} = {json.dumps(v)}" for k, v in acceptance.items()]
  lines += ["\n[oracle]", f"mockgpu_command = {_toml_array(spec.mockgpu_command)}", f"hardware_command = {_toml_array(spec.hardware_command)}"]
  heldout = tuple(str(x) for x in metadata.get("heldout_command", ()))
  if heldout: lines.append(f"heldout_command = {_toml_array(heldout)}")

  if candidate is not None:
    encoded = base64.b64encode(source.encode()).decode()
    lines += ["\n[[artifact]]", f"path = {_toml_string(seed_name)}", "role = \"implementation_cache\"",
              f"description = {_toml_string('Disposable implementation cache exported from ' + candidate.candidate_id)}",
              f"sha256 = {_toml_string(hashlib.sha256(source.encode()).hexdigest())}", f"content_base64 = {_toml_string(encoded)}"]

  out = Path(output)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return out
