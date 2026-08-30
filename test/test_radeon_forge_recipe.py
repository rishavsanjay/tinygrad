import base64
import tempfile
import unittest
from pathlib import Path

from extra.radeon_forge.synthesis import (CandidateWorkspace, ForgeRecipe, HookDescriptor, KernelSpec, RecipeLibrary,
                                          export_recipe, export_recipe_with_hook)


class TestForgeRecipe(unittest.TestCase):
  def _recipe_text(self, seed: str = "print('seed')\n") -> str:
    encoded = base64.b64encode(seed.encode()).decode()
    return f'''format_version = 1
name = "portable-vopd"
description = "One-file optimization knowledge"
operation = "Tune an RDNA3 instruction schedule"
target = "gfx1100"
objective = "minimize P95 latency"
agent_brief = "Inspect evidence, freely regenerate the implementation, and trust only the oracle."
invariants = ["match reference", "never use MockGPU timing"]
unknowns = ["best schedule is hardware dependent"]
seed_artifact = "seed.py"

[compatibility]
architecture = "gfx1100"

[acceptance]
maximum_error = 0.000001

[metadata.hook]
layer = "kernel"
target = "projection_gemm"
mode = "replace"
adapter = "request_metadata"
exclusive_group = "projection"

[metadata.hook.when]
stages = ["decode"]
batch_sizes = [1]
min_context_tokens = 1024
prefix_cache = "hit"

[metadata.search]
budgets = [3, 10]
reduction = 2
max_candidates = 16

[metadata.search.axes]
WAVES = [2, 4]
STAGES = [1, 2]

[oracle]
mockgpu_command = ["python3", "{{candidate}}", "--bundle", "{{bundle}}"]
hardware_command = ["python3", "{{candidate}}", "--benchmark"]

[[artifact]]
path = "seed.py"
role = "implementation_cache"
sha256 = "{__import__('hashlib').sha256(seed.encode()).hexdigest()}"
content_base64 = "{encoded}"

[[artifact]]
path = "knowledge/notes.md"
role = "knowledge"
content = "The implementation is disposable."
'''

  def test_install_is_content_addressed_and_seed_is_unverified(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source_a, source_b = root / "a.forge.toml", root / "b.forge.toml"
      source_a.write_text(self._recipe_text(), encoding="utf-8")
      source_b.write_text(self._recipe_text(), encoding="utf-8")
      recipe_a, recipe_b = ForgeRecipe.load(source_a), ForgeRecipe.load(source_b)
      self.assertEqual(recipe_a.recipe_id, recipe_b.recipe_id)

      workspace = CandidateWorkspace(root / "library")
      installed = RecipeLibrary(workspace).install(recipe_a)
      self.assertTrue((Path(installed.bundle_root) / "knowledge/notes.md").is_file())
      self.assertIsNotNone(installed.seed_candidate_id)
      candidate = workspace.load_candidate(installed.seed_candidate_id)
      self.assertEqual(candidate.status, "imported_unverified")
      self.assertEqual(Path(candidate.source_path).read_text(encoding="utf-8"), "print('seed')\n")

      spec = workspace.load_spec(installed.spec_id)
      descriptor = HookDescriptor.from_spec(spec)
      self.assertEqual(descriptor.layer.value, "kernel")
      self.assertEqual([x.value for x in descriptor.when.stages], ["decode"])
      self.assertEqual(descriptor.when.min_context_tokens, 1024)
      self.assertEqual(descriptor.when.prefix_cache, "hit")
      self.assertEqual(spec.metadata["search"]["axes"]["WAVES"], [2, 4])
      rendered = workspace.render_command(spec.mockgpu_command, spec, candidate, root)
      self.assertEqual(rendered[1], candidate.source_path)
      self.assertEqual(rendered[3], installed.bundle_root)

  def test_rejects_path_traversal(self):
    text = self._recipe_text().replace('path = "seed.py"', 'path = "../seed.py"', 1).replace('seed_artifact = "seed.py"', 'seed_artifact = "../seed.py"')
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "bad.forge.toml"
      path.write_text(text, encoding="utf-8")
      with self.assertRaises(ValueError): ForgeRecipe.load(path)

  def test_export_import_round_trip_preserves_contract_cache_stage_and_search(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source_workspace = CandidateWorkspace(root / "source")
      spec = KernelSpec(
        name="roundtrip", operation="specialize batch-one decode", target="gfx1100", language="python", extension=".py",
        invariants=("match reference", "no remote API"), objective="minimize p95 token latency",
        mockgpu_command=("python3", "{candidate}"), hardware_command=("python3", "{candidate}", "--benchmark"),
        metadata={"recipe_agent_brief":"Use the oracle as the contract and freely rewrite the implementation.",
                  "recipe_compatibility":{"architecture":"gfx1100"}, "recipe_acceptance":{"max_error":1e-6},
                  "search":{"axes":{"WAVES":[2,4], "UNROLL":[1,2]}, "budgets":[3,10,30], "reduction":2,
                            "max_candidates":16},
                  "hook":{"layer":"transformer_block", "target":"llama.decode.block", "mode":"replace",
                          "adapter":"python_transformer_block", "selector":{"indices":[0,1]},
                          "when":{"stages":["decode"], "batch_sizes":[1], "min_generated_token_index":1,
                                  "conditions":{"resume_after_tool":False}},
                          "exclusive_group":"decode-block"}},
      )
      source_workspace.save_spec(spec)
      candidate = source_workspace.create_candidate(spec.spec_id, "print('candidate')\n", "measured decode implementation")
      source_workspace.update(candidate.candidate_id, "hardware_passed", {"selected_parameters":{"WAVES":4, "UNROLL":2}})
      exported = export_recipe_with_hook(source_workspace, spec.spec_id, root / "shared.forge.toml", candidate.candidate_id)

      loaded = ForgeRecipe.load(exported)
      self.assertEqual(loaded.target, "gfx1100")
      self.assertEqual(loaded.invariants, spec.invariants)
      self.assertEqual(loaded.metadata["hook"]["when"]["stages"], ["decode"])
      self.assertEqual(loaded.metadata["search"]["axes"]["WAVES"], [2, 4])
      self.assertEqual(loaded.metadata["exported_winner"], {"WAVES":4, "UNROLL":2})
      destination = CandidateWorkspace(root / "destination")
      installed = RecipeLibrary(destination).install(loaded)
      imported = destination.load_candidate(installed.seed_candidate_id)
      self.assertEqual(Path(imported.source_path).read_text(encoding="utf-8"), "print('candidate')\n")
      self.assertEqual(imported.status, "imported_unverified")
      installed_spec = destination.load_spec(installed.spec_id)
      descriptor = HookDescriptor.from_spec(installed_spec)
      self.assertEqual(descriptor.target, "llama.decode.block")
      self.assertEqual([x.value for x in descriptor.when.stages], ["decode"])
      self.assertEqual(descriptor.when.min_generated_token_index, 1)
      self.assertEqual(descriptor.when.conditions["resume_after_tool"], False)
      self.assertEqual(installed_spec.metadata["search"]["axes"]["UNROLL"], [1, 2])
      self.assertEqual(installed_spec.metadata["exported_winner"]["WAVES"], 4)

  def test_base_export_remains_valid_without_hook_extension(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      workspace = CandidateWorkspace(root / "source")
      spec = workspace.save_spec(KernelSpec("plain", "plain operation", invariants=("correct",)))
      path = export_recipe(workspace, spec.spec_id, root / "plain.forge.toml")
      self.assertEqual(ForgeRecipe.load(path).name, "plain")


if __name__ == "__main__": unittest.main()
