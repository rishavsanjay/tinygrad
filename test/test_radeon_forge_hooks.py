import hashlib
import tempfile
import unittest
from pathlib import Path

from extra.radeon_forge.backends.stage_hooks import ModelStageHookRuntime
from extra.radeon_forge.synthesis.hooks import (ExecutionContext, ExecutionStage, HookRegistry, RuntimeFingerprint,
                                                StagePredicate)
from extra.radeon_forge.synthesis.workspace import CandidateWorkspace, KernelSpec


class FakeModel:
  def __init__(self):
    self.layers = [lambda value: value + 1, lambda value: value + 2]
    self.forward_jit = None


class TestStageAwareHooks(unittest.TestCase):
  def test_predicate_distinguishes_execution_state_and_workload(self):
    predicate = StagePredicate.from_mapping({"stages": ["decode"], "batch_sizes": [1], "min_context_tokens": 1024,
                                             "min_generated_token_index": 1, "prefix_cache": "hit",
                                             "conditions": {"resume_after_tool": True}})
    matching = ExecutionContext(ExecutionStage.DECODE, context_tokens=2048, generated_token_index=4,
                                prefix_reused_tokens=512, warm=True, attributes={"resume_after_tool": True})
    self.assertTrue(predicate.matches(matching))
    self.assertFalse(predicate.matches(ExecutionContext(ExecutionStage.PREFILL, context_tokens=2048,
                                                         prefix_reused_tokens=512, attributes={"resume_after_tool": True})))
    self.assertFalse(predicate.matches(ExecutionContext(ExecutionStage.DECODE, context_tokens=512,
                                                         generated_token_index=4, prefix_reused_tokens=256,
                                                         attributes={"resume_after_tool": True})))
    self.assertFalse(predicate.matches(ExecutionContext(ExecutionStage.DECODE, context_tokens=2048,
                                                         generated_token_index=0, prefix_reused_tokens=256,
                                                         attributes={"resume_after_tool": True})))

  def test_prefill_and_decode_implementations_can_coexist(self):
    with tempfile.TemporaryDirectory() as directory:
      workspace = CandidateWorkspace(directory)
      common = {"layer": "transformer_block", "target": "llama.block", "adapter": "python_transformer_block",
                "exclusive_group": "llama-block"}
      prefill_spec = workspace.save_spec(KernelSpec("prefill-block", "prefill block", target="gfx1100",
        invariants=("match reference",), metadata={"hook": {**common, "when": {"stages": ["prefill"]}}}))
      decode_spec = workspace.save_spec(KernelSpec("decode-block", "decode block", target="gfx1100",
        invariants=("match reference",), metadata={"hook": {**common, "when": {"stages": ["decode"],
                                                        "min_generated_token_index": 1}}}))
      prefill = workspace.create_candidate(prefill_spec.spec_id, "def build_replacement(*args): return args[0]\n", "prefill")
      decode = workspace.create_candidate(decode_spec.spec_id, "def build_replacement(*args): return args[0]\n", "decode")
      workspace.update(prefill.candidate_id, "hardware_passed", {})
      workspace.update(decode.candidate_id, "hardware_passed", {})
      registry = HookRegistry(workspace)
      fingerprint = RuntimeFingerprint(architecture="gfx1100", runtime="tinygrad", model_family="llama")
      registry.activate(prefill.candidate_id, fingerprint, "validated prefill path")
      registry.activate(decode.candidate_id, fingerprint, "validated decode path")
      self.assertEqual(len(registry.active()), 2)
      self.assertEqual(registry.resolve(ExecutionContext(ExecutionStage.PREFILL))[0].candidate_id, prefill.candidate_id)
      self.assertEqual(registry.resolve(ExecutionContext(ExecutionStage.FIRST_TOKEN, generated_token_index=0)), [])
      self.assertEqual(registry.resolve(ExecutionContext(ExecutionStage.DECODE, generated_token_index=2))[0].candidate_id,
                       decode.candidate_id)

  def test_model_adapter_rolls_back_bad_stage_hook(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      good = root / "good.py"
      good.write_text("def build_replacement(original, layer_index, model, context):\n  return lambda value: original(value) * 10\n",
                      encoding="utf-8")
      bad = root / "bad.py"
      bad.write_text("VALUE = 1\n", encoding="utf-8")
      descriptor = {"layer": "transformer_block", "target": "llama.block", "mode": "replace",
                    "adapter": "python_transformer_block", "selector": {"indices": [0]},
                    "when": {"stages": ["decode"]}, "priority": 0, "exclusive_group": "block", "description": ""}
      def hook(path, activation):
        return {"activation_id": activation, "candidate_id": activation, "spec_id": activation,
                "source_path": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "descriptor": descriptor}

      model = FakeModel()
      runtime = ModelStageHookRuntime(model)
      baseline = model.layers[0](2)
      applied = runtime.apply([hook(good, "good")], ExecutionContext(ExecutionStage.DECODE, generated_token_index=2))
      self.assertEqual(applied["active"], ["good"])
      self.assertEqual(model.layers[0](2), baseline * 10)
      rolled_back = runtime.apply([hook(bad, "bad")], ExecutionContext(ExecutionStage.DECODE, generated_token_index=2))
      self.assertTrue(rolled_back["rolled_back"])
      self.assertEqual(model.layers[0](2), baseline)


if __name__ == "__main__": unittest.main()
