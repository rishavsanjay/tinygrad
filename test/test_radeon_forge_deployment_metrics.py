import tempfile
import unittest

from extra.radeon_forge.synthesis.deployment import SafeHookRegistry
from extra.radeon_forge.synthesis.hooks import HookActivationError,RuntimeFingerprint
from extra.radeon_forge.synthesis.workspace import CandidateWorkspace,KernelSpec


SOURCE='''
def build_replacement(original, layer_index, model, context):
  return original
'''


class TestDeploymentMetricGate(unittest.TestCase):
  def test_stateful_candidate_requires_transition_and_agent_metric(self):
    with tempfile.TemporaryDirectory() as directory:
      workspace=CandidateWorkspace(directory)
      spec=workspace.save_spec(KernelSpec("decode","block",target="gfx1100",invariants=("match",),
        metadata={"heldout_command":["python3","transition.py"],"require_agent_evaluation":True,
          "hook":{"layer":"transformer_block","target":"llama.block","adapter":"python_transformer_block",
                  "when":{"stages":["first_token","decode"]}}}))
      candidate=workspace.create_candidate(spec.spec_id,SOURCE,"faster block")
      workspace.update(candidate.candidate_id,"heldout_passed",{"heldout":{"passed":True}})
      registry=SafeHookRegistry(workspace)
      fingerprint=RuntimeFingerprint(architecture="gfx1100",runtime="tinygrad",model_family="llama")
      with self.assertRaisesRegex(HookActivationError,"frozen agent-suite"):
        registry.activate(candidate.candidate_id,fingerprint,"deploy")

      workspace.update(candidate.candidate_id,"evaluation_failed",{"agent_evaluation_comparison":{"gate":{"passed":False}}})
      with self.assertRaisesRegex(HookActivationError,"frozen agent-suite"):
        registry.activate(candidate.candidate_id,fingerprint,"deploy")

      workspace.update(candidate.candidate_id,"evaluation_passed",{"agent_evaluation_comparison":{"gate":{"passed":True}}})
      active=registry.activate(candidate.candidate_id,fingerprint,"quality and latency gates passed")
      self.assertEqual(active.candidate_id,candidate.candidate_id)

  def test_stateless_non_final_hook_keeps_lighter_policy(self):
    with tempfile.TemporaryDirectory() as directory:
      workspace=CandidateWorkspace(directory)
      spec=workspace.save_spec(KernelSpec("advisory","block",target="gfx1100",invariants=("match",),
        metadata={"heldout_command":["python3","transition.py"],
          "hook":{"layer":"transformer_block","target":"llama.block","adapter":"python_transformer_block",
                  "when":{"stages":["decode"]}}}))
      candidate=workspace.create_candidate(spec.spec_id,SOURCE,"validated")
      workspace.update(candidate.candidate_id,"heldout_passed",{})
      active=SafeHookRegistry(workspace).activate(candidate.candidate_id,
        RuntimeFingerprint(architecture="gfx1100"),"transition passed")
      self.assertEqual(active.candidate_id,candidate.candidate_id)


if __name__=="__main__":unittest.main()
