import tempfile
import unittest
from pathlib import Path

from extra.radeon_forge.workloads.block_hook_harness import HarnessConfig, benchmark, validate


IDENTITY = '''
PARAMETERS = {}
def configure(parameters):
  global PARAMETERS
  PARAMETERS = dict(parameters)
def build_replacement(original, layer_index, model, context):
  assert context["parameters"] == PARAMETERS
  return original
'''

BAD = '''
def build_replacement(original, layer_index, model, context):
  def replacement(x, start_pos, freqs_cis, mask):
    return x * 0
  return replacement
'''


class TestBlockHookHarness(unittest.TestCase):
  def test_identity_replacement_passes_transition_and_benchmark(self):
    with tempfile.TemporaryDirectory() as directory:
      path=Path(directory)/"identity.py"
      path.write_text(IDENTITY,encoding="utf-8")
      config=HarnessConfig(dim=32,hidden_dim=64,n_heads=4,n_kv_heads=4,max_context=32,prompt_tokens=2,
                           max_abs_error=1e-5,max_rel_error=1e-4)
      result=validate(path,config,(1,2,4),{"WAVES":2})
      self.assertTrue(result["passed"],result)
      self.assertEqual(len(result["cases"]),3)
      self.assertGreater(result["checked_values"],0)
      samples,metrics=benchmark(path,config,2,{"WAVES":2})
      self.assertEqual(len(samples),2)
      self.assertTrue(all(value>0 for value in samples))
      self.assertEqual(metrics["stage"],"decode")
      self.assertEqual(metrics["parameters"],{"WAVES":2})

  def test_incorrect_replacement_fails_reference_oracle(self):
    with tempfile.TemporaryDirectory() as directory:
      path=Path(directory)/"bad.py"
      path.write_text(BAD,encoding="utf-8")
      config=HarnessConfig(dim=32,hidden_dim=64,n_heads=4,n_kv_heads=4,max_context=16,prompt_tokens=2,
                           max_abs_error=1e-5,max_rel_error=1e-4)
      result=validate(path,config,(2,),{})
      self.assertFalse(result["passed"])
      self.assertGreater(result["max_abs_error"],config.max_abs_error)
      self.assertIn("diverged",result["reason"])


if __name__=="__main__": unittest.main()
