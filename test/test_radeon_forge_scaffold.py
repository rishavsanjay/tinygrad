import ast
import tempfile
import unittest
from pathlib import Path

from extra.radeon_forge.synthesis.scaffold import render_candidate_scaffold
from extra.radeon_forge.synthesis.workspace import KernelSpec


class TestCandidateScaffold(unittest.TestCase):
  def test_transformer_block_scaffold_is_executable_and_preserves_contract(self):
    spec=KernelSpec("decode-block","specialize decode",target="gfx1100",invariants=("match reference","preserve KV"),
      metadata={"search":{"axes":{"WAVES":[1,2,4]}},"hook":{"layer":"transformer_block","target":"llama.decode.block",
        "mode":"replace","adapter":"python_transformer_block","selector":{"indices":"all"},
        "when":{"stages":["first_token","decode"],"batch_sizes":[1]}}})
    source=render_candidate_scaffold(spec)
    ast.parse(source)
    namespace={}
    exec(compile(source,"candidate.py","exec"),namespace)
    namespace["configure"]({"WAVES":4})
    self.assertEqual(namespace["RADEON_FORGE_PARAMETERS"],{"WAVES":4})
    original=lambda x,start_pos,freqs,mask:x+1
    replacement=namespace["build_replacement"](original,0,object(),{"stage":"decode"})
    self.assertEqual(replacement(2,0,None,None),3)
    self.assertIn("first_token|decode",source)
    self.assertIn("preserve KV",source)
    self.assertIn('"WAVES"',source)

  def test_non_block_scaffold_does_not_invent_a_dsl(self):
    spec=KernelSpec("gemv","direct gemv",invariants=("correct",),metadata={"hook":{"layer":"kernel","target":"gemv"}})
    source=render_candidate_scaffold(spec)
    ast.parse(source)
    self.assertIn("def build_kernel",source)
    self.assertIn("NotImplementedError",source)
    self.assertNotIn("class Tile",source)


if __name__=="__main__": unittest.main()
