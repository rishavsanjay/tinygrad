import sys
import unittest
from pathlib import Path

from extra.radeon_forge.backends.plugins import WorkerPlugin, WorkerPluginRegistry, default_worker_plugins


class TestWorkerPlugins(unittest.TestCase):
  def test_legacy_llama_command_is_local_and_explicit(self):
    plugin=default_worker_plugins().get("legacy-llama")
    command=plugin.command({"model":Path("/models/local.gguf"),"tokenizer":Path("/models/tokenizer.model"),
                            "size":"1B","max_context":4096,"seed":7})
    self.assertEqual(command[:3],[sys.executable,"-m","extra.radeon_forge.backends.tinygrad_llama_worker"])
    self.assertIn("/models/local.gguf",command)
    self.assertIn("--max-context",command)
    self.assertIn("4096",command)
    self.assertNotIn("http", " ".join(command))

  def test_required_and_unknown_fields_fail_closed(self):
    plugin=default_worker_plugins().get("legacy-llama")
    with self.assertRaisesRegex(ValueError,"missing required"):
      plugin.command({})
    with self.assertRaisesRegex(ValueError,"unsupported fields"):
      plugin.command({"model":"x","remote_api":"https://example.com"})

  def test_registry_rejects_external_modules_and_duplicates(self):
    registry=WorkerPluginRegistry()
    with self.assertRaisesRegex(ValueError,"in-tree local"):
      registry.register(WorkerPlugin("remote","third_party.worker","bad",("model",)))
    local=WorkerPlugin("local","extra.radeon_forge.backends.worker","ok",("model",))
    registry.register(local)
    with self.assertRaisesRegex(ValueError,"duplicate"):
      registry.register(local)

  def test_discovery_is_stable(self):
    rows=default_worker_plugins().list()
    self.assertEqual([row["name"] for row in rows],["legacy-llama"])
    self.assertIn("stage hooks",rows[0]["description"])


if __name__=="__main__": unittest.main()
