import unittest

from extra.radeon_forge.backends.kv_state import KVReuseLedger


class TestKVReuseLedger(unittest.TestCase):
  def test_emitted_but_unprocessed_token_is_not_reported_as_cache_hit(self):
    ledger = KVReuseLedger()
    prompt = [1, 2, 3]
    self.assertEqual(ledger.begin("session-a", prompt), 0)
    ledger.commit_prefill(prompt[:-1])
    ledger.commit_decode_input(prompt[-1], expected_position=2)
    self.assertEqual(ledger.cached_tokens, [1, 2, 3])

    # Token 4 was emitted to the user but generation stopped before it became
    # the input to another decode step. It is intentionally absent from KV.
    next_prompt = [1, 2, 3, 4, 5]
    self.assertEqual(ledger.begin("session-a", next_prompt), 3)
    ledger.commit_prefill(next_prompt[:-1])
    self.assertEqual(ledger.cached_tokens, [1, 2, 3, 4])

  def test_session_switch_invalidates_single_resident_cache(self):
    ledger = KVReuseLedger("session-a", [1, 2, 3])
    self.assertEqual(ledger.begin("session-b", [1, 2, 3]), 0)
    self.assertEqual(ledger.active_session, "session-b")
    self.assertEqual(ledger.cached_tokens, [])

  def test_decode_position_mismatch_fails_closed(self):
    ledger = KVReuseLedger("session-a", [1, 2])
    with self.assertRaisesRegex(RuntimeError, "position mismatch"):
      ledger.commit_decode_input(3, expected_position=4)
    self.assertEqual(ledger.cached_tokens, [1, 2])


if __name__ == "__main__": unittest.main()
