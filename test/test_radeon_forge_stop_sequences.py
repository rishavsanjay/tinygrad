import unittest

from extra.radeon_forge.runtime.stop_sequences import StopSequenceMatcher


class TestStopSequenceMatcher(unittest.TestCase):
  def test_stop_split_across_tokens_is_not_emitted(self):
    matcher = StopSequenceMatcher(["</final>"])
    self.assertEqual(matcher.feed("answer</fi").text, "answer")
    matched = matcher.feed("nal>ignored")
    self.assertEqual(matched.text, "")
    self.assertEqual(matched.matched, "</final>")
    self.assertEqual(matcher.finalize(), "")

  def test_ambiguous_prefix_is_released_when_it_stops_matching(self):
    matcher = StopSequenceMatcher(["STOP"])
    self.assertEqual(matcher.feed("hello ST").text, "hello ")
    self.assertEqual(matcher.feed("AR").text, "STAR")
    self.assertEqual(matcher.finalize(), "")

  def test_earliest_stop_wins(self):
    matcher = StopSequenceMatcher(["END", "STOP"])
    result = matcher.feed("one STOP two END")
    self.assertEqual(result.text, "one ")
    self.assertEqual(result.matched, "STOP")

  def test_finalize_releases_safe_suffix(self):
    matcher = StopSequenceMatcher(["STOP"])
    self.assertEqual(matcher.feed("value ST").text, "value ")
    self.assertEqual(matcher.finalize(), "ST")

  def test_empty_stop_is_rejected(self):
    with self.assertRaises(ValueError): StopSequenceMatcher([""])


if __name__ == "__main__": unittest.main()
