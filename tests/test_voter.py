import unittest

from src.face.matcher import MatchResult
from src.tracking.voter import MultiFrameVoter


class VoterTest(unittest.TestCase):
    def test_confirms_same_person_after_enough_votes(self):
        voter = MultiFrameVoter(window=5, confirm_count=3)
        result = None
        for _ in range(3):
            result = voter.add("track-1", MatchResult("matched", "STU1", "Alice", 0.91))

        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.result_type, "matched")
        self.assertEqual(result.person_id, "STU1")

    def test_confirms_unknown_after_enough_votes(self):
        voter = MultiFrameVoter(window=5, confirm_count=3)
        result = None
        for _ in range(3):
            result = voter.add("track-2", MatchResult("unknown", None, None, 0.1))

        self.assertIsNotNone(result)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.result_type, "unknown")


if __name__ == "__main__":
    unittest.main()

