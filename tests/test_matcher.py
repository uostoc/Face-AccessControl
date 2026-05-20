import unittest

from src.face.matcher import FaceMatcher, cosine_similarity
from src.storage.database import EmbeddingRecord


class MatcherTest(unittest.TestCase):
    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_matcher_classifies_thresholds(self):
        matcher = FaceMatcher(
            [
                EmbeddingRecord(
                    person_id="STU1",
                    person_name="Alice",
                    embedding=[1.0, 0.0],
                    angle="front",
                    quality_score=1.0,
                    model_name="test",
                )
            ],
            confirm_threshold=0.8,
            suspect_threshold=0.4,
        )

        matched = matcher.match([1.0, 0.0])
        self.assertEqual(matched.result_type, "matched")
        self.assertEqual(matched.person_id, "STU1")

        suspected = matcher.match([0.5, 0.8660254])
        self.assertEqual(suspected.result_type, "suspected")
        self.assertEqual(suspected.person_id, "STU1")

        unknown = matcher.match([0.0, 1.0])
        self.assertEqual(unknown.result_type, "unknown")
        self.assertIsNone(unknown.person_id)


if __name__ == "__main__":
    unittest.main()

