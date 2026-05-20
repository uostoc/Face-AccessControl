import unittest

from src.tracking.tracker import CentroidTracker


class TrackerTest(unittest.TestCase):
    def test_assigns_stable_track_for_nearby_bbox(self):
        tracker = CentroidTracker(max_distance=50)
        first = tracker.assign([(10, 10, 50, 50)])
        second = tracker.assign([(15, 15, 55, 55)])

        self.assertEqual(first[0], second[0])


if __name__ == "__main__":
    unittest.main()

