import unittest
from datetime import datetime

from src.events.decision import EventDecisionEngine, is_after_curfew
from src.tracking.voter import VoteResult


class DecisionTest(unittest.TestCase):
    def test_curfew_with_grace_period(self):
        self.assertFalse(is_after_curfew(datetime(2026, 5, 15, 22, 35), "22:30", 10))
        self.assertTrue(is_after_curfew(datetime(2026, 5, 15, 22, 40), "22:30", 10))

    def test_late_return_event_for_confirmed_match_after_curfew(self):
        engine = EventDecisionEngine(curfew_enabled=True, curfew_time="22:30", grace_period_minutes=10)
        decision = engine.decide(
            VoteResult("matched", "STU1", "Alice", 0.9, True),
            now=datetime(2026, 5, 15, 22, 45),
        )
        self.assertEqual(decision.event_type, "late_return")
        self.assertTrue(decision.needs_review)

    def test_stranger_event_for_confirmed_unknown(self):
        engine = EventDecisionEngine()
        decision = engine.decide(VoteResult("unknown", None, None, 0.0, True))
        self.assertEqual(decision.event_type, "stranger")
        self.assertTrue(decision.needs_review)


if __name__ == "__main__":
    unittest.main()

