from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from src.tracking.voter import VoteResult


@dataclass(frozen=True)
class Decision:
    log_type: str
    event_type: str | None
    needs_review: bool


def parse_hhmm(value: str) -> time:
    hour, minute = value.split(":", 1)
    return time(hour=int(hour), minute=int(minute))


def is_after_curfew(now: datetime, curfew_time: str, grace_period_minutes: int) -> bool:
    curfew = parse_hhmm(curfew_time)
    curfew_at = datetime.combine(now.date(), curfew)
    return now >= curfew_at + timedelta(minutes=grace_period_minutes)


class EventDecisionEngine:
    def __init__(
        self,
        curfew_enabled: bool = True,
        curfew_time: str = "22:30",
        grace_period_minutes: int = 10,
    ) -> None:
        self.curfew_enabled = curfew_enabled
        self.curfew_time = curfew_time
        self.grace_period_minutes = grace_period_minutes

    def decide(self, vote: VoteResult, now: datetime | None = None) -> Decision:
        current_time = now or datetime.now()

        if vote.result_type == "unknown" and vote.confirmed:
            return Decision("unknown", "stranger", True)

        if vote.result_type == "matched" and vote.confirmed:
            if self.curfew_enabled and is_after_curfew(
                current_time, self.curfew_time, self.grace_period_minutes
            ):
                return Decision("matched", "late_return", True)
            return Decision("matched", None, False)

        if vote.result_type == "suspected":
            return Decision("suspected", None, True)

        return Decision(vote.result_type, None, False)

