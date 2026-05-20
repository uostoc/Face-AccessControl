from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from src.face.matcher import MatchResult


@dataclass(frozen=True)
class VoteResult:
    result_type: str
    person_id: str | None
    person_name: str | None
    similarity: float
    confirmed: bool


class MultiFrameVoter:
    def __init__(self, window: int = 5, confirm_count: int = 3) -> None:
        self.window = window
        self.confirm_count = confirm_count
        self._history: dict[str, deque[MatchResult]] = {}

    def add(self, track_id: str, result: MatchResult) -> VoteResult:
        history = self._history.setdefault(track_id, deque(maxlen=self.window))
        history.append(result)

        matched = [item for item in history if item.result_type == "matched" and item.person_id]
        if matched:
            counts = Counter(item.person_id for item in matched)
            person_id, count = counts.most_common(1)[0]
            if count >= self.confirm_count:
                candidates = [item for item in matched if item.person_id == person_id]
                best = max(candidates, key=lambda item: item.similarity)
                return VoteResult("matched", best.person_id, best.person_name, best.similarity, True)

        unknown_count = sum(1 for item in history if item.result_type == "unknown")
        if unknown_count >= self.confirm_count:
            best_unknown = max(history, key=lambda item: item.similarity)
            return VoteResult("unknown", None, None, best_unknown.similarity, True)

        latest = history[-1]
        return VoteResult(
            latest.result_type,
            latest.person_id,
            latest.person_name,
            latest.similarity,
            False,
        )

