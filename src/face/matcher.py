from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.storage.database import EmbeddingRecord


@dataclass(frozen=True)
class MatchResult:
    result_type: str
    person_id: str | None
    person_name: str | None
    similarity: float


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class FaceMatcher:
    def __init__(
        self,
        records: Iterable[EmbeddingRecord],
        confirm_threshold: float = 0.60,
        suspect_threshold: float = 0.40,
    ) -> None:
        self.records = list(records)
        self.confirm_threshold = confirm_threshold
        self.suspect_threshold = suspect_threshold

    def match(self, embedding: Sequence[float]) -> MatchResult:
        if not self.records:
            return MatchResult("unknown", None, None, 0.0)

        best_record: EmbeddingRecord | None = None
        best_score = -1.0
        for record in self.records:
            score = cosine_similarity(embedding, record.embedding)
            if score > best_score:
                best_score = score
                best_record = record

        if best_record is None:
            return MatchResult("unknown", None, None, 0.0)

        if best_score >= self.confirm_threshold:
            result_type = "matched"
        elif best_score >= self.suspect_threshold:
            result_type = "suspected"
        else:
            result_type = "unknown"

        return MatchResult(
            result_type=result_type,
            person_id=best_record.person_id if result_type != "unknown" else None,
            person_name=best_record.person_name if result_type != "unknown" else None,
            similarity=best_score,
        )

