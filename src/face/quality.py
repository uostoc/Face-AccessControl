from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    reason: str
    score: float


class FaceQualityEvaluator:
    def __init__(self, min_area: int = 80 * 80, min_score: float = 0.5) -> None:
        self.min_area = min_area
        self.min_score = min_score

    def evaluate(self, bbox: tuple[int, int, int, int], detector_score: float) -> QualityResult:
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < self.min_area:
            return QualityResult(False, "face_too_small", detector_score)
        if detector_score < self.min_score:
            return QualityResult(False, "low_detector_score", detector_score)
        return QualityResult(True, "accepted", detector_score)

