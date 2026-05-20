from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Track:
    track_id: str
    center: tuple[float, float]
    missed: int = 0


class CentroidTracker:
    def __init__(self, max_distance: float = 120.0, max_missed: int = 5) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._tracks: dict[str, _Track] = {}
        self._next_id = 1

    def assign(self, bboxes: list[tuple[int, int, int, int]]) -> list[str]:
        centers = [_center(bbox) for bbox in bboxes]
        assigned_track_ids: list[str] = []
        used_tracks: set[str] = set()

        for center in centers:
            best_track: _Track | None = None
            best_distance = self.max_distance
            for track in self._tracks.values():
                if track.track_id in used_tracks:
                    continue
                distance = _distance(center, track.center)
                if distance < best_distance:
                    best_distance = distance
                    best_track = track

            if best_track is None:
                track_id = f"track-{self._next_id}"
                self._next_id += 1
                self._tracks[track_id] = _Track(track_id=track_id, center=center)
                assigned_track_ids.append(track_id)
                used_tracks.add(track_id)
            else:
                best_track.center = center
                best_track.missed = 0
                assigned_track_ids.append(best_track.track_id)
                used_tracks.add(best_track.track_id)

        for track_id in list(self._tracks):
            if track_id not in used_tracks:
                self._tracks[track_id].missed += 1
                if self._tracks[track_id].missed > self.max_missed:
                    del self._tracks[track_id]

        return assigned_track_ids


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5

