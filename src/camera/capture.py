from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class CameraSettings:
    source: int | str = 0
    fps: int = 30
    width: int = 1280
    height: int = 720


class CameraCapture:
    def __init__(self, settings: CameraSettings) -> None:
        self.settings = settings
        self._capture: Any | None = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to use camera capture") from exc

        self._capture = cv2.VideoCapture(self.settings.source)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.settings.fps)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.settings.source}")

    def frames(self) -> Iterator[Any]:
        if self._capture is None:
            self.open()
        assert self._capture is not None

        while True:
            ok, frame = self._capture.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")
            yield frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

