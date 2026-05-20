from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.onnx_runtime import preload_onnxruntime_dlls, resolve_providers


@dataclass(frozen=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]
    embedding: list[float]
    quality_score: float


class InsightFaceRecognizer:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: int = 320,
        providers: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.det_size = det_size
        self.providers = resolve_providers(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._app: Any | None = None

    def load(self) -> None:
        preload_onnxruntime_dlls()
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "insightface is not installed. Run `pip install -r requirements.txt` "
                "to enable real face recognition."
            ) from exc

        self._app = FaceAnalysis(name=self.model_name, providers=self.providers)
        self._app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))

    def extract_from_image(self, image_path: str | Path) -> DetectedFace:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to read images") from exc

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        faces = self.detect(image)
        if not faces:
            raise ValueError(f"No face detected in image: {image_path}")
        faces.sort(key=lambda face: _area(face.bbox), reverse=True)
        return faces[0]

    def detect(self, frame: Any) -> list[DetectedFace]:
        if self._app is None:
            self.load()
        assert self._app is not None

        faces = self._app.get(frame)
        results: list[DetectedFace] = []
        for face in faces:
            x1, y1, x2, y2 = [int(value) for value in face.bbox]
            embedding = [float(value) for value in face.embedding]
            det_score = float(getattr(face, "det_score", 1.0))
            results.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    embedding=embedding,
                    quality_score=det_score,
                )
            )
        return results


def _area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)
