from __future__ import annotations

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.camera.capture import CameraCapture, CameraSettings
from src.config import load_config
from src.events.alert import AlertService
from src.events.decision import EventDecisionEngine
from src.face.matcher import FaceMatcher
from src.face.quality import FaceQualityEvaluator
from src.face.recognizer import InsightFaceRecognizer
from src.onnx_runtime import missing_cuda_dlls, preload_onnxruntime_dlls, resolve_providers
from src.storage.database import Database, Person
from src.tracking.tracker import CentroidTracker
from src.tracking.voter import MultiFrameVoter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face access control prototype")
    parser.add_argument("--config", default="config/config.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="Initialize SQLite database")
    subparsers.add_parser("check-gpu", help="Show ONNX Runtime execution providers")
    run_admin = subparsers.add_parser("run-admin", help="Run FastAPI admin backend")
    run_admin.add_argument("--host", default="127.0.0.1")
    run_admin.add_argument("--port", type=int, default=8000)
    run_admin.add_argument("--reload", action="store_true")

    add_person = subparsers.add_parser("add-person", help="Add or update a person")
    add_person.add_argument("--person-id", required=True)
    add_person.add_argument("--name", required=True)
    add_person.add_argument("--role", default="student")
    add_person.add_argument("--department", default="")
    add_person.add_argument("--status", default="active")
    add_person.add_argument("--valid-from")
    add_person.add_argument("--valid-until")
    add_person.add_argument("--consent-status", default="granted")

    enroll = subparsers.add_parser("enroll-face", help="Extract and store a face embedding")
    enroll.add_argument("--person-id", required=True)
    enroll.add_argument("--image", required=True)
    enroll.add_argument("--angle", default="front")

    run_camera = subparsers.add_parser("run-camera", help="Run real-time camera recognition")
    run_camera.add_argument("--camera-id", default="camera-1")
    run_camera.add_argument("--location", default="default")
    run_camera.add_argument("--max-frames", type=int)
    run_camera.add_argument(
        "--show-window",
        action="store_true",
        help="Show an OpenCV preview window with face boxes. Press q to quit.",
    )

    return parser


def database_from_config(config: dict[str, Any]) -> Database:
    return Database(config["storage"]["database_path"])


def command_init_db(config: dict[str, Any]) -> None:
    db = database_from_config(config)
    db.init_schema()
    for path_key in ("snapshot_dir", "registered_face_dir"):
        Path(config["storage"][path_key]).mkdir(parents=True, exist_ok=True)
    print(f"Initialized database at {db.path}")


def command_add_person(config: dict[str, Any], args: argparse.Namespace) -> None:
    db = database_from_config(config)
    db.init_schema()
    db.upsert_person(
        Person(
            person_id=args.person_id,
            name=args.name,
            role=args.role,
            department=args.department,
            status=args.status,
            valid_from=args.valid_from,
            valid_until=args.valid_until,
            consent_status=args.consent_status,
        )
    )
    print(f"Saved person {args.person_id} {args.name}")


def command_enroll_face(config: dict[str, Any], args: argparse.Namespace) -> None:
    db = database_from_config(config)
    db.init_schema()
    recognizer = build_recognizer(config)
    face = recognizer.extract_from_image(args.image)
    db.add_embedding(
        person_id=args.person_id,
        image_path=args.image,
        embedding=face.embedding,
        angle=args.angle,
        quality_score=face.quality_score,
        model_name=config["recognition"]["model_name"],
    )
    print(f"Stored face embedding for {args.person_id}")


def command_run_camera(config: dict[str, Any], args: argparse.Namespace) -> None:
    cv2 = None
    if args.show_window:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for --show-window") from exc
        cv2 = cv2_module

    db = database_from_config(config)
    db.init_schema()

    recognizer = build_recognizer(config)
    quality = FaceQualityEvaluator()
    matcher = FaceMatcher(
        db.load_active_embeddings(),
        confirm_threshold=float(config["recognition"]["confirm_threshold"]),
        suspect_threshold=float(config["recognition"]["suspect_threshold"]),
    )
    voter = MultiFrameVoter(
        window=int(config["recognition"]["vote_window"]),
        confirm_count=int(config["recognition"]["vote_confirm_count"]),
    )
    tracker = CentroidTracker()
    decision_engine = EventDecisionEngine(
        curfew_enabled=bool(config["curfew"]["enabled"]),
        curfew_time=str(config["curfew"]["curfew_time"]),
        grace_period_minutes=int(config["curfew"]["grace_period_minutes"]),
    )
    alerts = AlertService(
        enable_led=bool(config["event"]["enable_led_alert"]),
        enable_buzzer=bool(config["event"]["enable_buzzer_alert"]),
    )

    camera_config = config["camera"]
    capture = CameraCapture(
        CameraSettings(
            source=camera_config["source"],
            fps=int(camera_config["fps"]),
            width=int(camera_config["width"]),
            height=int(camera_config["height"]),
        )
    )

    max_faces = int(config["recognition"]["max_faces_per_frame"])
    detection_interval = 1.0 / float(config["recognition"]["detection_fps"])
    last_detection = 0.0
    processed_frames = 0
    last_overlays: list[tuple[tuple[int, int, int, int], str, str | None, float]] = []

    def process_frame(frame: Any) -> list[tuple[tuple[int, int, int, int], str, str | None, float]]:
        return _process_recognition_frame(
            frame=frame,
            recognizer=recognizer,
            quality=quality,
            matcher=matcher,
            voter=voter,
            tracker=tracker,
            decision_engine=decision_engine,
            alerts=alerts,
            db=db,
            camera_id=args.camera_id,
            location=args.location,
            max_faces=max_faces,
        )

    if cv2 is not None:
        _run_camera_with_async_preview(
            capture=capture,
            process_frame=process_frame,
            detection_interval=detection_interval,
            max_frames=args.max_frames,
        )
        return

    try:
        for frame in capture.frames():
            processed_frames += 1
            now_monotonic = time.monotonic()
            if now_monotonic - last_detection < detection_interval:
                if args.max_frames and processed_frames >= args.max_frames:
                    break
                continue
            last_detection = now_monotonic

            last_overlays = process_frame(frame)

            if args.max_frames and processed_frames >= args.max_frames:
                break
    finally:
        capture.release()


def _area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _process_recognition_frame(
    frame: Any,
    recognizer: InsightFaceRecognizer,
    quality: FaceQualityEvaluator,
    matcher: FaceMatcher,
    voter: MultiFrameVoter,
    tracker: CentroidTracker,
    decision_engine: EventDecisionEngine,
    alerts: AlertService,
    db: Database,
    camera_id: str,
    location: str,
    max_faces: int,
) -> list[tuple[tuple[int, int, int, int], str, str | None, float]]:
    faces = recognizer.detect(frame)
    faces.sort(key=lambda item: _area(item.bbox), reverse=True)
    selected_faces = faces[:max_faces]
    track_ids = tracker.assign([face.bbox for face in selected_faces])
    overlays: list[tuple[tuple[int, int, int, int], str, str | None, float]] = []

    for face, track_id in zip(selected_faces, track_ids):
        quality_result = quality.evaluate(face.bbox, face.quality_score)
        if not quality_result.accepted:
            db.add_recognition_log(
                camera_id=camera_id,
                location=location,
                result_type="low_quality",
                track_id=track_id,
            )
            overlays.append((face.bbox, "low_quality", None, face.quality_score))
            continue

        match = matcher.match(face.embedding)
        vote = voter.add(track_id, match)
        decision = decision_engine.decide(vote, datetime.now())
        db.add_recognition_log(
            camera_id=camera_id,
            location=location,
            result_type=decision.log_type,
            person_id=vote.person_id,
            person_name=vote.person_name,
            similarity=vote.similarity,
            track_id=track_id,
        )

        if decision.event_type:
            db.add_security_event(
                event_type=decision.event_type,
                camera_id=camera_id,
                location=location,
                person_id=vote.person_id,
                person_name=vote.person_name,
                confidence=vote.similarity,
            )
            alerts.notify(decision.event_type)

        overlays.append((face.bbox, vote.result_type, vote.person_name, vote.similarity))

    return overlays


def _run_camera_with_async_preview(
    capture: CameraCapture,
    process_frame: Any,
    detection_interval: float,
    max_frames: int | None,
) -> None:
    import cv2

    lock = threading.Lock()
    worker: threading.Thread | None = None
    worker_busy = False
    last_detection = 0.0
    processed_frames = 0
    latest_overlays: list[tuple[tuple[int, int, int, int], str, str | None, float]] = []

    def run_worker(frame_copy: Any) -> None:
        nonlocal latest_overlays, worker_busy
        try:
            overlays = process_frame(frame_copy)
            with lock:
                latest_overlays = overlays
        finally:
            with lock:
                worker_busy = False

    try:
        for frame in capture.frames():
            processed_frames += 1
            with lock:
                overlays = list(latest_overlays)
                busy = worker_busy

            for overlay in overlays:
                _draw_face_result(frame, *overlay)

            cv2.imshow("Face Access Control", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            now_monotonic = time.monotonic()
            if now_monotonic - last_detection >= detection_interval and not busy:
                last_detection = now_monotonic
                with lock:
                    worker_busy = True
                worker = threading.Thread(target=run_worker, args=(frame.copy(),), daemon=True)
                worker.start()

            if max_frames and processed_frames >= max_frames:
                break
    finally:
        capture.release()
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        cv2.destroyAllWindows()


def build_recognizer(config: dict[str, Any]) -> InsightFaceRecognizer:
    return InsightFaceRecognizer(
        config["recognition"]["model_name"],
        det_size=int(config["recognition"]["det_size"]),
        providers=list(config["recognition"]["providers"]),
    )


def command_check_gpu() -> None:
    preload_onnxruntime_dlls()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime or onnxruntime-gpu is not installed") from exc

    providers = ort.get_available_providers()
    requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    resolved = resolve_providers(requested)
    print("ONNX Runtime available providers:")
    for provider in providers:
        print(f"- {provider}")
    print("Project resolved providers:")
    for provider in resolved:
        print(f"- {provider}")

    missing = missing_cuda_dlls()
    if missing:
        print("Missing CUDA/cuDNN DLLs:")
        for dll_name in missing:
            print(f"- {dll_name}")

    if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" in resolved:
        print("CUDAExecutionProvider is available. GPU inference can be used.")
    else:
        print("CUDAExecutionProvider is NOT available. The project will fall back to CPU.")


def command_run_admin(config: dict[str, Any], args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required. Run `pip install -r requirements.txt`.") from exc

    from src.admin.app import create_app

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def _draw_face_result(
    frame: Any,
    bbox: tuple[int, int, int, int],
    result_type: str,
    person_name: str | None,
    similarity: float,
) -> None:
    import cv2

    x1, y1, x2, y2 = bbox
    if result_type == "matched":
        color = (0, 180, 0)
    elif result_type == "suspected":
        color = (0, 180, 255)
    else:
        color = (0, 0, 255)

    label_name = person_name or "unknown"
    label = f"{result_type}: {label_name} {similarity:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    _draw_unicode_text(frame, label, (x1, max(0, y1 - 34)), color)


def _draw_unicode_text(frame: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        import cv2

        cv2.putText(
            frame,
            _ascii_fallback(text),
            (origin[0], max(20, origin[1] + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        return

    font = _load_display_font(26)
    if font is None:
        cv2.putText(
            frame,
            _ascii_fallback(text),
            (origin[0], max(20, origin[1] + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        return

    x, y = origin
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    rgb_color = (color[2], color[1], color[0])
    draw.text((x, y), text, font=font, fill=rgb_color)
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _load_display_font(size: int) -> Any:
    from pathlib import Path

    from PIL import ImageFont

    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return None


def _ascii_fallback(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "init-db":
        command_init_db(config)
    elif args.command == "check-gpu":
        command_check_gpu()
    elif args.command == "run-admin":
        command_run_admin(config, args)
    elif args.command == "add-person":
        command_add_person(config, args)
    elif args.command == "enroll-face":
        command_enroll_face(config, args)
    elif args.command == "run-camera":
        command_run_camera(config, args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
