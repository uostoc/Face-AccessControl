from __future__ import annotations

import argparse
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
from src.storage.database import Database, Person
from src.tracking.tracker import CentroidTracker
from src.tracking.voter import MultiFrameVoter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face access control prototype")
    parser.add_argument("--config", default="config/config.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="Initialize SQLite database")

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
    recognizer = InsightFaceRecognizer(config["recognition"]["model_name"])
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

    recognizer = InsightFaceRecognizer(config["recognition"]["model_name"])
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

    try:
        for frame in capture.frames():
            processed_frames += 1
            now_monotonic = time.monotonic()
            if now_monotonic - last_detection < detection_interval:
                if args.max_frames and processed_frames >= args.max_frames:
                    break
                continue
            last_detection = now_monotonic

            faces = recognizer.detect(frame)
            faces.sort(key=lambda item: _area(item.bbox), reverse=True)
            selected_faces = faces[:max_faces]
            track_ids = tracker.assign([face.bbox for face in selected_faces])
            for face, track_id in zip(selected_faces, track_ids):
                quality_result = quality.evaluate(face.bbox, face.quality_score)
                if not quality_result.accepted:
                    db.add_recognition_log(
                        camera_id=args.camera_id,
                        location=args.location,
                        result_type="low_quality",
                        track_id=track_id,
                    )
                    continue

                match = matcher.match(face.embedding)
                vote = voter.add(track_id, match)
                decision = decision_engine.decide(vote, datetime.now())
                db.add_recognition_log(
                    camera_id=args.camera_id,
                    location=args.location,
                    result_type=decision.log_type,
                    person_id=vote.person_id,
                    person_name=vote.person_name,
                    similarity=vote.similarity,
                    track_id=track_id,
                )

                if decision.event_type:
                    db.add_security_event(
                        event_type=decision.event_type,
                        camera_id=args.camera_id,
                        location=args.location,
                        person_id=vote.person_id,
                        person_name=vote.person_name,
                        confidence=vote.similarity,
                    )
                    alerts.notify(decision.event_type)

                if cv2 is not None:
                    _draw_face_result(frame, face.bbox, vote.result_type, vote.person_name, vote.similarity)

            if cv2 is not None:
                cv2.imshow("Face Access Control", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and processed_frames >= args.max_frames:
                break
    finally:
        capture.release()
        if cv2 is not None:
            cv2.destroyAllWindows()


def _area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


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
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "init-db":
        command_init_db(config)
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
