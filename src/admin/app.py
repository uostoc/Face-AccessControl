from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.admin.schemas import PersonPayload, ReviewPayload
from src.face.recognizer import InsightFaceRecognizer
from src.storage.database import Database, Person


def create_app(config: dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Face Access Control Admin API")
    db = Database(config["storage"]["database_path"])
    db.init_schema()
    app.state.db = db

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/persons")
    def list_persons(
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        return db.list_persons(limit=limit, offset=offset)

    @app.post("/api/persons")
    def create_person(payload: PersonPayload) -> dict[str, Any]:
        db.upsert_person(_to_person(payload))
        person = db.get_person(payload.person_id)
        assert person is not None
        return person

    @app.put("/api/persons/{person_id}")
    def update_person(person_id: str, payload: PersonPayload) -> dict[str, Any]:
        if payload.person_id != person_id:
            raise HTTPException(status_code=400, detail="person_id in path and body must match")
        db.upsert_person(_to_person(payload))
        person = db.get_person(person_id)
        assert person is not None
        return person

    @app.post("/api/persons/{person_id}/face-images")
    def upload_face_image(
        person_id: str,
        angle: str = Form(default="front"),
        image: UploadFile = File(...),
    ) -> dict[str, Any]:
        person = db.get_person(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="person not found")

        suffix = Path(image.filename or "").suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            raise HTTPException(status_code=400, detail="unsupported image file type")

        person_dir = Path(config["storage"]["registered_face_dir"]) / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        image_path = person_dir / f"{angle}_{timestamp}{suffix}"

        try:
            image_path.write_bytes(image.file.read())
        finally:
            image.file.close()

        recognizer = _build_recognizer(config)
        try:
            face = recognizer.extract_from_image(image_path)
        except (RuntimeError, ValueError) as exc:
            image_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        db.add_embedding(
            person_id=person_id,
            image_path=str(image_path),
            embedding=face.embedding,
            angle=angle,
            quality_score=face.quality_score,
            model_name=config["recognition"]["model_name"],
        )
        return {
            "person_id": person_id,
            "image_path": str(image_path),
            "angle": angle,
            "quality_score": face.quality_score,
            "model_name": config["recognition"]["model_name"],
        }

    @app.get("/api/persons/{person_id}/embeddings")
    def list_person_embeddings(person_id: str) -> list[dict[str, Any]]:
        if db.get_person(person_id) is None:
            raise HTTPException(status_code=404, detail="person not found")
        return db.list_person_embeddings(person_id)

    @app.get("/api/recognition-logs")
    def list_recognition_logs(
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        return db.list_recognition_logs(limit=limit, offset=offset)

    @app.get("/api/security-events")
    def list_security_events(
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        return db.list_security_events(limit=limit, offset=offset)

    @app.put("/api/security-events/{event_id}/review")
    def review_security_event(event_id: int, payload: ReviewPayload) -> dict[str, Any]:
        try:
            event = db.update_security_event_review(
                event_id=event_id,
                review_status=payload.review_status,
                reviewer=payload.reviewer,
                review_comment=payload.review_comment,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if event is None:
            raise HTTPException(status_code=404, detail="security event not found")
        return event

    @app.get("/api/configs")
    def get_configs() -> dict[str, Any]:
        return config

    @app.get("/api/stats/summary")
    def get_summary() -> dict[str, int]:
        return db.get_summary_stats()

    return app


def _to_person(payload: PersonPayload) -> Person:
    return Person(
        person_id=payload.person_id,
        name=payload.name,
        role=payload.role,
        department=payload.department,
        status=payload.status,
        valid_from=payload.valid_from,
        valid_until=payload.valid_until,
        consent_status=payload.consent_status,
    )


def _build_recognizer(config: dict[str, Any]) -> InsightFaceRecognizer:
    return InsightFaceRecognizer(
        config["recognition"]["model_name"],
        det_size=int(config["recognition"]["det_size"]),
        providers=list(config["recognition"]["providers"]),
    )
