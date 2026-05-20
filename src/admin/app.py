from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.admin.schemas import PersonPayload, ReviewPayload
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
