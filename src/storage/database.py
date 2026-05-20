from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Person:
    person_id: str
    name: str
    role: str = "student"
    department: str = ""
    status: str = "active"
    valid_from: str | None = None
    valid_until: str | None = None
    consent_status: str = "granted"


@dataclass(frozen=True)
class EmbeddingRecord:
    person_id: str
    person_name: str
    embedding: list[float]
    angle: str
    quality_score: float
    model_name: str


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._memory_connection: sqlite3.Connection | None = None
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        if str(self.path) == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_connection.row_factory = sqlite3.Row
                self._memory_connection.execute("PRAGMA journal_mode=MEMORY")
            return self._memory_connection

        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=MEMORY")
        return connection

    @contextmanager
    def session(self):
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        finally:
            if str(self.path) != ":memory:":
                connection.close()

    def init_schema(self) -> None:
        with self.session() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'student',
                    department TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    valid_from TEXT,
                    valid_until TEXT,
                    consent_status TEXT NOT NULL DEFAULT 'granted',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    angle TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(person_id)
                );

                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    location TEXT NOT NULL,
                    person_id TEXT,
                    person_name TEXT,
                    similarity REAL,
                    result_type TEXT NOT NULL,
                    snapshot_path TEXT,
                    track_id TEXT
                );

                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    person_id TEXT,
                    person_name TEXT,
                    camera_id TEXT NOT NULL,
                    location TEXT NOT NULL,
                    confidence REAL,
                    snapshot_path TEXT,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    reviewer TEXT,
                    review_comment TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS system_configs (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def upsert_person(self, person: Person) -> None:
        now = utc_now()
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO persons (
                    person_id, name, role, department, status, valid_from,
                    valid_until, consent_status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    name = excluded.name,
                    role = excluded.role,
                    department = excluded.department,
                    status = excluded.status,
                    valid_from = excluded.valid_from,
                    valid_until = excluded.valid_until,
                    consent_status = excluded.consent_status,
                    updated_at = excluded.updated_at
                """,
                (
                    person.person_id,
                    person.name,
                    person.role,
                    person.department,
                    person.status,
                    person.valid_from,
                    person.valid_until,
                    person.consent_status,
                    now,
                    now,
                ),
            )

    def add_embedding(
        self,
        person_id: str,
        image_path: str,
        embedding: Iterable[float],
        angle: str,
        quality_score: float,
        model_name: str,
    ) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO face_embeddings (
                    person_id, image_path, embedding, angle, quality_score,
                    model_name, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    image_path,
                    json.dumps(list(embedding)),
                    angle,
                    quality_score,
                    model_name,
                    utc_now(),
                ),
            )

    def load_active_embeddings(self) -> list[EmbeddingRecord]:
        with self.session() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.person_id,
                    p.name AS person_name,
                    e.embedding,
                    e.angle,
                    e.quality_score,
                    e.model_name
                FROM face_embeddings e
                JOIN persons p ON p.person_id = e.person_id
                WHERE p.status IN ('active', 'visitor')
                  AND p.consent_status = 'granted'
                  AND (p.valid_until IS NULL OR p.valid_until = '' OR p.valid_until >= DATE('now'))
                """
            ).fetchall()
        return [
            EmbeddingRecord(
                person_id=row["person_id"],
                person_name=row["person_name"],
                embedding=[float(value) for value in json.loads(row["embedding"])],
                angle=row["angle"],
                quality_score=float(row["quality_score"]),
                model_name=row["model_name"],
            )
            for row in rows
        ]

    def add_recognition_log(
        self,
        camera_id: str,
        location: str,
        result_type: str,
        person_id: str | None = None,
        person_name: str | None = None,
        similarity: float | None = None,
        snapshot_path: str | None = None,
        track_id: str | None = None,
        event_time: str | None = None,
    ) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO recognition_logs (
                    event_time, camera_id, location, person_id, person_name,
                    similarity, result_type, snapshot_path, track_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_time or utc_now(),
                    camera_id,
                    location,
                    person_id,
                    person_name,
                    similarity,
                    result_type,
                    snapshot_path,
                    track_id,
                ),
            )

    def add_security_event(
        self,
        event_type: str,
        camera_id: str,
        location: str,
        person_id: str | None = None,
        person_name: str | None = None,
        confidence: float | None = None,
        snapshot_path: str | None = None,
        event_time: str | None = None,
    ) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO security_events (
                    event_type, event_time, person_id, person_name, camera_id,
                    location, confidence, snapshot_path, review_status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    event_type,
                    event_time or utc_now(),
                    person_id,
                    person_name,
                    camera_id,
                    location,
                    confidence,
                    snapshot_path,
                    utc_now(),
                ),
            )

    def list_persons(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute(
                """
                SELECT
                    person_id, name, role, department, status, valid_from,
                    valid_until, consent_status, created_at, updated_at
                FROM persons
                ORDER BY updated_at DESC, person_id ASC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        with self.session() as conn:
            row = conn.execute(
                """
                SELECT
                    person_id, name, role, department, status, valid_from,
                    valid_until, consent_status, created_at, updated_at
                FROM persons
                WHERE person_id = ?
                """,
                (person_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_recognition_logs(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute(
                """
                SELECT
                    id, event_time, camera_id, location, person_id, person_name,
                    similarity, result_type, snapshot_path, track_id
                FROM recognition_logs
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_security_events(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with self.session() as conn:
            rows = conn.execute(
                """
                SELECT
                    id, event_type, event_time, person_id, person_name, camera_id,
                    location, confidence, snapshot_path, review_status, reviewer,
                    review_comment, created_at
                FROM security_events
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_security_event_review(
        self,
        event_id: int,
        review_status: str,
        reviewer: str | None = None,
        review_comment: str | None = None,
    ) -> dict[str, Any] | None:
        if review_status not in {"confirmed", "rejected"}:
            raise ValueError("review_status must be confirmed or rejected")

        with self.session() as conn:
            conn.execute(
                """
                UPDATE security_events
                SET review_status = ?, reviewer = ?, review_comment = ?
                WHERE id = ?
                """,
                (review_status, reviewer, review_comment, event_id),
            )
            row = conn.execute(
                """
                SELECT
                    id, event_type, event_time, person_id, person_name, camera_id,
                    location, confidence, snapshot_path, review_status, reviewer,
                    review_comment, created_at
                FROM security_events
                WHERE id = ?
                """,
                (event_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_summary_stats(self) -> dict[str, int]:
        with self.session() as conn:
            person_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            log_count = conn.execute("SELECT COUNT(*) FROM recognition_logs").fetchone()[0]
            pending_event_count = conn.execute(
                "SELECT COUNT(*) FROM security_events WHERE review_status = 'pending'"
            ).fetchone()[0]
            today_late_return_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM security_events
                WHERE event_type = 'late_return'
                  AND DATE(event_time) = DATE('now')
                """
            ).fetchone()[0]
        return {
            "person_count": int(person_count),
            "log_count": int(log_count),
            "pending_event_count": int(pending_event_count),
            "today_late_return_count": int(today_late_return_count),
        }
