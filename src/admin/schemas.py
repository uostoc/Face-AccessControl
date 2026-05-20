from __future__ import annotations

from pydantic import BaseModel, Field


class PersonPayload(BaseModel):
    person_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: str = "student"
    department: str = ""
    status: str = "active"
    valid_from: str | None = None
    valid_until: str | None = None
    consent_status: str = "granted"


class ReviewPayload(BaseModel):
    review_status: str
    reviewer: str | None = None
    review_comment: str | None = None

