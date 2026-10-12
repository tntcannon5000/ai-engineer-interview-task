"""Fixture-backed adapter: no Granola account or live API connection."""
import json
from pydantic import AwareDatetime, Field, TypeAdapter
from app.config import DATA_DIR
from app.schemas import StrictInput


class Meeting(StrictInput):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    timestamp: AwareDatetime
    participants: list[str]
    text: str = Field(min_length=1)


def list_meetings() -> list[Meeting]:
    meetings = TypeAdapter(list[Meeting]).validate_python(json.loads(
        (DATA_DIR / "granola_meetings.json").read_text(encoding="utf-8")))
    if len({m.id for m in meetings}) != len(meetings):
        raise ValueError("Duplicate meeting IDs in simulation fixtures")
    return sorted(meetings, key=lambda m: m.timestamp, reverse=True)


def get_meeting(meeting_id: str) -> Meeting | None:
    return next((m for m in list_meetings() if m.id == meeting_id), None)
