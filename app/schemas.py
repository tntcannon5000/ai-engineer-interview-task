from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.normalization import normalize_state
from app.taxonomy import ALL_SKILLS, normalize_skills


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoteInput(StrictInput):
    note: str = Field(min_length=1)
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("note")
    @classmethod
    def non_blank_note(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("note must not be blank")
        return value

    @field_validator("timestamp")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)


class CandidateNote(NoteInput):
    id: int
    candidate_id: int


class ImportNote(NoteInput):
    candidate_id: int = Field(gt=0)
    timestamp: AwareDatetime


class CandidateFields(BaseModel):
    skills: list[str] = Field(default_factory=list)
    current_job_title: str
    past_job_titles: list[str] = Field(default_factory=list)
    years_experience: int = Field(ge=0)
    location_city: str
    location_region: str

    @field_validator("skills")
    @classmethod
    def canonical_skills(cls, value):
        return normalize_skills(value)

    @field_validator("location_region")
    @classmethod
    def canonical_region(cls, value):
        return normalize_state(value)

    @field_validator("current_job_title", "location_city")
    @classmethod
    def trim_field(cls, value):
        return value.strip()

    @field_validator("past_job_titles")
    @classmethod
    def clean_titles(cls, value):
        return list(dict.fromkeys(title.strip() for title in value if title.strip()))


class Candidate(CandidateFields):
    id: int
    name: str
    resume: str


class CandidateDetails(Candidate):
    notes: list[CandidateNote] = Field(default_factory=list)


class SearchRequest(StrictInput):
    strategy: Literal["baseline", "reranked"] = "reranked"
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=100)
    skills: list[str] | None = None
    current_job_title: str | None = None
    past_job_titles: list[str] | None = None
    min_years_experience: int | None = Field(default=None, ge=0)
    max_years_experience: int | None = Field(default=None, ge=0)
    location_city: str | None = None
    location_region: str | None = None

    @field_validator("query", "current_job_title", "location_city", "location_region")
    @classmethod
    def non_blank_text(cls, value):
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("must contain non-whitespace characters")
        return value

    @field_validator("skills", "past_job_titles")
    @classmethod
    def non_blank_filters(cls, value):
        if value is None:
            return value
        if any(not item.strip() for item in value):
            raise ValueError("filter entries must not be blank")
        return list(dict.fromkeys(item.strip() for item in value))

    @field_validator("skills")
    @classmethod
    def known_skills(cls, value):
        if value is None:
            return value
        lookup = {skill.casefold(): skill for skill in ALL_SKILLS}
        if any(skill.casefold() not in lookup for skill in value):
            raise ValueError("skills must belong to the skills taxonomy")
        return list(dict.fromkeys(lookup[skill.casefold()] for skill in value))

    @model_validator(mode="after")
    def ordered_experience(self):
        if self.min_years_experience is not None and self.max_years_experience is not None:
            if self.min_years_experience > self.max_years_experience:
                raise ValueError("min_years_experience must not exceed max_years_experience")
        return self


class Requirement(StrictInput):
    text: str = Field(min_length=1, max_length=300)
    mandatory: bool = True


class EvidenceQuote(StrictInput):
    source_id: str
    quote: str = Field(min_length=1, max_length=500)


class RequirementAssessment(StrictInput):
    requirement_index: int = Field(ge=0)
    status: Literal["supported", "contradicted", "unknown"]
    evidence: list[EvidenceQuote] = Field(default_factory=list, max_length=3)


class CandidateAssessment(StrictInput):
    candidate_id: int
    requirements: list[RequirementAssessment]


class CandidateLookup(BaseModel):
    id: int
    name: str
    current_job_title: str
    location_city: str
    location_region: str


class SearchResult(Candidate):
    relevance_grade: int | None = Field(default=None, ge=0, le=4)
    score: float
    match_source: Literal["resume", "note"] = "resume"
    matched_note: CandidateNote | None = None
    assessment: CandidateAssessment | None = None
    evidence_truncated: bool = False


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_matching_filters: int
    strategy: Literal["baseline", "reranked"] = "baseline"
    requirements: list[Requirement] = Field(default_factory=list)
    shortlist_size: int = 0
    warning: str | None = None


class ReviewedCandidateFields(CandidateFields):
    model_config = ConfigDict(extra="forbid")


class AddCandidateRequest(StrictInput):
    name: str
    resume: str
    notes: list[NoteInput] = Field(default_factory=list)
    fields: ReviewedCandidateFields | None = None

    @field_validator("name", "resume")
    @classmethod
    def non_blank_candidate(cls, value, info):
        if not value.strip():
            raise ValueError("must contain non-whitespace characters")
        return value.strip() if info.field_name == "name" else value


class SeedCandidate(CandidateFields):
    model_config = ConfigDict(extra="forbid")
    name: str
    resume: str

    @field_validator("name", "resume")
    @classmethod
    def non_blank_candidate(cls, value, info):
        return AddCandidateRequest.non_blank_candidate(value, info)
