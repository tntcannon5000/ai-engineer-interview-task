"""Structured field extraction from resume text using Gemini."""

import os

from google import genai
from google.genai import types

from app.config import GEMINI_MODEL
from app.schemas import CandidateFields
from app.taxonomy import SKILLS_TAXONOMY
from app.normalization import normalize_state
from app.provider import provider_call

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        # Fail fast on stalled requests instead of hanging forever (ms).
        _client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120_000))
    return _client


def _taxonomy_block() -> str:
    lines = []
    for category, skills in SKILLS_TAXONOMY.items():
        lines.append(f"{category}:")
        lines.extend(f"  - {s}" for s in skills)
    return "\n".join(lines)


SYSTEM_INSTRUCTION = f"""You extract structured data from job applicant resumes for airport roles.

Return these fields:
- skills: a list of skills the candidate demonstrably has. You MUST choose ONLY from the
  taxonomy below, copying each skill name exactly. Do not invent skills. Include a skill
  only if the resume gives clear evidence for it.
- current_job_title: the candidate's most recent / current role title.
- past_job_titles: earlier role titles, most recent first, excluding the current one.
- years_experience: total years of relevant work experience as a whole number. Use a stated
  total if the resume gives one; otherwise sum the dated roles. Use 0 if unknown.
- location_city: city where the candidate lives or is based.
- location_region: the US state, spelled out in full (e.g. "Texas", not "TX").

If a field is genuinely unknown, use an empty string or empty list.

SKILLS TAXONOMY:
{_taxonomy_block()}
"""


@provider_call(f"Resume extraction ({GEMINI_MODEL})")
def extract_fields(resume: str) -> CandidateFields:
    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=f"RESUME:\n\n{resume}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=CandidateFields,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )
    fields: CandidateFields = response.parsed
    if fields is None:
        fields = CandidateFields.model_validate_json(response.text)
    return CandidateFields.model_validate(fields.model_dump() if isinstance(fields, CandidateFields) else fields)
