# Instructions

This repository contains the backend for a candidate search system.

Our first customer is a recruitment agency that specialises in airport staffing (pilots, flight attendants, air traffic controllers,
mechanics, ramp agents, customer service, TSA, dispatch, etc.).

We designed the system to search candidate resumes. However, the customer has started testing and said they forgot to tell us that they have a lot of valuable information in recruiter notes that they want to be able to search as well.

### Your task

Design a way to incorporate candidate notes into the search. Bear the following requirements in mind:

- We should support a way to ingest notes when candidates are first added to the database via the `/candidates` endpoint (see below)
- We should support the ability to add notes to existing candidates

The notes the customer has asked us to add can be found in `/data/notes.json`.

### Deliverable

You will present your proposed approach in a final interview with the Head of AI and CTO. We are most interested to see how you thought about the problem and solution. 

Please produce a document, deck, or whatever medium works best for you and send it to **trevor@joinpopp.com** 24 hours before your interview. This will give us time to come up with follow-up questions and allow us to have a deeper discussion.

# Repository details & setup

A simple FastAPI service for semantically searching a database of ~100
synthetic resumes.

- **Vector store:** SQLite stores one float32 embedding per candidate resume and one per
  recruiter note. Structured filters run in SQL; cosine similarity runs in NumPy.
- **Ranking:** A candidate's score is the maximum similarity across their resume and notes.
  Each candidate appears once. Results include `match_source` and the winning `matched_note`
  when a note outranks the resume. Ties favour the resume, then the first inserted note;
  candidates with equal scores retain candidate ID order.
- **Embeddings:** Gemini `gemini-embedding-2` (768-d, L2-normalised), with document and search-query prefixes for retrieval. Each resume is sent as a separate `Content` object so it receives its own embedding.
- **Model changes:** The DB records embedding model, dimensions and retrieval-format version.
  Search and embedding writes reject missing/incompatible metadata. Rebuild with
  `./setup.sh --rebuild` after changing the model or formatting; increment
  `EMBEDDING_FORMAT_VERSION` when changing retrieval prefixes. Dimensions alone cannot
  establish compatibility.
- **Extraction:** Gemini (`google-genai`) turns free-text resumes into structured fields,
  choosing skills only from the taxonomy in `app/taxonomy.py`.

## Getting started

### 1. Prerequisites

- **Python 3.13+** and **[uv](https://docs.astral.sh/uv/getting-started/installation/)**.
  Install uv with `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) or
  `brew install uv`. uv creates the virtual environment and installs everything else.
- A **Gemini API key**. The service uses Gemini for embeddings and for extracting structured
  fields from resumes, so it cannot run without one. Tests do not need it.

### 2. Gemini API key

We will send you a Gemini API key, so don't worry about creating or using your own.

### 3. Set the API key

Put it in a local `.env` file in the repository root (gitignored):

```dotenv
GEMINI_API_KEY=your-key
```

Alternatively, export it in the shell you'll run the script from:

```bash
export GEMINI_API_KEY=AIza...your-key...
```

### 4. Run the setup script

```bash
./setup.sh
```

On Windows, use the native PowerShell launcher:

```powershell
.\setup.ps1
```

Both launchers use the same Python setup logic and accept `--no-run`, `--rebuild`,
`--port 8002` and `--help`. `uv` must be on `PATH`.

The script is idempotent and does, in order:

1. Checks for uv and installs the locked dependencies (`uv sync --locked`).
2. Loads the API key from the environment or `.env` and validates setup options.
3. Validates the resume source and imports only missing seeds. Existing seeds must match
   their stored identity mapping; incompatible metadata or ambiguous IDs stop setup.
   An existing but empty/incomplete DB is not treated as a successful installation.
4. Validates stored embeddings and imports supplied notes, skipping existing entries.
5. Starts the API with uvicorn on `http://127.0.0.1:8000`.

Fresh setup uses Gemini; later runs skip provider calls for verified existing data.

Open `http://127.0.0.1:8000/` for the recruiter UI: search and filter candidates, view
resumes and notes, add candidates, and add notes. The page uses the existing API endpoints.
Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### Script options

| Command | Effect |
|---|---|
| `./setup.sh` | Set up if needed, then run on port 8000 |
| `PORT=8002 ./setup.sh` | Run on a different port (use this if 8000 is taken) |
| `./setup.sh --rebuild` | Atomically replace candidates/notes after successful embedding, then reimport supplied notes |
| `./setup.sh --no-run` | Do the setup steps but don't start the server |
| `./setup.sh --help` | Print the options |

Stop the server with `Ctrl-C`. Re-running setup verifies existing data and skips duplicate
imports. Startup creates missing schema tables without re-embedding resumes.

### Existing databases without metadata

A legacy DB cannot reveal which model/format generated its vectors. If that provenance is
known to match the current configuration, explicitly adopt it:

```bash
uv run python scripts/build_db.py --adopt-existing
```

This verifies every seed candidate against its original source position and fields, validates all
vectors and foreign keys, and adds metadata/mappings in one transaction. It does not alter
candidate/note rows or generate embeddings. It refuses conflicting metadata or identities.
If provenance is unknown, use an explicit rebuild instead.

Read-only integrity/configuration check:

```bash
uv run python scripts/build_db.py --check
```

### Import the supplied notes

Setup imports supplied notes automatically. To repeat the import separately:

```bash
uv run python scripts/import_notes.py
```

This imports `data/notes.json` into the existing candidates without re-embedding resumes.
It validates all entries and candidate IDs before calling Gemini, generates the new note
vectors before writing, and commits the import in one transaction. Repeated entries with
the same candidate, text, and UTC timestamp are skipped, including on subsequent runs.
The shipped notes reference the one-based positions in the original `resumes.json` list.
The importer creates these associations internally and verifies stored candidate fingerprints;
the sample resumes need no added IDs. Preserve source order when using these notes.
Changed or reordered sources are rejected against an existing mapping. Other files use database candidate IDs by default. Use `--seed-ids` or
`--database-ids` to choose explicitly when using `--file path/to/notes.json`.
If candidates were built with a custom `--file`, pass the corresponding resume source as
`--seed-source path/to/resumes.json --seed-ids` when importing its notes.

Input-only validation and quality counts, without Gemini calls or DB changes:

```bash
uv run python scripts/import_notes.py --audit
```

The supplied data contains seven placeholders and five numeric-only notes. These, recruitment
activity and full transcripts are retained verbatim and remain searchable in this baseline.
The audit flags simple placeholder/numeric cases; it does not infer candidate facts or
classify transcripts/messages. Classification, junk-indexing policy and chunking are planned
separately in `to-do-improvements.md`.

A successful explicit rebuild removes candidates and their notes, including manually added
records. Setup reimports shipped notes afterwards; the standalone builder requires a separate
notes import. Validation/provider/write failures preserve the previous candidate and note data.

### Troubleshooting

- **`uv is required`** — install uv (see Prerequisites) and open a new shell so it's on your `PATH`.
- **Missing API key** — set it in the root `.env` or export it in the same shell.
- **Incompatible/missing embedding metadata** — explicitly adopt a legacy DB only with known
  matching provenance, or rebuild. The app returns `503` instead of mixing vector spaces.
- **Seed mapping mismatch** — verify the input source and its IDs; setup does not overwrite an
  occupied/changed candidate. Rebuild explicitly only when replacement is intended.
- **`[Errno 48] Address already in use`** — something else is on port 8000. Run with
  `PORT=8002 ./setup.sh`.
- **`401`/`403` from Gemini** during the DB build — the key is wrong, revoked, or from a project
  with the Generative Language API disabled. Generate a fresh key in AI Studio.
- **Requests hang** — Gemini calls time out after two minutes. Embedding requests retry transient errors (including `429`) up to five total attempts with backoff; bulk setup can take longer while waiting for quota to reset.

## Manual steps

If you'd rather not use the script:

```bash
uv sync
uv run python scripts/build_db.py        # safe default: import missing seeds, skip verified ones
uv run python scripts/import_notes.py    # import notes into existing candidates
uv run uvicorn main:app --reload
```

Use `scripts/build_db.py --rebuild` for explicit replacement. Embeddings are generated before
the replacement transaction; a database change during generation aborts the build rather than
overwriting that change. `--no-reset` / `--ensure` are aliases for the default import mode.
For isolated runs, set `POPP_DB_PATH` in the shell to a separate SQLite file.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/search` | `{"query", "top_k", "skills", "current_job_title", "past_job_titles", "min_years_experience", "max_years_experience", "location_city", "location_region"}` — filters are optional; text filters are case-insensitive substring matches (`captain` matches `Senior Captain`), skills must all be present exactly, each past-title value must be contained in some past title. Default `strategy: "reranked"` uses requirement retrieval and LLM assessment; `strategy: "baseline"` retains maximum resume/note cosine similarity. `total_matching_filters` counts candidates before ranking |
| `POST` | `/candidates` | `{"name", "resume", "notes": [{"note", "timestamp"}]}` — notes are optional; extract resume fields, embed resume and notes separately, then insert together |
| `POST` | `/candidates/{id}/notes` | `{"note", "timestamp"}` — attach an independently embedded note to an existing candidate |
| `GET` | `/candidates/{id}` | Fetch one candidate and their notes in timestamp order |
| `GET` | `/taxonomy` | Skills taxonomy grouped by category |
| `GET` | `/health` | Liveness |

Note text must contain non-whitespace characters. API timestamps may be omitted, defaulting
to the current UTC time; supplied timestamps must include a timezone and are normalised to
UTC. Import timestamps are required. An unknown candidate returns `404`; an identical note
for the same candidate and timestamp returns `409`. Provider failures return `503` without
saving a partial candidate or note.

Candidate names/resumes and search queries must be nonblank. Unknown request fields,
blank filter entries, unknown taxonomy skills and inverted experience ranges return `422`.
An empty filter list means no filter. Supplied skills and states are normalised consistently
for imported and extracted candidate fields; unknown extracted skills are dropped.
Imports validate UTF-8 JSON and report the first invalid row/field before provider calls.

Candidate creation rejects an exact existing name/resume pair with `409`, including the
existing candidate ID. Names are trimmed by request validation; resume text is compared
verbatim, case-sensitively. The early check avoids Gemini calls, and a unique database index
also prevents concurrent duplicates. Initial notes in a rejected request are not saved;
add them using the existing candidate's notes endpoint. Changed resumes are not merged.

For example, a candidate can be created with initial notes:

```json
{
  "name": "Example Candidate",
  "resume": "Aircraft maintenance technician with five years of experience...",
  "notes": [
    {"note": "Open to seasonal work in Anchorage.", "timestamp": "2026-01-01T12:00:00Z"}
  ]
}
```

Notes affect semantic ranking and do not overwrite structured candidate fields. Filters
continue to use those fields, including current residence rather than relocation preferences.
The initial ranking preserves raw notes, including administrative entries. Cosine similarity
does not enforce negated restrictions or resolve conflicting historical information.

## Tests

```bash
uv run pytest
```

Tests use a temporary DB, a stubbed extractor, and a fake embedder, so no Gemini key is needed.

## Layout

```
app/          config, taxonomy, embeddings, db, ingest, schemas, extraction, FastAPI app
index.html    recruiter UI with inline CSS and JavaScript
data/         resumes.json in its original format, notes.json (committed); applicants.db (gitignored)
scripts/      build_db.py, import_notes.py, setup.py
setup.sh      Bash setup + run
setup.ps1     PowerShell setup + run
tests/        test_api.py, test_notes.py, test_embeddings.py, test_cleanup.py
```


## Document uploads and review

The single-file UI accepts TXT/Markdown (UTF-8), text-based PDF and DOCX resumes.
Drag a file into **Add candidate** or use the file picker. Text extraction runs locally,
then Pydantic AI calls the configured Gemini model with a native `CandidateDraft` schema.
Review/correct the name, resume and extracted fields before saving. Optional reviewed
`fields` on `POST /candidates` avoid a second extraction call. Existing clients can omit
that property and retain the original extract-on-save behaviour.

- `POST /documents/extract`: multipart `file`; returns text, filename and local extraction time.
- `POST /candidates/pre-extract`: JSON `{ "text": "..." }`; returns a draft and model extraction time.
- Existing candidates: open **Recruiter notes**, upload a document, review the text and save it.
- Uploads are limited to 10 MB and 100,000 extracted characters; PDFs to 100 pages;
  DOCX expanded contents to 40 MB. File extensions and parsed content are checked.
- Uploaded bytes are not retained. Draft preparation makes no database writes.
- Scanned PDFs/images, OCR and legacy `.doc` are out of scope. Scanned documents with no
  text receive a clear error. Mixed PDFs can have unreadable image-only content; review
  the extracted text. OCR fallback and extraction-quality detection remain future work.

## Simulated Granola

Open a candidate and select **Recruiter notes → Add information**. The **Granola** button
opens a recent-transcripts panel beside the document drop area. Drag a transcript into
that area (or select it), review its text and save it to the open candidate. Files and
manually entered details use the same form. The candidate's existing resume is preserved.

This is a simulated connection, with fictional meetings in `data/granola_meetings.json`;
there is no OAuth flow, account access or call to a real Granola API. The icon uses the
favicon from the official Granola website. `app/granola.py` is the adapter boundary.

- `GET /integrations/granola/meetings` returns validated meeting fixtures.
- `POST /integrations/granola/meetings/{id}/import` accepts `candidate_id` and reviewed `note`.
- The server supplies the original timestamp and source ID/title. A SQLite association
  records the meeting ID, candidate ID and note ID in the same transaction as the note.
- Reimporting a meeting for the same candidate returns the existing note, even if the
  submitted text has changed; it does not overwrite text or regenerate embeddings.
- Candidate matching is explicit. The open candidate determines the destination; the simulation does not infer identity from a name.

## Evaluation runner

```bash
uv run python scripts/evaluate.py --validate-only
uv run python scripts/evaluate.py --output evals/results/baseline.json
uv run python scripts/evaluate.py --output evals/results/next.json --compare evals/results/baseline.json
```

The runner calls the same search implementation as the API and the Pydantic AI draft
extractor. Runs use live Gemini calls and do not write candidate data. JSON reports include
per-case outputs, errors and latency, model/configuration details, dataset and database
fingerprints, retrieval recall@k, precision@k, MRR, binary nDCG@k, and exact extraction-field
accuracy. Comparison requires the same dataset and database fingerprint. Estimated paid-tier cost and provider/estimated token counts are recorded; actual billing is unmeasured. Reports may contain extracted candidate data and are gitignored.

`evals/starter.json` contains **draft smoke cases**, not an approved recruiter benchmark.
Literal resume/note queries check plumbing, and an impossible location checks empty results;
these do not establish semantic relevance or abstention quality. Retrieval labels are
non-exhaustive. Add reviewed recruiter judgments before using these metrics to choose a
ranking strategy. Cases may test retrieval or extraction; failed cases remain visible and
make the CLI exit non-zero. Metric averages cover successful applicable cases, so always
read them alongside the error count. No LLM judge or classification policy is introduced.


## Requirement retrieval and evidence-based reranking

Search defaults to `strategy: "reranked"`; the UI also offers the similarity baseline.
No document re-embedding or database migration is needed for this search change.

1. Apply only explicit structured filters. These remain hard filters, so current city is
   not interchangeable with willingness to relocate. No matches means no provider calls.
2. Gemini 3.5 Flash-Lite returns up to five typed requirements, distinguishing mandatory
   conditions from preferences. Requirement texts are embedded together with Embedding 2.
3. For each requirement, take the best resume/note similarity per candidate. Combine
   mandatory-requirement scores with a soft minimum (temperature 0.15); if all requirements
   are preferences, combine all of them. This is a retrieval heuristic, not a probability.
4. Keep at most 12 candidates, regardless of `top_k`. Send their source documents in one
   structured LLM assessment call. Evidence is capped at 8,000 characters per candidate;
   notes receive at least half the budget when present and are considered newest first.
   Omitted/truncated evidence is explicitly flagged. This cap can hide older evidence;
   smarter evidence selection and chunking remain future experiments.
5. Require supported/contradicted/unknown judgments per requirement, with source IDs and
   verbatim quotes for supported/contradicted claims. Validate candidate IDs, complete
   requirement coverage and citation provenance. Invalid assessments get one bounded
   regeneration attempt, with both calls included in cost/time accounting.
6. Rank deterministically: fewer mandatory contradictions, more supported mandatory
   requirements, more supported preferences, fewer contradicted preferences, then retrieval
   score and candidate ID. Unknown is distinct from contradicted. Contradicted candidates
   are demoted, not deleted; this implementation does not promise semantic abstention.
7. Show assessments on result cards and quotations in candidate details. `score` remains
   a retrieval score, not an LLM confidence or qualification percentage. Requests for more
   than 12 results receive a warning and at most 12 assessed results.

Provider/validation failures return 503; unassessed results are never silently presented as
AI-ranked. The baseline can be selected explicitly. The maximum normal search is one planning
call, one batched query-embedding call and one assessment call, plus at most one evidence
validation retry. Existing SDK transport retries can add time and are not fully visible in
usage counters. Output tokens are capped (1,200 planning / 12,000 assessment).

## Measured comparison

See `evals/RESULTS.md` for the nine-query comparison on 100 candidates and 119 notes,
including per-query ordering, latency, estimated cost and limitations. This is a draft,
source-backed benchmark with non-exhaustive labels; human review is still pending.

```bash
uv run python scripts/evaluate.py --dataset evals/recruiter-queries.json --strategy baseline --output evals/results/baseline.json
uv run python scripts/evaluate.py --dataset evals/recruiter-queries.json --strategy reranked --output evals/results/reranked.json --compare evals/results/baseline.json
```

Reports include stage timings, token counts, a shortlist-recall diagnostic separate from
final ranking metrics, prompt/configuration metadata and dataset/database fingerprints.
The token basis is labelled per call: provider counts where returned, characters/4 for
embedding responses without counts. Rates are versioned in `app/search_telemetry.py` and
unknown-model prices remain unknown. Charges for failed requests may be unavailable.
Use `--retry-errors previous.json` to retry only failed cases while preserving their prior
attempts and cumulative time/cost. Quality averages must be read alongside error counts.
