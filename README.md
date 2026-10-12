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

- **Vector store:** SQLite table with a float32 embedding blob per candidate; structured
  filters run in SQL, cosine similarity runs in numpy.
- **Embeddings:** Gemini `gemini-embedding-001` (768-d, L2-normalised), `RETRIEVAL_DOCUMENT` for resumes and `RETRIEVAL_QUERY` for searches.
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

Export it in the shell you'll run the script from:

```bash
export GEMINI_API_KEY=AIza...your-key...
```

### 4. Run the setup script

```bash
./setup.sh
```

The script is idempotent and does, in order:

1. Checks that uv is installed and that `GEMINI_API_KEY` is exported. It exits with a clear
   message if either is missing.
2. Installs dependencies (`uv sync`).
3. Builds `data/applicants.db` by embedding every resume in `data/resumes.json` **only if the
   DB is missing**. This takes a few seconds.
4. Starts the API with uvicorn on `http://127.0.0.1:8000`.

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

### Script options

| Command | Effect |
|---|---|
| `./setup.sh` | Set up if needed, then run on port 8000 |
| `PORT=8002 ./setup.sh` | Run on a different port (use this if 8000 is taken) |
| `./setup.sh --rebuild` | Drop and rebuild the DB before starting, e.g. after editing `data/resumes.json` |
| `./setup.sh --no-run` | Do the setup steps but don't start the server |
| `./setup.sh --help` | Print the options |

Stop the server with `Ctrl-C`. Re-running the script later skips straight to starting the
server, because the DB already exists.

### Troubleshooting

- **`uv is required`** — install uv (see Prerequisites) and open a new shell so it's on your `PATH`.
- **`GEMINI_API_KEY is not set`** — export it (step 3) in the same shell before running the script.
- **`[Errno 48] Address already in use`** — something else is on port 8000. Run with
  `PORT=8002 ./setup.sh`.
- **`401`/`403` from Gemini** during the DB build — the key is wrong, revoked, or from a project
  with the Generative Language API disabled. Generate a fresh key in AI Studio.
- **Requests hang** — Gemini calls time out after two minutes and retry; check your network.

## Manual steps

If you'd rather not use the script:

```bash
uv sync
uv run python scripts/build_db.py        # build/rebuild data/applicants.db (or POST /db/create)
uv run uvicorn main:app --reload
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/db/create` | `{"reset": true}` — (re)build the SQLite DB from `data/resumes.json` |
| `POST` | `/search` | `{"query", "top_k", "skills", "current_job_title", "past_job_titles", "min_years_experience", "max_years_experience", "location_city", "location_region"}` — filters are optional; text filters are case-insensitive substring matches (`captain` matches `Senior Captain`), skills must all be present exactly, each past-title value must be contained in some past title. Results are ranked by cosine similarity; `total_matching_filters` is the count before ranking |
| `POST` | `/candidates` | `{"name", "resume"}` — extract fields with Gemini, embed, insert |
| `GET` | `/candidates/{id}` | Fetch one candidate |
| `GET` | `/taxonomy` | Skills taxonomy grouped by category |
| `GET` | `/health` | Liveness |


## Tests

```bash
uv run pytest
```

Tests use a temporary DB, a stubbed extractor, and a fake embedder, so no Gemini key is needed.

## Layout

```
app/          config, taxonomy, embeddings, db, ingest, schemas, extraction, FastAPI app
data/         resumes.json (committed), applicants.db (generated, gitignored)
scripts/      build_db.py
setup.sh      one-shot setup + run
tests/        test_api.py
```
