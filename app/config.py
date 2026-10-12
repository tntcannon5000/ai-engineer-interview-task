from pathlib import Path
import os

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESUMES_PATH = DATA_DIR / "resumes.json"
NOTES_PATH = DATA_DIR / "notes.json"
DB_PATH = Path(os.environ.get("POPP_DB_PATH", DATA_DIR / "applicants.db"))

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 768
EMBEDDING_FORMAT_VERSION = "retrieval-prefixes-v1"
GEMINI_MODEL = "gemini-3.5-flash-lite"

load_dotenv(ROOT_DIR / ".env")
