"""Import notes into an existing candidate database.

Usage: uv run python scripts/import_notes.py [--file path/to/notes.json]
"""

import argparse
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import NOTES_PATH  # noqa: E402
from app.ingest import import_notes  # noqa: E402
from app.ingest import note_quality_counts, read_records  # noqa: E402
from app.schemas import ImportNote  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=NOTES_PATH)
    ids = parser.add_mutually_exclusive_group()
    ids.add_argument("--seed-ids", action="store_true", help="resolve IDs through verified seed mappings")
    ids.add_argument("--database-ids", action="store_true", help="treat IDs as database candidate IDs")
    parser.add_argument("--seed-source", type=Path, help="resume source to verify when resolving seed IDs")
    parser.add_argument("--audit", action="store_true", help="validate and report input quality without calling Gemini or writing")
    args = parser.parse_args()
    try:
        if args.audit:
            print(note_quality_counts(read_records(args.file, ImportNote)))
        else:
            seed_ids = True if args.seed_ids else False if args.database_ids else None
            if args.seed_source is not None and not args.seed_ids:
                parser.error("--seed-source requires --seed-ids")
            result = import_notes(args.file, seed_ids=seed_ids, seed_source=args.seed_source)
            print(f"inserted {result['inserted']} notes; skipped {result['skipped']} existing or repeated notes")
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(1, f"Import failed: {exc}\n")
