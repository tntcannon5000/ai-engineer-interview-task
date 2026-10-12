"""Build, check or explicitly adopt the candidate database (no HTTP administration)."""

import argparse
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DB_PATH  # noqa: E402
from app.ingest import adopt_existing, build_database, check_database  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rebuild", action="store_true", help="explicitly replace candidates and notes after successful embedding")
    mode.add_argument("--no-reset", "--ensure", dest="ensure", action="store_true", help="import missing seeds; skip verified existing seeds (default)")
    mode.add_argument("--check", action="store_true", help="read-only database integrity/configuration check")
    mode.add_argument("--adopt-existing", action="store_true", help="assert legacy vectors used the CURRENT model and format; verify identities/vectors and add metadata")
    parser.add_argument("--file", type=Path, help="resume source in the original ID-free format")
    args = parser.parse_args()
    try:
        if args.check:
            check_database(args.file)
            print("Database is complete and compatible")
        elif args.adopt_existing:
            print(adopt_existing(args.file))
        else:
            result = build_database(reset=args.rebuild, path=args.file)
            print(f"inserted {result['inserted']} candidates ({result['total']} total) into {DB_PATH}")
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(1, f"Build failed: {exc}\n")
