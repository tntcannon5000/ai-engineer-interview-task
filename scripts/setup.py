"""Shared setup logic for Bash and PowerShell launchers."""

import argparse
import os
import subprocess
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import build_database, check_database, import_notes  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="replace all candidates and notes, then reimport supplied notes")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--port", type=int, default=os.environ.get("PORT", "8000"))
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        parser.error("GEMINI_API_KEY is missing; set it in .env or export it in the shell")
    try:
        print("==> Checking/importing seed candidates", flush=True)
        print(build_database(reset=args.rebuild), flush=True)
        check_database()
        print("==> Importing supplied notes", flush=True)
        print(import_notes(), flush=True)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(1, f"Setup failed: {exc}\n")
    if not args.no_run:
        print(f"==> Recruiter UI: http://127.0.0.1:{args.port}/ (API docs: /docs)", flush=True)
        return subprocess.call([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                                "--port", str(args.port)], cwd=Path(__file__).resolve().parent.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
