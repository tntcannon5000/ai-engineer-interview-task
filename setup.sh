#!/usr/bin/env bash
# Setup, import supplied notes and run the API.
# ./setup.sh | PORT=8002 ./setup.sh | ./setup.sh --rebuild | ./setup.sh --no-run
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if [[ "${1:-}" != "--help" && "${1:-}" != "-h" ]]; then
  uv sync --locked --quiet
fi
exec uv run python scripts/setup.py "$@"
