#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Get it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [ ! -f "samples/sample_paper.pdf" ]; then
  uv run python scripts/generate_sample_paper.py
fi

exec uv run readtome "$@"
