#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

(
  cd backend
  uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
) &
BACKEND_PID=$!

cd frontend
npm run dev
