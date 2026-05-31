#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "Installing backend dependencies..."
(
  cd "$ROOT_DIR/backend"
  uv sync
)

echo "Installing frontend dependencies..."
(
  cd "$ROOT_DIR/frontend"
  npm install
)

echo "Starting backend on http://127.0.0.1:8000"
(
  cd "$ROOT_DIR/backend"
  uv run uvicorn main:app --reload
) &
BACKEND_PID=$!

echo "Starting frontend on http://127.0.0.1:5173"
(
  cd "$ROOT_DIR/frontend"
  npm run dev
) &
FRONTEND_PID=$!

wait "$BACKEND_PID" "$FRONTEND_PID"
