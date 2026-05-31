.PHONY: build dev export test

dev:
	./scripts/dev.sh

test:
	cd backend && uv run --extra test python -m pytest -q
	cd frontend && npm run build

build:
	cd frontend && npm run build

export:
	cd backend && uv run python export_challenge_csv.py
