.PHONY: build dev export export-ml score-ml test

dev:
	./scripts/dev.sh

test:
	cd backend && uv run --extra test python -m pytest -q
	cd frontend && npm run build

build:
	cd frontend && npm run build

export:
	cd backend && uv run python export_challenge_csv.py

score-ml:
	cd backend && uv run python -m scripts.score_transactions ../transactions.csv

export-ml:
	cd backend && uv run python -m scripts.score_transactions ../transactions.csv -o ../ml_analyzed_transactions.csv
