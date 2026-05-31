# Implementation Plan

## Architecture

- Backend: FastAPI with pandas-based feature engineering and scoring.
- Frontend: React + Vite with local queue state and backend review sync.
- Data handoff: CSV upload to `/upload`, analysis from `/analysis/all/{file_hash}`, decisions through `/review/{file_hash}/{transaction_id}/{action}`, and CSV export from `/export/{file_hash}`.

## Detection Pipeline

1. Parse and validate required CSV columns.
2. Sort transactions by timestamp and transaction ID.
3. Build prior-only per-card baselines for amount, category, device, IP, and country.
4. Build cross-card fanout and merchant burst features across the uploaded file.
5. Compute weighted component scores and high-confidence rules.
6. Attach ordered explainability payloads and export-ready columns.

## Reviewer UI

- Upload screen sends the CSV to the backend and joins analysis results with original rows.
- Queue view shows ranked candidates one at a time with a compact list for navigation.
- Keyboard shortcuts support next, previous, approve, dismiss, escalate, and undo.
- False-positive and missed-fraud cost sliders derive the visible risk cutoff.
- Session audit log records review actions and lets reviewers reopen recent decisions.

## Work Division

- Backend/model: ingestion, feature engineering, scoring, review API, export.
- Frontend: upload flow, review queue, card context, keyboard handling, audit surface.
- Documentation: README, PRD, implementation plan, hypothesis log, generated output artifact.

## Deliberate Skips

- Durable storage was skipped to keep the demo lightweight; review decisions are included in CSV export for handoff.
- Supervised LightGBM code remains experimental and is not wired into the challenge app because hidden labels are unavailable and leakage risk is high.
- Full browser automation was skipped; frontend build and backend API tests cover the critical paths.
