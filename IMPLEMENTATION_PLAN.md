# Implementation Plan

## Architecture

- **Backend:** FastAPI with pandas-based feature engineering and scoring (`fraud_scorer.py`, optional `ml_fraud_scorer.py`).
- **Frontend:** React + Vite with TanStack Query, paginated queue loading, and backend review sync.
- **Data handoff:** CSV upload to `/upload`, summary from `/analysis/summary/{file_hash}`, paginated queue from `/analysis/queue/{file_hash}`, on-demand detail from `/analysis/transaction/{file_hash}/{id}`, decisions through `/review/{file_hash}/{transaction_id}/{action}`, audit from `/review-log/{file_hash}`, CSV export from `/export/{file_hash}`.

See [docs/architecture.md](docs/architecture.md) for the full sequence diagram.

## Detection pipeline

1. Parse and validate required CSV columns.
2. Sort transactions by timestamp and transaction ID.
3. Build prior-only per-card baselines for amount, category, device, IP, and country.
4. Build cross-card fanout and merchant burst features across the uploaded file.
5. Compute weighted component scores and high-confidence rules (heuristic), or run LightGBM + guardrails (ML when artifact present).
6. Attach ordered explainability payloads and export-ready columns.

## Reviewer UI

- Upload screen sends the CSV to the backend; the queue loads flagged rows from paginated `/analysis/queue` responses (no local CSV parsing).
- Queue view shows ranked candidates one at a time with a compact sidebar list.
- Keyboard shortcuts support next, previous, approve, dismiss, escalate, and undo.
- Threshold and false-positive/missed-fraud cost sliders tune the visible queue client-side.
- Heuristic vs model scoring toggle when `fraud_model.pkl` is available.
- Card history, related transactions, and cross-card panels fetch on demand.
- Session metadata persists in browser `localStorage`; review decisions persist on the backend for the process lifetime.
- In-session audit log records review actions.

## Work division

- **Backend/model:** ingestion, feature engineering, heuristic and ML scoring, review API, export, tests.
- **Frontend:** upload flow, paginated queue, card context, keyboard handling, audit surface, dual-scorer UI.
- **Documentation:** [docs/](docs/) hub, backend guides, README, PRD, implementation plan, hypothesis log, generated output artifact.

## Deliberate skips

- Durable storage was skipped to keep the demo lightweight; review decisions are included in CSV export for handoff.
- Full browser automation was skipped; frontend build and backend API tests cover critical paths.
- Automatic retraining from reviewer dismissals within a session was skipped; dismissals are audited and threshold sliders allow manual exploration.
