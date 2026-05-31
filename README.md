# Fraud Hunter

Fraud Hunter is a reviewer-first fraud triage app for the MCP Hacks challenge. It ingests `transactions.csv`, scores all 1,000 transactions, explains each alert, and gives a human reviewer a keyboard-driven queue for approve, dismiss, and escalate decisions.

## Run

```bash
make dev
```

This starts FastAPI on `http://127.0.0.1:8000` and Vite on the frontend dev URL printed by Vite. Upload `transactions.csv` in the browser to begin review.

## Test

```bash
make test
```

This runs the backend pytest suite and builds the frontend.

## Export The Flagged CSV

```bash
make export
```

The committed `analyzed_transactions.csv` is the current detector output for the challenge dataset. It preserves the original transaction fields and adds `is_fraud`, `fraud_score`, `fraud_reasons`, explainability JSON, card/cross-card feature columns, and review handoff columns.

## Backend documentation

Full backend guides (API, both scorers, ML training, operations) for technical and non-technical readers: [backend/docs/README.md](backend/docs/README.md).

## Detection Strategy

The detector in `backend/fraud_scorer.py` (wired through `backend/main.py`) uses a weighted rules model built around temporal baselines rather than static absolute thresholds:

- Per-card baselines: historical median/mean/std amount, prior category/device/IP/country usage, and transaction count before the current row.
- Per-card × merchant_category baselines: amount ratio and z-score within the same category on that card (e.g. a large grocery charge vs this card's usual grocery spend).
- Cross-card signals: shared device fanout, shared IP fanout, merchant transaction bursts over 30 minutes, and unique-card merchant bursts over 2 hours.
- Geo scoring is conditional: cross-border alone does not flag; it must pair with amount or identity anomalies, or be a first-seen merchant country for the card.
- Explainability: every scored transaction carries ordered reason objects with labels, details, weights, signal type, observed value, and baseline.

On the provided dataset the current detector processes all 1,000 rows and flags 61 transactions. The hidden labels are not available, so tuning is intentionally conservative to avoid flooding the reviewer queue.

## Reviewer Workflow

The React UI opens directly into a working review queue after upload. Reviewers can search, filter by decision, inspect card history, tune the queue with threshold and cost sliders, and use keyboard shortcuts:

- `j` / `ArrowDown`: next transaction
- `k` / `ArrowUp`: previous transaction
- `a`: approve
- `d`: dismiss
- `e`: escalate
- `u`: undo

Review actions are synced to the backend, surfaced in an in-session audit log, and included in exported CSVs.

## With Another Week

- Calibrate the score threshold against labeled challenge results and report precision/recall/F1.
- Persist uploads and review decisions in SQLite or Postgres instead of process memory.
- Add reviewer notes and bulk export controls to the UI.
- Add end-to-end browser tests for keyboard review, undo, and export.
- Replace hand-tuned weights with a calibrated model only if labeled data is available and leakage is controlled.
