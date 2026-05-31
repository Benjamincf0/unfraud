# Fraud Hunter

Fraud Hunter is a **reviewer-first fraud triage app** built for the MCP Hacks challenge. It ingests a CSV of card transactions, scores every row for fraud risk, explains each alert in plain language, and gives a human reviewer a keyboard-driven queue to approve, dismiss, or escalate decisions.

**New here?** Start with the [documentation hub](docs/README.md) — it includes a non-technical [getting started guide](docs/getting-started.md) and an [architecture overview](docs/architecture.md) of how the frontend and backend work together.

## What it does

| Step | What happens |
|------|----------------|
| Upload | You send `transactions.csv` to the backend (1,000 rows in the challenge dataset) |
| Score | The detector compares each payment to that card’s history and to patterns across all cards |
| Explain | Every flagged transaction gets human-readable reasons (unusual amount, new device, shared IP, etc.) |
| Review | You work through alerts one at a time with keyboard shortcuts and card context |
| Export | Decisions and scores can be downloaded as an enriched CSV |

The app **prioritizes and explains** — it does not block payments automatically.

## Run

```bash
make dev
```

This starts FastAPI on `http://127.0.0.1:8000` and the React frontend (URL printed in the terminal, usually `http://localhost:5173`). Upload `transactions.csv` in the browser to begin review.

## Test

```bash
make test
```

Runs the backend pytest suite and builds the frontend.

## Export the flagged CSV

```bash
make export
```

The committed `analyzed_transactions.csv` is the current heuristic detector output for the challenge dataset.

## Score with the trained ML model

After training (`cd backend && uv run python -m scripts.train_fraud_model`):

```bash
make score-ml
```

Prints how many of the 1,000 challenge rows the hybrid LightGBM + guardrail detector flags. To also write an enriched CSV:

```bash
make export-ml
```

Creates `ml_analyzed_transactions.csv` at the repo root. It preserves the original transaction fields and adds fraud scores, reasons, explainability JSON, card/cross-card feature columns, and review handoff columns.

## Documentation

| Audience | Document |
|----------|----------|
| Everyone (start here) | [docs/README.md](docs/README.md) |
| Reviewers / non-technical users | [docs/getting-started.md](docs/getting-started.md) |
| System design | [docs/architecture.md](docs/architecture.md) |
| Backend API & scoring depth | [backend/docs/README.md](backend/docs/README.md) |
| Frontend developers | [frontend/README.md](frontend/README.md) |
| Product context | [PRD.md](PRD.md) |
| Challenge brief | [CHALLENGE.md](CHALLENGE.md) |

## Detection strategy

The default **heuristic** scorer in `backend/fraud_scorer.py` uses weighted rules built around temporal baselines rather than static absolute thresholds:

- **Per-card baselines:** historical median/mean/std amount, prior category/device/IP/country usage, and transaction count before the current row.
- **Per-card × merchant_category baselines:** amount ratio and z-score within the same category on that card.
- **Cross-card signals:** shared device fanout, shared IP fanout, merchant transaction bursts over 30 minutes, and unique-card merchant bursts over 2 hours.
- **Reviewer feedback:** when a reviewer escalates a transaction, other rows sharing that IP receive a session risk boost and a “Previously escalated IP” explanation.
- **Geo scoring is conditional:** cross-border alone does not flag; it must pair with amount or identity anomalies, or be a first-seen merchant country for the card.
- **Explainability:** every scored transaction carries ordered reason objects with labels, details, weights, signal type, observed value, and baseline.

An optional **ML scorer** (LightGBM + guardrail rules) is available when `algo/ops/fraud_model.pkl` is present; the UI can compare both. See [backend/docs/05-machine-learning-model.md](backend/docs/05-machine-learning-model.md).

On the provided dataset the heuristic detector processes all 1,000 rows and flags 61 transactions. Hidden labels are not available, so tuning is intentionally conservative to avoid flooding the reviewer queue.

## Reviewer workflow

The React UI opens into a working review queue after upload. Reviewers can search, filter by decision, inspect card history and cross-card links, tune the queue with threshold and cost sliders, and use keyboard shortcuts:

| Key | Action |
|-----|--------|
| `j` / `ArrowDown` | Next transaction |
| `k` / `ArrowUp` | Previous transaction |
| `a` | Approve |
| `d` | Dismiss |
| `e` | Escalate |
| `u` | Undo |

Review actions sync to the backend, appear in an in-session audit log, update shared-IP risk when escalated, and are included in exported CSVs.

## With another week

- Calibrate the score threshold against labeled challenge results and report precision/recall/F1.
- Persist uploads and review decisions in SQLite or Postgres instead of process memory.
- Add reviewer notes and bulk export controls to the UI.
- Add end-to-end browser tests for keyboard review, undo, and export.
- Close the loop from dismissals to threshold or rule adjustments automatically.
