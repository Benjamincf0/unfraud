# Fraud Detection Backend

FastAPI service for Fraud Hunter: upload transaction CSVs, score fraud risk, support human review, and export enriched results.

## Documentation (start here)

**Non-technical and full-system guides:** [../docs/README.md](../docs/README.md)

| Guide | Topic |
|-------|--------|
| [docs/getting-started.md](../docs/getting-started.md) | Using the review UI — no code required |
| [docs/architecture.md](../docs/architecture.md) | Frontend ↔ backend data flow |
| [../docs/README.md](../docs/README.md) | Documentation hub |

**Backend depth:**

| Guide | Topic |
|-------|--------|
| [docs/01-overview.md](docs/01-overview.md) | Purpose, two scorers, architecture |
| [docs/02-data-and-workflow.md](docs/02-data-and-workflow.md) | CSV format, upload → review → export |
| [docs/03-api-guide.md](docs/03-api-guide.md) | HTTP API in plain language |
| [docs/04-heuristic-scoring.md](docs/04-heuristic-scoring.md) | Default rules-based detector |
| [docs/05-machine-learning-model.md](docs/05-machine-learning-model.md) | LightGBM, hybrid scoring, SHAP |
| [docs/06-training-and-tuning.md](docs/06-training-and-tuning.md) | Training, metrics, Optuna |
| [docs/07-operations.md](docs/07-operations.md) | Run, test, artifacts, troubleshooting |

## Quick reference — endpoints

### File upload
- **POST /upload** — Upload CSV → `{file_hash, message}`

### Analysis (`?use_model=true` for ML)
- **GET /analysis/summary/{file_hash}** — Counts, queue stats, ML queue-cause breakdown, ML availability
- **GET /analysis/queue/{file_hash}** — Paginated review queue (primary UI endpoint)
- **GET /analysis/transaction/{file_hash}/{transaction_id}** — Full explainability for one row
- **GET /analysis/related/{file_hash}/{transaction_id}** — Same card / device / IP
- **GET /analysis/all/{file_hash}** — All transactions with explainability (bulk)
- **GET /analysis/user/{file_hash}/{card_id}** — Per-card history
- **GET /analysis/ip/{file_hash}/{ip_address}** — Per-IP slice

### Review
- **POST /review/{file_hash}/{transaction_id}/{action}** — `approve` | `dismiss` | `escalate` | `pending`
- **GET /review/{file_hash}/audit** — Audit trail
- **GET /review-log/{file_hash}** — Review log

### Other
- **GET /** — Health
- **GET /scoring/status** — Heuristic vs ML availability
- **GET /export/{file_hash}** — Download analyzed CSV

## Installation

```bash
uv sync
```

## Running the server

```bash
uvicorn main:app --reload
```

Or from repo root: `make dev`

## Train ML model (optional)

```bash
uv run python -m scripts.train_fraud_model
```

Writes `algo/ops/fraud_model.pkl`. See [docs/06-training-and-tuning.md](docs/06-training-and-tuning.md).

## Testing

```bash
uv run --extra test python -m pytest -q
```

## Offline hyperparameter tuning (optional)

```bash
uv sync --extra tune
uv run --extra tune python -m algo.tune_lgbm fraudTrain_part1.csv --n-trials 40
```

See [docs/06-training-and-tuning.md](docs/06-training-and-tuning.md).

## Export challenge CSV

From repo root:

```bash
make export
```

Uses the **heuristic** scorer on `transactions.csv`.

## Score challenge CSV with ML

Requires `algo/ops/fraud_model.pkl` (see training above).

```bash
make score-ml    # print flag counts for transactions.csv
make export-ml   # same + write ml_analyzed_transactions.csv
```

Or manually:

```bash
cd backend
uv run python -m scripts.score_transactions ../transactions.csv --list-flagged
```
