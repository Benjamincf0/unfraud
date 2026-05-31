# Operations, scripts, and maintenance

## Running the server

From repository root (recommended):

```bash
make dev
```

Starts FastAPI (port 8000) and the frontend dev server.

Manual backend only:

```bash
cd backend
uv sync
uvicorn main:app --reload
```

Dependencies are managed with **uv** (`pyproject.toml`, `uv.lock`). Python 3.14+ is specified in the project file.

## Project layout (backend)

| Path | Role |
|------|------|
| `main.py` | HTTP API, in-memory storage, routing to scorers |
| `fraud_scorer.py` | Heuristic batch scorer |
| `ml_fraud_scorer.py` | ML batch scorer + API column adapter |
| `algo/algo.py` | Features, training, guardrails, SHAP, pipeline class |
| `algo/lgbm_params.py` | Default and tuned LightGBM settings |
| `algo/tune_lgbm.py` | Optuna hyperparameter search |
| `algo/ops/fraud_model.pkl` | Trained model artifact (binary) |
| `algo/ops/drift_metrics.json` | Weekly PR-AUC history |
| `algo/ops/best_lgbm_params.json` | Optional tuned hyperparameters |
| `scripts/train_fraud_model.py` | CLI to train and save model |
| `scripts/score_transactions.py` | Offline ML scoring + flag summary for challenge CSV |
| `export_challenge_csv.py` | Offline heuristic export for challenge |
| `tests/` | Automated tests (API, scorers, algo) |

## Project layout (frontend)

See [../docs/architecture.md](../docs/architecture.md). Key paths:

| Path | Role |
|------|------|
| `frontend/src/App.tsx` | Upload vs queue routing |
| `frontend/src/api/review.ts` | Backend HTTP client |
| `frontend/src/components/ReviewQueue.tsx` | Review workflow |

## Auxiliary scripts

| Script | Purpose |
|--------|---------|
| `convert_dataset_script.py` | Dataset format conversions (training prep) |
| `split_csv.py` | Split large CSVs |
| `export_challenge_csv.py` | Write root `analyzed_transactions.csv` |
| `scripts/score_transactions.py` | Score CSV with ML; `make score-ml` / `make export-ml` |

## Testing

```bash
make test
```

Runs `pytest` in `backend/tests` and builds the frontend. Test modules cover:

- API upload, analysis, review, export
- Heuristic detection behavior
- ML scoring when artifact present
- Algorithm utilities and params

## Artifacts and git

| File | Usually committed? |
|------|-------------------|
| `fraud_model.pkl` | Yes if you ship a default trained model |
| `drift_metrics.json` | Optional (ops telemetry) |
| `best_lgbm_params.json` | Yes if you standardize on tuned params |
| `optuna.db` | No (gitignored, local experiment DB) |

Large training CSVs (`fraudTrain_part1.csv`) may live locally; check `.gitignore` before assuming they are in the repo.

## Security and data handling notes

- Uploads are held **in process memory** — not encrypted at rest on the server.
- No authentication on API endpoints in the demo — suitable only for local/trusted networks.
- Do not commit `.env` files or credentials if you add external integrations later.

## Performance expectations

- First analysis of 1,000 rows: heuristic scoring is fast (seconds).
- ML scoring adds feature engineering + model inference; still suitable for batch challenge size.
- Results are **cached** per `(file_hash, use_model)` until the server restarts or the same file is re-uploaded.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `503` on analysis with `use_model=true` | No `fraud_model.pkl` | Run training script |
| `400` on upload | Missing columns or bad timestamps | Fix CSV per [02-data-and-workflow.md](02-data-and-workflow.md) |
| Empty review queue | Threshold too high or low fraud rate | Check `fraud_score` distribution in export |
| ML and heuristic disagree | Different logic by design | Compare exports with `use_model` true/false |
| Server “lost” reviews | Process restarted | Expected in demo; add DB for production |

## Production checklist (future)

Items called out in the project README as improvements:

- Persist uploads and reviews in SQLite or Postgres
- Authenticate API users
- Calibrate thresholds on labeled production outcomes
- Scheduled retrain jobs wired to `DriftMonitor`
- Separate environments for training vs inference

## Related documentation

- [README.md](README.md) — index and glossary
- [../README.md](../README.md) — developer quick reference (endpoints, install)
