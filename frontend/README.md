# Fraud Hunter Frontend

React + Vite reviewer UI for Fraud Hunter. The frontend uploads a transaction CSV to the backend, loads flagged transactions from paginated analysis endpoints, and runs a keyboard-driven human review workflow.

**User guide:** [../docs/getting-started.md](../docs/getting-started.md)  
**Architecture (frontend ↔ backend):** [../docs/architecture.md](../docs/architecture.md)

## Run

```bash
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000` by default.  
Set `VITE_API_BASE_URL` when the backend runs on a different origin:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

From the repo root, `make dev` starts both backend and frontend.

## Backend contract

All HTTP calls live in `src/api/review.ts`. The base URL is `import.meta.env.VITE_API_BASE_URL ?? '/api'`.

### Upload and session open

```text
POST /upload                    → { file_hash, message }
GET  /analysis/summary/{hash}   → counts, queue stats, ml_model_available
```

After upload (or when restoring a saved session), the app fetches the summary, then loads the flagged queue in pages:

```text
GET /analysis/queue/{hash}?limit=5000&offset=0&slim=true
GET /analysis/queue/{hash}?limit=5000&offset=5000&slim=true  …
```

Optional query params on queue and related endpoints:

- `use_model=true` — ML scorer (when `fraud_model.pkl` exists)
- `flagged_only=false` — include non-flagged rows
- `transaction_id=…` — single row lookup

### Detail and context (on demand)

```text
GET /analysis/transaction/{hash}/{transaction_id}  → heuristic + optional model detail
GET /analysis/related/{hash}/{transaction_id}    → same card / device / IP
GET /analysis/user/{hash}/{card_id}              → full card timeline
```

### Review sync

```text
POST /review/{hash}/{transaction_id}/{action}    → approve | dismiss | escalate | pending
GET  /review-log/{hash}                          → audit list
```

Undo sends `pending`, which clears the review record on the backend.

### Export

```text
GET /export/{hash}?use_model=false
```

Returns the full analyzed CSV with review columns. The live UI does not use export for session restore — it reconnects via summary + queue as long as the backend still holds the upload in memory.

## Browser persistence

`src/lib/reviewSessions.ts` stores session **metadata** (file hash, original filename, upload time) in `localStorage`. On reload, the app reopens the active session by hash. If the backend was restarted, re-upload the CSV.

## Reviewer controls

| Key | Action |
|-----|--------|
| `j` or `ArrowDown` | Next transaction |
| `k` or `ArrowUp` | Previous transaction |
| `a` | Approve |
| `d` | Dismiss |
| `e` | Escalate |
| `u` | Undo |

## Frontend scope

Fraud detection and explainability payloads are owned by the backend. The frontend:

- Calls the API boundary in `api/review.ts`
- Maps backend responses to `TransactionFlag` and related types
- Manages queue navigation, filters, threshold/cost sliders (client-side queue tuning)
- Syncs review decisions and displays the audit log
- Optionally compares heuristic vs ML scores when the model artifact is available

Key files:

| Path | Role |
|------|------|
| `src/App.tsx` | Upload screen vs review queue routing |
| `src/components/ReviewQueue.tsx` | Main review workflow |
| `src/api/review.ts` | Backend HTTP client |
| `src/lib/scoringViews.ts` | Threshold helpers, session types |
