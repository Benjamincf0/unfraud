# Fraud Hunter Frontend

React + Vite reviewer UI for the Fraud Hunter challenge. The frontend focuses on the human review workflow: ranked flags, clear reasons, triage actions, keyboard navigation, undo, search, filtering, audit trail, and cost threshold tuning.

## Run

```bash
npm install
npm run dev
```

Set `VITE_API_BASE_URL` when the backend runs on a different origin:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

## Backend contract

The app requests flagged transactions from:

```text
GET /api/review/flags
```

Expected response shape:

```ts
TransactionFlag[]
```

See `src/types.ts` for the full contract. If the endpoint is unavailable or returns invalid data, the frontend uses local sample data so the reviewer experience remains demoable.

## Reviewer controls

- `j` or `ArrowDown`: next transaction
- `k` or `ArrowUp`: previous transaction
- `a`: approve
- `d`: dismiss
- `e`: escalate
- `u`: undo

## Frontend scope

The model, detector, CSV ingestion, and updated flagged CSV are owned by the backend/model agents. This frontend keeps the API boundary explicit and avoids duplicating fraud detection logic in the browser.
