# Fraud Hunter Frontend

React + Vite reviewer UI for the Fraud Hunter challenge. The frontend starts with CSV upload, sends the file to the backend detector, then opens a human review workflow for the returned flagged transactions.

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

The app uploads the transaction file to:

```text
POST /api/review/upload
```

Request body is multipart form data with a `file` field containing `transactions.csv`.

Expected response shape:

```ts
TransactionFlag[]
```

The frontend also accepts `{ items: TransactionFlag[] }`. See `src/types.ts` for the full transaction contract.

## Reviewer controls

- `j` or `ArrowDown`: next transaction
- `k` or `ArrowUp`: previous transaction
- `a`: approve
- `d`: dismiss
- `e`: escalate
- `u`: undo

## Frontend scope

The model, detector, CSV parsing, and updated flagged CSV are owned by the backend/model agents. This frontend keeps the API boundary explicit and avoids duplicating fraud detection logic in the browser.
