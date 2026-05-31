# API guide (for humans)

The backend is a **REST API** built with FastAPI. “REST” here simply means: the frontend calls specific URLs with standard actions (GET = read, POST = write).

Base URL in development: `http://127.0.0.1:8000`

Interactive docs (for developers): `http://127.0.0.1:8000/docs`

## Health and configuration

### `GET /`

**Purpose:** Check that the server is running.

**Response:** `{ "status": "ok" }`

### `GET /scoring/status`

**Purpose:** See which scoring engines are available.

**Response example:**

```json
{
  "heuristic": true,
  "ml_model_path": ".../algo/ops/fraud_model.pkl",
  "ml_model_available": true
}
```

- `heuristic` is always available.
- `ml_model_available` is `false` until you train and save `fraud_model.pkl`.

## File upload

### `POST /upload`

**Purpose:** Send a transaction CSV and receive an ID for all follow-up calls.

**Input:** Multipart file upload (the CSV binary).

**Success response:**

```json
{
  "file_hash": "abc123...",
  "message": "File uploaded successfully"
}
```

**Common errors:**

| HTTP code | Meaning |
|-----------|---------|
| 400 | CSV invalid, missing required columns, or bad timestamps |
| (duplicate) | Same file already uploaded — returns existing hash |

**Required CSV columns:** See [02-data-and-workflow.md](02-data-and-workflow.md).

## Analysis (read scored transactions)

All analysis endpoints accept optional query parameter:

- `use_model=false` (default) — heuristic scorer
- `use_model=true` — ML scorer (503 error if model file missing)

### `GET /analysis/all/{file_hash}`

**Purpose:** Full scored dataset for the review queue and dashboards.

**Returns:** A list of rich objects per transaction, including:

- Core fields (id, time, card, amount, merchant, channel, countries)
- `is_fraud`, `fraud_score`
- `reasons` — short labels derived from the score breakdown
- `score_breakdown` — detailed weighted reasons
- `card_baseline`, `cross_card_signals`, `graph_features`, `card_amount_series`

### `GET /analysis/user/{file_hash}/{card_id}`

**Purpose:** All transactions for one card, in time order — for “card history” panels.

**Note:** Scoring still used the **full file** (cross-card signals stay correct); this endpoint only **filters** the result.

### `GET /analysis/ip/{file_hash}/{ip_address}`

**Purpose:** All transactions sharing an IP — useful when investigating account takeover rings.

## Review actions

### `POST /review/{file_hash}/{transaction_id}/{action}`

**Purpose:** Record what the reviewer decided.

**URL `action`:** one of `approve`, `dismiss`, `escalate`, `pending`

**JSON body (must match URL action):**

```json
{
  "action": "dismiss",
  "reviewer_notes": "Customer confirmed purchase by phone"
}
```

**Success response:** Confirmation with `reviewed_at` timestamp (UTC).

**Errors:**

| Code | Cause |
|------|-------|
| 404 | Unknown file_hash or transaction_id |
| 400 | Invalid action or body mismatch |

### `GET /review/{file_hash}/audit`

**Purpose:** Chronological audit trail of review actions for this upload (newest first).

### `GET /review-log/{file_hash}`

**Purpose:** Same underlying data as audit, shaped as a review log list (also sorted by time).

## Export

### `GET /export/{file_hash}`

**Purpose:** Download one CSV with transactions, fraud columns, and review columns.

**Query:** `use_model` as above.

**Response:** File download (`analyzed_transactions_<hash-prefix>.csv`).

## Typical frontend sequence

```mermaid
flowchart TD
  A[POST /upload] --> B[GET /analysis/all]
  B --> C[Reviewer works queue]
  C --> D[POST /review/.../action]
  D --> C
  C --> E[GET /export]
```

## Error handling philosophy

The API returns **clear HTTP status codes** instead of silent failures:

- **404** — You referenced a file or transaction that was never uploaded.
- **400** — The CSV or review request is malformed.
- **503** — ML scoring requested but the model artifact is not on disk.

This helps the UI show actionable messages (“train the model first” vs “file not found”).
