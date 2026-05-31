# Getting started — using Unfraudify

This guide is for **reviewers and new users**. No programming knowledge required.

## What you need

- The Unfraudify app running locally (see [Run the app](#run-the-app) below).
- A transaction CSV file. The challenge provides `transactions.csv` with 1,000 payments.

## Run the app

Open a terminal in the project folder and run:

```bash
make dev
```

Two things start:

1. **Backend** (the scoring engine) at `http://127.0.0.1:8000`
2. **Frontend** (the review UI) — the terminal prints a URL, usually `http://localhost:5173`

Open the frontend URL in your browser.

## Step 1 — Upload your file

On the upload screen, choose your `transactions.csv` file (drag-and-drop or file picker).

What happens behind the scenes:

- The file is sent to the backend.
- The server checks that required columns are present and stores the data in memory.
- You receive a short summary: total transactions and how many were flagged.

When upload finishes, you land directly in the **review queue**.

## Step 2 — Work the review queue

The queue shows **one suspicious transaction at a time** with full context:

- Payment details (amount, merchant, channel, countries)
- **Risk score** and **reasons** — why the system flagged it
- **Card history** — what is normal for this card (typical spend, usual categories)
- **Related activity** — other payments on the same card, device, or IP

### Review actions

For each transaction, choose one of:

| Action | When to use it |
|--------|----------------|
| **Approve** | You agree this is fraud (or the alert is valid) |
| **Dismiss** | False alarm — legitimate customer activity |
| **Escalate** | Uncertain — needs a senior analyst or more investigation |

After you decide, the next transaction loads automatically.

### Keyboard shortcuts

These make review fast once you learn them:

| Key | Action |
|-----|--------|
| `j` or `↓` | Next transaction |
| `k` or `↑` | Previous transaction |
| `a` | Approve |
| `d` | Dismiss |
| `e` | Escalate |
| `u` | Undo last decision |

### Filter and search

Use the tabs above the queue to switch views:

- **Review queue** — only flagged transactions (`is_fraud`), sorted by risk score
- **All transactions** — every scored row (useful for exploring borderline cases)

Within the review queue, filter by decision:

- **Pending**, **approved**, **dismissed**, or **escalated**

Use the sidebar to **search** by transaction ID, card, merchant, or other fields.

### Tune what you see (threshold sliders)

Two sliders help you explore trade-offs without re-running detection:

- **Risk threshold** — raise it to see fewer, higher-confidence alerts; lower it to see more.
- **False-positive cost** — shifts the effective threshold when you care more about avoiding false alarms vs catching more fraud.

These adjust **which flagged items appear in your working queue** in the UI. They do not change the underlying scores stored on the server.

### Heuristic vs model scoring

By default the app uses the **heuristic** scorer (hand-crafted rules). If a trained ML model is installed on the backend, switch to **Model** scoring with the toggle in the queue toolbar.

In model mode you get:

- A **status breakdown** in the header: how many queued rows came from the model threshold alone, from a strict **alert rule** alone, from both, or were only **elevated** by a soft rule (higher score but not auto-queued)
- **Queue cause tabs** (when the review queue is active): **All queued**, **Model only**, **Alert rule only**, and **Model + alert** — slice the queue by what triggered each flag
- Side-by-side **heuristic vs model** detail when you open a transaction

Alert rules are high-confidence guardrails (extreme amount, IP fanout, merchant burst, etc.) that can queue a row even when the model probability is below threshold. See [ML model docs](../backend/docs/05-machine-learning-model.md) for the full hybrid logic.

## Step 3 — Audit trail

Every decision you make is:

- Saved on the **backend** for the current session
- Shown in the in-app **audit log** so you can revisit recent actions
- Included when you **export** results (see below)

**Undo** (`u`) sends a “pending” action back to the server, clearing your previous decision for that transaction.

## Sessions and browser reload

The app remembers **upload session names** in your browser (file name and upload time). If you refresh the page while the backend is still running, it reconnects using the stored file ID.

**If you restart the backend**, you must upload the CSV again — server memory is cleared on restart.

## Export results

To download a CSV with scores, explanations, and your review decisions:

- Use the backend export endpoint directly: `GET /export/{file_hash}` (developers can open `http://127.0.0.1:8000/docs` for interactive API docs).
- Or from the repo root, run `make export` to regenerate the committed challenge output file using the heuristic scorer.

Exported files include original transaction columns plus fraud scores, reason text, JSON explainability fields, and review metadata.

## Tips for effective review

1. **Read the reasons first** — each flag should cite at least one concrete signal (unusual amount, new device, shared IP across cards, etc.).
2. **Check card history** — a large charge may be normal for one customer and alarming for another.
3. **Use cross-card panels** — some fraud patterns only appear when the same device or merchant hits many cards.
4. **Escalate when unsure** — dismiss only when you would comfortably let the payment through.
5. **Work pending first** — filter to pending to see what is left to triage.

## Troubleshooting

| Problem | Likely cause | What to do |
|---------|--------------|------------|
| Upload fails | Missing CSV columns or bad timestamps | Check the file matches the expected format (see [backend docs on CSV format](../backend/docs/02-data-and-workflow.md)) |
| “File not found” after reload | Backend was restarted | Upload the CSV again |
| Empty queue | Threshold sliders set too high | Lower the risk threshold |
| Model toggle unavailable | No trained model on disk | Use heuristic mode; see [ML training docs](../backend/docs/06-training-and-tuning.md) |

## Next steps

- [Architecture](architecture.md) — how the browser and server communicate
- [Backend overview](../backend/docs/01-overview.md) — how fraud scoring works
- [Heuristic scoring](../backend/docs/04-heuristic-scoring.md) — what signals the default detector uses
