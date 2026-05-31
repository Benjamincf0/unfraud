# Unfraudify — documentation

Unfraudify is a **fraud triage tool**: it reads a batch of card transactions, flags the ones that look suspicious, explains why each flag fired, and gives a human reviewer a fast queue to approve, dismiss, or escalate each alert.

This folder is the main entry point for **everyone** — reviewers, product owners, and engineers. You do not need to read source code to use the app.

## Who should read what

| If you are… | Start here |
|-------------|------------|
| A reviewer or non-technical user who wants to try the app | [Getting started](getting-started.md) |
| Someone who wants to understand how the pieces fit together | [Architecture](architecture.md) |
| A developer running or extending the backend | [Backend docs](../backend/docs/README.md) |
| A developer working on the React UI | [Frontend README](../frontend/README.md) |

## What the project does (30 seconds)

Imagine you work on a payments team. Every day thousands of charges go through, and a small fraction are fraudulent — stolen cards, account takeovers, merchant scams. You cannot inspect every payment by hand, and you cannot block everything that looks slightly unusual without angering real customers.

Unfraudify helps by:

1. **Ingesting** a CSV file of transactions (the challenge dataset has 1,000 rows across 50 cards).
2. **Scoring** each transaction for fraud risk (0 = safe, 1 = very suspicious).
3. **Explaining** every alert in plain language — for example, “amount 14× this card’s usual spend” or “same device used on 8 different cards.”
4. **Queueing** flagged transactions for a human reviewer, one at a time, with keyboard shortcuts.
5. **Recording** every approve / dismiss / escalate decision in an audit trail and in exported CSVs.

The system **does not block payments automatically**. It prioritizes and explains; the human decides.

## How to run it

From the repository root:

```bash
make dev
```

Then open the URL printed by the frontend (usually `http://localhost:5173`), upload `transactions.csv`, and start reviewing.

See [Getting started](getting-started.md) for a full walkthrough.

## Documentation map

### Project-wide (this folder)

| Document | Contents |
|----------|----------|
| [getting-started.md](getting-started.md) | Upload, review queue, filters, ML queue-cause tabs, keyboard shortcuts, sliders, sessions |
| [architecture.md](architecture.md) | Frontend ↔ backend communication, data flow, scoring modes |

### Backend (technical depth)

| Document | Contents |
|----------|----------|
| [backend/docs/README.md](../backend/docs/README.md) | Index and glossary |
| [01-overview.md](../backend/docs/01-overview.md) | Backend purpose, two scoring engines |
| [02-data-and-workflow.md](../backend/docs/02-data-and-workflow.md) | CSV format, upload → review → export |
| [03-api-guide.md](../backend/docs/03-api-guide.md) | Every HTTP endpoint in plain language |
| [04-heuristic-scoring.md](../backend/docs/04-heuristic-scoring.md) | Default rules-based detector |
| [05-machine-learning-model.md](../backend/docs/05-machine-learning-model.md) | Optional LightGBM model |
| [06-training-and-tuning.md](../backend/docs/06-training-and-tuning.md) | Training and hyperparameter search |
| [07-operations.md](../backend/docs/07-operations.md) | Run, test, artifacts, limitations |

### Product and challenge context

| Document | Contents |
|----------|----------|
| [PRD.md](../PRD.md) | Product requirements — user, goals, non-goals |
| [CHALLENGE.md](../CHALLENGE.md) | MCP Hacks challenge brief and judging criteria |
| [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) | Engineering architecture and deliberate trade-offs |
| [HYPOTHESIS_LOG.md](../HYPOTHESIS_LOG.md) | Detection experiments tried during the build |

## Glossary

| Term | Meaning |
|------|---------|
| **Transaction** | One card payment (amount, merchant, time, country, etc.) |
| **Fraud score** | A number from 0 to 1 — higher means more suspicious |
| **Flag / alert** | A transaction the system thinks deserves human review |
| **Heuristic scorer** | Hand-crafted rules — works without a trained model file |
| **ML scorer** | A trained machine-learning model (optional; needs `fraud_model.pkl`) |
| **Baseline** | What is “normal” for this card before this payment (typical amount, usual categories) |
| **Cross-card signal** | A pattern across many cards (shared device, merchant burst) |
| **File hash** | A unique ID for an uploaded CSV, used in all API calls |
| **Review decision** | What the human chose: approve, dismiss, escalate, or pending |

## Important limitations (demo scope)

These are intentional for the challenge build:

- **No database** — uploads and review decisions live in server memory until you restart the backend.
- **No login** — suitable for local or trusted networks only.
- **Batch only** — scores a CSV file, not live payment streams.
- **No auto-block** — recommendations only; humans take action outside this tool.

If the backend restarts, you must upload the same CSV again. Your browser may still remember past session names in local storage, but the server will not have the data until you re-upload.
