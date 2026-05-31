# Backend documentation

This folder explains the **Fraud Hunter** backend in plain language. You do not need a programming or machine-learning background to follow it.

**Project-wide docs (recommended first):**

- [../../docs/README.md](../../docs/README.md) — documentation hub for all users
- [../../docs/getting-started.md](../../docs/getting-started.md) — using the review UI without reading code
- [../../docs/architecture.md](../../docs/architecture.md) — how the React frontend and this backend communicate

Technical quick reference (endpoints, install commands) remains in [../README.md](../README.md).

## Who this is for

- Product owners and fraud analysts who need to understand what the system does
- Compliance or audit teams who care about **why** a transaction was flagged
- Engineers onboarding to the project (start here, then read the code)

## How to read these guides

| Document | What you will learn |
|----------|---------------------|
| [01-overview.md](01-overview.md) | What the backend is, the two ways it scores fraud, and the big picture |
| [02-data-and-workflow.md](02-data-and-workflow.md) | What data goes in, what comes out, and the reviewer journey |
| [03-api-guide.md](03-api-guide.md) | Every HTTP endpoint, explained as user actions rather than code |
| [04-heuristic-scoring.md](04-heuristic-scoring.md) | The default **rules-based** detector (no trained model file required) |
| [05-machine-learning-model.md](05-machine-learning-model.md) | LightGBM, two-tier guardrails, SHAP, design rationale & limitations |
| [06-training-and-tuning.md](06-training-and-tuning.md) | How the ML model is trained, evaluated, and optionally tuned |
| [07-operations.md](07-operations.md) | Files on disk, drift monitoring, scripts, tests, and limitations |

## Glossary (short)

| Term | Plain meaning |
|------|----------------|
| **Transaction** | One card payment row (amount, merchant, time, country, etc.) |
| **Fraud score** | A number from 0 to 1: higher means “more suspicious” |
| **Flag / alert** | A transaction marked for human review (`is_fraud = true` in exports) |
| **Heuristic scorer** | Hand-crafted rules and weights — works without a model file |
| **ML scorer** | A trained model (LightGBM) loaded from `algo/ops/fraud_model.pkl` |
| **Baseline** | What is “normal” for this card before this transaction (e.g. typical amount) |
| **Cross-card signal** | Suspicious pattern involving many cards (shared device, merchant burst) |
| **Guardrail** | A hard rule that can flag fraud even when the ML score is low |
| **SHAP** | A method that lists which factors pushed the ML score up for one alert |
| **Temporal split** | Training on older data and testing on newer data (like real life) |

## One-sentence summary

The backend accepts a CSV of transactions, scores each row for fraud risk with explainable reasons, lets reviewers approve or dismiss alerts, and exports enriched CSVs for audit and handoff.
