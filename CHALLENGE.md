# Unfraudify

**A 24-hour code challenge for MCP Hacks, sponsored by Valsoft.**

You have 1,000 credit card transactions. Some of them are fraud. Your job, over the next 24 hours, is to find them — and to build a tool that a real human reviewer would actually want to use.

Twenty-four hours is a lot of time. It's enough time to write something rushed and then rewrite it properly. It's enough time to discover the four fraud patterns hiding in the dataset, build per-card baselines, and ship a review queue your team would actually use on Monday morning. It is *not* enough time to do everything you might want to do — pick your battles.

This is not a Kaggle competition. We care about your code, your reasoning, and the tool you hand off at the end.

---

## The problem

You are on the trust and safety team at a payments company. Every day, thousands of transactions flow through your system, and a small fraction of them are fraudulent — stolen cards, account takeovers, merchant scams. You cannot review every transaction by hand, and you cannot block every transaction that looks slightly off. False positives cost you customers; false negatives cost you money and trust.

You have one dataset: `transactions.csv` — one month of activity across 50 cards. Build a tool that helps a human reviewer find the fraud quickly and confidently.

## What you ship

A working tool — web app, CLI with a UI layer, notebook plus a real frontend, your call — that:

1. **Ingests `transactions.csv`** and processes all 1,000 rows
2. **Flags suspicious transactions** with a score, label, or ranked list
3. **Explains its flags** — every flagged transaction must come with at least one human-readable reason
4. **Supports a reviewer** — there is a real path through your UI for a human to triage flagged transactions: approve, dismiss, escalate. Keyboard navigation expected.
5. **Has a README** — one page that explains what you built, how to run it, what your detection strategy is, and what you'd do with another week
6. **An updated transaction file** An updated version of the CSV with the identified fraud transactions marked/flagged

Plus, alongside the code:

6. **A PRD** — a short product requirements doc: who's the user, what problem are you solving for them, what does success look like, what are you explicitly *not* building. One to two pages.
7. **An implementation plan** — a brief architecture and engineering plan: tech choices, how the pieces fit together, how you divided the work across the team, what you decided to skip and why. One page.

We want to see what drove your design and engineering decisions, not just the final artifact. The PRD and implementation plan are part of what we judge.

## The dataset

`transactions.csv` has these columns:

| Column | Type | Notes |
|---|---|---|
| `transaction_id` | string | Unique, like `tx_000123` |
| `timestamp` | ISO 8601 | Local time, no timezone — treat as UTC |
| `card_id` | string | 50 unique cards in the dataset |
| `amount` | float | CAD |
| `merchant_name` | string | E.g. `Amazon.ca`, `Tim Hortons`, `QuickPay Online` |
| `merchant_category` | string | `grocery`, `gas`, `restaurant`, `online_retail`, `electronics`, `travel`, `subscription`, `entertainment`, `utilities`, `atm`, `gift_card` |
| `channel` | string | `online`, `in_person`, or `atm` |
| `cardholder_country` | string | ISO-2 country of the card's owner |
| `merchant_country` | string | ISO-2 country where the transaction was processed |
| `device_id` | string | Present only for `online` transactions |
| `ip_address` | string | Present only for `online` transactions |

The dataset is synthetic but realistic. It contains around 7% fraud, distributed across a few different patterns. We are not telling you which patterns — discovering them is part of the challenge.

## Judging

Three judges. Each team gets 7 minutes to demo and 5 minutes of questions. Scores along three axes:

**Detection quality — 40 points.** We score your flagged set against a hidden answer key. We compute precision (of your flags, how many are real fraud?), recall (of the real fraud, how much did you catch?), and F1. With 24 hours of build time, top teams should be landing F1 above 0.85; do not chase 1.0 — over-flagging will tank your precision.

**Reviewer experience — 40 points.** A non-technical judge will sit down at your tool with no instructions and try to review flagged transactions. We're looking for: clear flag reasoning, sensible defaults, a fast keyboard-driven review queue, the ability to approve/dismiss/escalate, and a layout that doesn't drown the user in fields. With 24 hours, expectations are higher than a static table — we expect interactive state, undo, search/filter, and reasonable performance on 1,000 rows.

**Engineering craft — 20 points.** Code readability, sensible structure, at least one meaningful test, a clear README, reproducibility (one-command run from a clean clone). We will read your repo, not just skim it. With 24 hours, "I didn't have time to test" is not the excuse it would be at a half-day event.

Bonus points (up to +5 total): a hypothesis log showing what you tried and what worked, an unexpected edge case caught well, a feedback loop that learns from reviewer decisions inside the session, or a thoughtful "if we had another week" section in the README.

## What good looks like at 24 hours

These are not stretch goals — at 24 hours, top submissions will have most of these. Aim accordingly.

- **Per-card anomaly scoring.** A baseline of each card's normal behavior (typical amount, typical categories, typical devices/IPs); flags fire on deviation, not on absolute values.
- **Cross-card aggregation.** At least one signal that requires looking across cards — merchant-level burst detection, device reuse across cards, IP reuse across cards. One of the four fraud patterns in the data is invisible without this.
- **Explainable score.** Don't just say "score 0.87" — say "score 0.87: new device, foreign IP, atypical category for this card, amount 14× this card's median."
- **Reviewer queue, not a table.** Flagged transactions appear one at a time with full context; reviewer hits a key to approve/dismiss/escalate and the next one loads. Undo works.
- **Feedback loop.** When the reviewer dismisses a flag, your system learns from it within the session — adjusts thresholds, suppresses similar future flags, or at minimum surfaces the dismissal in an audit log.
- **Cost-aware tuning.** A slider or input for "cost of a false positive vs. missed fraud." Show how the flagged set changes.
- **Tests.** At least one test that exercises your detector on a known fraud case and a known legitimate case.

## Stretch (for teams hitting all of the above by hour 18)

- **Hypothesis log.** A short markdown file documenting every detection rule you tried, what F1 it produced, what you kept or dropped, and why.
- **Receipt / audit trail.** When a reviewer makes a decision, what does the audit log look like? Most teams skip this. The ones that nail it stand out.
- **A novel signal.** Find something the judges didn't think of. Defend it in your demo.

Good luck. We're excited to see what you build.
