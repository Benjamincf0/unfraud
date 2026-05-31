# Heuristic (rules-based) scoring

This is the **default** engine used by the live API and by `make export`. It lives in `fraud_scorer.py` and does **not** need labeled fraud data or a `.pkl` model file.

## Idea in plain language

For each transaction, the system asks:

1. **Compared to this card’s past**, is the amount, category, device, IP, or country unusual?
2. **Compared to all cards in the file**, is this device, IP, or merchant involved in suspicious sharing or bursts?

Those answers become partial “risk scores” that are added together, capped at 1.0. If the total crosses a threshold—or a rare **high-confidence rule** fires—the transaction is flagged.

## Processing order (why time matters)

Rows are sorted by **timestamp**, then scored in order. For every metric “before this transaction,” only **earlier rows on the same card** count as history. That mimics real life: at the moment of payment, you only know the past.

If timestamps are invalid, scoring fails with a clear error.

## Per-card signals

### Amount vs card history

- **Median and mean** of past amounts on this card (expanding window).
- **Amount ratio** — current amount ÷ typical (median) amount.
- **Z-score** — how many standard deviations above/below the card’s past mean.

Large ratios or z-scores increase risk (e.g. a $2,000 charge when the card usually spends $40).

### Amount vs category on this card

Same idea, but scoped to **this merchant category** on this card (grocery vs electronics vs travel). A large travel charge may be normal globally but abnormal for a card that only buys groceries.

### Novelty flags

After a few prior transactions on the card, the system flags:

| Signal | Meaning |
|--------|---------|
| **Novel category** | First time this category appears on the card |
| **Novel device** | Online payment from a device never seen on this card |
| **Novel IP** | Online payment from an IP never seen on this card |
| **Novel country** | First time merchant country appears for this card |

Novelty alone is not always fraud; it combines with amount and other signals.

### Geography

- **Foreign country** — cardholder country ≠ merchant country.
- Cross-border risk is **damped** unless there is also an amount or identity anomaly (avoids flagging every international purchase).
- **Novel country** adds stronger geo risk.

## Cross-card signals

These detect patterns that involve **many cards**:

| Signal | What it measures |
|--------|------------------|
| **Device fanout** | How many distinct cards used the same `device_id` in the file |
| **IP fanout** | How many distinct cards used the same `ip_address` |
| **Merchant burst (30 min)** | How many transactions at this merchant in a rolling 30-minute window |
| **Merchant unique cards (2 h)** | How many different cards hit this merchant in 2 hours |

High fanout or bursts suggest stolen credentials, merchant compromise, or bot activity.

## Reviewer feedback

Reviewer escalation feeds back into the session scorer. When a reviewer escalates a transaction with an `ip_address`, every row sharing that IP receives an additional review-feedback signal:

- **Label:** “Previously escalated IP”
- **Score impact:** +0.22, capped at 1.0
- **Flag behavior:** the matching rows are treated as fraud candidates immediately
- **Undo behavior:** returning the escalated transaction to pending removes the feedback boost

This is intentionally session-local. It helps the reviewer chase an active suspicious IP without permanently changing the base heuristic model.

## How the final score is built

Core heuristic **components** contribute weighted risk (each capped, then summed and clipped to [0, 1]):

| Component | What it captures | Approx. weight in blend |
|-----------|------------------|-------------------------|
| Amount outlier | Large vs card median / z-score | ~26% |
| Category amount | Large vs category norm on card | ~26% |
| Category shift | Unfamiliar category for card | ~7% |
| Device shift | New device (online) | ~10% |
| IP shift | New IP (online) | ~9% |
| Geo shift | Country anomalies | ~7% |
| Device reuse (cross-card) | Shared device | ~11% |
| IP reuse (cross-card) | Shared IP | ~9% |
| Merchant burst (cross-card) | Velocity across cards | ~14% |

Exact weights are defined in code; the table reflects the intended emphasis.

## When is a transaction flagged?

Two paths:

1. **Score threshold** — `fraud_score >= 0.55` (chosen for ~1% fraud prevalence so the queue stays reviewable).
2. **High-confidence rules** (flag even if score is lower):
   - Amount ≥ 7× card median, or
   - Strong category amount spike with history, or
   - IP used on ≥ 4 cards, or
   - Merchant seen on ≥ 7 unique cards in 2 hours.

## Explainability

Every flagged row gets a **score breakdown**: ordered list of reasons with:

- **label** — short title shown in UI (“Amount anomaly”, “IP shared across cards”)
- **detail** — sentence with numbers (“Amount is 3.2× this card’s historical median (42.00)”)
- **weight** — how much this reason contributed
- **signal_type** — `per_card` vs `cross_card` vs `composite`

Additional JSON bundles:

- **card_baseline** — history count, typical amounts, usual categories/countries/devices/IPs
- **cross_card_signals** — fanout and burst counts
- **card_amount_series** — recent points for sparkline charts

## Strengths and limitations

**Strengths**

- Transparent: every flag ties to explicit rules analysts can debate.
- Works on **unlabeled** challenge data.
- Conservative threshold reduces alert fatigue.

**Limitations**

- Weights are hand-tuned, not learned from your bank’s fraud labels.
- Cross-card metrics in this scorer use **whole-file** snapshots for some fanout (simpler than 24h rolling windows used in ML features).
- Cannot adapt automatically when fraud patterns drift (unless you change code or switch to ML + retrain).

For the ML alternative, see [05-machine-learning-model.md](05-machine-learning-model.md).
