"""Score a transaction CSV with the trained ML model and print detection stats.

Default input is the challenge file at the repository root (``transactions.csv``).

Examples::

    cd backend
    uv run python -m scripts.score_transactions

    uv run python -m scripts.score_transactions ../transactions.csv --output ../ml_analyzed_transactions.csv

    uv run python -m scripts.score_transactions data.csv --list-flagged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_BACKEND = Path(__file__).resolve().parent.parent
_ROOT = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from algo.algo import DEFAULT_MODEL_PATH
from ml_fraud_scorer import ModelNotAvailableError, get_pipeline, is_model_available, ml_fraud_detection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hybrid ML fraud detector on a transaction CSV and summarize flags.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(_ROOT / "transactions.csv"),
        help="Transaction CSV to score (default: repo root transactions.csv)",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to fraud_model.pkl (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="Write enriched CSV (same columns as API export) to PATH",
    )
    parser.add_argument(
        "--list-flagged",
        action="store_true",
        help="Print transaction_id for each flagged row",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="After the summary, print the top N rows by fraud_score (0 = skip)",
    )
    return parser.parse_args()


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "true_fraud": int(yt.sum()),
        "true_legit": int((yt == 0).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _print_summary(
    scored: pd.DataFrame,
    *,
    threshold: float,
    input_path: Path,
    ground_truth: pd.Series | None,
) -> None:
    total = len(scored)
    flagged = scored["is_fraud"].astype(bool)
    n_flagged = int(flagged.sum())
    pct = (100.0 * n_flagged / total) if total else 0.0

    model_only = int((scored["flagged_by_model"] & ~scored["flagged_by_rules"]).sum())
    rules_only = int((scored["flagged_by_rules"] & ~scored["flagged_by_model"]).sum())
    both = int((scored["flagged_by_model"] & scored["flagged_by_rules"]).sum())
    model_hits = int(scored["flagged_by_model"].sum())
    rule_hits = int(scored["flagged_by_rules"].sum())

    print(f"Input:     {input_path}")
    print(f"Rows:      {total}")
    print(f"Threshold: {threshold:.4f} (from model artifact)")
    print()
    print(f"Flagged (fraud alerts): {n_flagged} ({pct:.1f}% of rows)")
    print(f"  — model probability ≥ threshold: {model_hits}")
    print(f"  — any guardrail rule fired:      {rule_hits}")
    print(f"  — flagged by model only:         {model_only}")
    print(f"  — flagged by rules only:         {rules_only}")
    print(f"  — flagged by both:               {both}")

    if ground_truth is not None:
        metrics = _classification_metrics(ground_truth, scored["is_fraud"])
        print()
        print("Ground truth (is_fraud column in input CSV):")
        print(f"  Known fraud rows: {metrics['true_fraud']}")
        print(f"  Known legit rows: {metrics['true_legit']}")
        print(
            f"  Precision: {metrics['precision']:.4f}  "
            f"Recall: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}"
        )
        print(
            f"  Confusion — TP: {metrics['tp']}  FP: {metrics['fp']}  "
            f"FN: {metrics['fn']}  TN: {metrics['tn']}"
        )
    else:
        print()
        print(
            "Note: transactions.csv has no fraud labels. "
            "Counts above are detector flags, not confirmed fraud."
        )


def main() -> None:
    args = _parse_args()
    input_path = Path(args.csv_path).resolve()
    model_path = Path(args.model).resolve()

    if not input_path.exists():
        print(f"Error: CSV not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not is_model_available(model_path):
        print(
            f"Error: ML model not found at {model_path}\n"
            "Train first:  cd backend && uv run python -m scripts.train_fraud_model",
            file=sys.stderr,
        )
        sys.exit(1)

    transactions = pd.read_csv(input_path)
    ground_truth: pd.Series | None = None
    if "is_fraud" in transactions.columns:
        ground_truth = transactions["is_fraud"].copy()

    try:
        pipeline = get_pipeline(model_path)
        scored = ml_fraud_detection(transactions, model_path=model_path)
    except ModelNotAvailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_summary(
        scored,
        threshold=float(pipeline.threshold),
        input_path=input_path,
        ground_truth=ground_truth,
    )

    if args.list_flagged:
        flagged_ids = scored.loc[scored["is_fraud"], "transaction_id"].astype(str).tolist()
        print()
        print(f"Flagged transaction_ids ({len(flagged_ids)}):")
        for tx_id in flagged_ids:
            print(f"  {tx_id}")

    if args.top > 0:
        print()
        print(f"Top {args.top} by fraud_score:")
        top = scored.nlargest(args.top, "fraud_score")
        for _, row in top.iterrows():
            flag = "FLAG" if row["is_fraud"] else "    "
            print(
                f"  [{flag}] {row['transaction_id']}  "
                f"score={row['fraud_score']:.4f}  model={row['model_score']:.4f}  "
                f"{row.get('fraud_reasons', '')[:80]}"
            )

    if args.output:
        out_path = Path(args.output).resolve()
        export = scored.copy()
        for col in ("review_decision", "reviewer_notes", "reviewed_at"):
            if col not in export.columns:
                export[col] = ""
        export.to_csv(out_path, index=False)
        print()
        print(f"Wrote {out_path} ({len(export)} rows)")


if __name__ == "__main__":
    main()
