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
from fraud_scorer import simple_fraud_detection
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


def _pct(n: int, total: int) -> str:
    if not total:
        return "0.0%"
    return f"{100.0 * n / total:.1f}%"


def _print_summary(
    scored: pd.DataFrame,
    *,
    threshold: float,
    input_path: Path,
    ground_truth: pd.Series | None,
    heuristic_flags: int | None = None,
) -> None:
    total = len(scored)
    hybrid = scored["is_fraud"].astype(bool)
    model_only_mask = scored["flagged_by_model"] & ~scored["flagged_by_rules"]
    n_hybrid = int(hybrid.sum())
    n_model_only = int(model_only_mask.sum())
    n_model_hits = int(scored["flagged_by_model"].sum())
    n_alert_hits = int(scored["flagged_by_rules"].sum())
    alerts_only = int((scored["flagged_by_rules"] & ~scored["flagged_by_model"]).sum())
    both = int((scored["flagged_by_model"] & scored["flagged_by_rules"]).sum())

    print(f"Input:     {input_path}")
    print(f"Rows:      {total}")
    print(f"Threshold: {threshold:.4f} (from model artifact)")
    print()
    print("Review queue size (not the same as ~7% ground-truth fraud rate):")
    print(
        f"  Hybrid (model OR rules — used by API / export): "
        f"{n_hybrid} ({_pct(n_hybrid, total)})"
    )
    print(
        f"  Model-only (probability ≥ threshold, no rule): "
        f"{n_model_only} ({_pct(n_model_only, total)})"
    )
    if heuristic_flags is not None:
        print(
            f"  Heuristic scorer (reference):                  "
            f"{heuristic_flags} ({_pct(heuristic_flags, total)})"
        )
    print()
    print("Hybrid breakdown:")
    print(f"  — model probability ≥ threshold: {n_model_hits}")
    print(f"  — high-confidence rule alert:    {n_alert_hits}")
    print(f"  — flagged by model only:         {n_model_only}")
    print(f"  — flagged by alert only:         {alerts_only}")
    print(f"  — flagged by both:               {both}")

    if ground_truth is not None:
        for label, pred in (
            ("Hybrid", scored["is_fraud"]),
            ("Model-only", model_only_mask),
        ):
            metrics = _classification_metrics(ground_truth, pred)
            print()
            print(f"Ground truth vs {label}:")
            print(f"  Known fraud rows: {metrics['true_fraud']}")
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
            "Note: challenge CSV has no labels. ~7% is hidden true fraud; "
            "flag counts are review-queue size, not fraud prevalence."
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

    heuristic_flags = int(simple_fraud_detection(transactions)["is_fraud"].sum())

    _print_summary(
        scored,
        threshold=float(pipeline.threshold),
        input_path=input_path,
        ground_truth=ground_truth,
        heuristic_flags=heuristic_flags,
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
