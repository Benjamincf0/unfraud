"""ML fraud scorer: model decides flags/scores, SHAP explains them in reviewer language."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from algo.algo import (
    DEFAULT_MODEL_PATH,
    FraudDetectionPipeline,
    apply_rule_guardrails,
    build_features,
    ensure_inference_columns,
    prepare_features,
    shap_score_breakdown_for_rows,
    shrink,
)
from fraud_scorer import simple_fraud_detection

class ModelNotAvailableError(RuntimeError):
    """Raised when ``use_model=True`` but no trained artifact exists."""


_pipeline: Optional[FraudDetectionPipeline] = None

ML_DECISION_COLUMNS = (
    "is_fraud",
    "fraud_score",
    "model_score",
    "flagged_by_model",
    "flagged_by_rules",
)

EXPLAIN_COLUMNS = (
    "fraud_reasons",
    "score_breakdown",
    "card_baseline_json",
    "cross_card_signals_json",
    "graph_features_json",
    "card_amount_series_json",
)


def is_model_available(path: Path | str = DEFAULT_MODEL_PATH) -> bool:
    return Path(path).exists()


def get_pipeline(path: Path | str = DEFAULT_MODEL_PATH) -> FraudDetectionPipeline:
    global _pipeline
    target = Path(path)
    if _pipeline is None or getattr(_pipeline, "_artifact_path", None) != str(target.resolve()):
        if not target.exists():
            raise ModelNotAvailableError(
                f"ML model artifact not found at {target}. "
                "Train with: uv run python -m scripts.train_fraud_model"
            )
        _pipeline = FraudDetectionPipeline.load(target)
        _pipeline._artifact_path = str(target.resolve())
        _pipeline.ensure_explainer()
    return _pipeline


def _empty_result(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    for column, default_value in [
        ("is_fraud", False),
        ("fraud_score", 0.0),
        ("fraud_reasons", ""),
        ("score_breakdown", "[]"),
        ("card_baseline_json", "{}"),
        ("cross_card_signals_json", "{}"),
        ("graph_features_json", "{}"),
        ("card_amount_series_json", "[]"),
    ]:
        working[column] = default_value
    return working


def _prepare_upload_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = ensure_inference_columns(df.copy())
    if working.empty:
        return working

    if "device_id" not in working.columns:
        working["device_id"] = pd.NA
    if "ip_address" not in working.columns:
        working["ip_address"] = pd.NA

    working["device_id"] = working["device_id"].replace("", pd.NA)
    working["ip_address"] = working["ip_address"].replace("", pd.NA)
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    if working["timestamp"].isna().any():
        raise ValueError("Invalid timestamp format in CSV")

    return working.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


def _model_decisions(
    pipeline: FraudDetectionPipeline,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = _prepare_upload_frame(df)
    featured = apply_rule_guardrails(build_features(shrink(working.copy())))
    model_prob, combined = pipeline.hybrid_scores(featured, prepare_features(featured))
    threshold = pipeline.threshold
    rule_alert = (
        featured["rule_alert"].values
        if "rule_alert" in featured.columns
        else featured["rule_guardrail"].values
    )
    flagged = (model_prob >= threshold) | rule_alert

    decisions = pd.DataFrame(
        {
            "transaction_id": featured["transaction_id"].astype(str).values,
            "model_score": np.round(model_prob, 4),
            "fraud_score": np.round(combined, 4),
            "is_fraud": flagged,
            "flagged_by_model": (model_prob >= threshold),
            "flagged_by_rules": rule_alert,
        }
    )
    return decisions, featured


def _apply_shap_explanations(
    result: pd.DataFrame,
    pipeline: FraudDetectionPipeline,
    featured: pd.DataFrame,
    heuristic: pd.DataFrame,
) -> pd.DataFrame:
    """Replace explanations on flagged rows with SHAP; heuristic fallback if SHAP empty."""
    out = result.copy()
    pipeline.ensure_explainer()
    if pipeline.explainer is None:
        return out

    heuristic_by_tx = heuristic.set_index(heuristic["transaction_id"].astype(str))
    featured = featured.copy()
    featured["transaction_id"] = featured["transaction_id"].astype(str)

    flagged_tx = out.loc[out["is_fraud"], "transaction_id"].astype(str)
    if flagged_tx.empty:
        out["score_breakdown"] = "[]"
        out["fraud_reasons"] = ""
        return out

    flagged_order = pd.DataFrame(
        {"transaction_id": flagged_tx.tolist(), "_ord": range(len(flagged_tx))}
    )
    flagged_featured = (
        featured.merge(flagged_order, on="transaction_id", how="inner")
        .sort_values("_ord")
        .drop(columns="_ord")
    )

    breakdowns = shap_score_breakdown_for_rows(pipeline.explainer, flagged_featured)
    breakdown_by_tx = {
        str(tx_id): breakdown
        for tx_id, breakdown in zip(flagged_featured["transaction_id"], breakdowns)
    }

    score_breakdown_col: List[str] = []
    fraud_reasons_col: List[str] = []

    for _, row in out.iterrows():
        tx_id = str(row["transaction_id"])
        if not bool(row["is_fraud"]):
            score_breakdown_col.append("[]")
            fraud_reasons_col.append("")
            continue

        breakdown = breakdown_by_tx.get(tx_id, [])
        if not breakdown and tx_id in heuristic_by_tx.index:
            breakdown = json.loads(
                heuristic_by_tx.loc[tx_id].get("score_breakdown") or "[]"
            )

        score_breakdown_col.append(json.dumps(breakdown))
        fraud_reasons_col.append(
            "; ".join(item["label"] for item in breakdown if item.get("label"))
        )

    out["score_breakdown"] = score_breakdown_col
    out["fraud_reasons"] = fraud_reasons_col
    return out


def _rebuild_card_amount_series(df: pd.DataFrame, points: int = 12) -> pd.DataFrame:
    """Rebuild amount history charts using the active fraud_score column."""
    out = df.copy()
    if out.empty:
        return out

    working = out.copy()
    working["timestamp_dt"] = pd.to_datetime(working["timestamp"], errors="coerce")
    series_map: Dict[str, List[List[Dict[str, Any]]]] = {}

    for card_id, card_df in working.groupby("card_id", sort=False):
        card_df = card_df.sort_values("timestamp_dt")
        running: List[Dict[str, Any]] = []
        snapshots: List[List[Dict[str, Any]]] = []
        for _, row in card_df.iterrows():
            running.append(
                {
                    "timestamp": row["timestamp_dt"].isoformat(),
                    "amount": float(row["amount"]),
                    "risk_score": round(float(row["fraud_score"]), 4),
                }
            )
            snapshots.append(running[-points:].copy())
        series_map[str(card_id)] = snapshots

    card_counters: Dict[str, int] = {}
    series_column: List[str] = []
    for _, row in working.iterrows():
        card_id = str(row["card_id"])
        position = card_counters.get(card_id, 0)
        series_column.append(json.dumps(series_map[card_id][position]))
        card_counters[card_id] = position + 1

    out["card_amount_series_json"] = series_column
    return out


def _apply_model_fallback_reasons(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """When the model flags a row with no breakdown, add plain language."""
    out = df.copy()
    breakdowns: List[str] = []
    reason_labels: List[str] = []

    for _, row in out.iterrows():
        breakdown = json.loads(row.get("score_breakdown") or "[]")
        if bool(row["is_fraud"]) and not breakdown:
            score = float(row["fraud_score"])
            breakdown = [
                {
                    "code": "model_risk",
                    "label": "Elevated model risk",
                    "detail": (
                        f"The fraud model scored this transaction {score:.0%}, "
                        f"above the review threshold ({threshold:.0%})."
                    ),
                    "weight": round(score, 4),
                    "signal_type": "model",
                    "value": round(score, 4),
                    "baseline": round(threshold, 4),
                }
            ]

        breakdowns.append(json.dumps(breakdown))
        existing = str(row.get("fraud_reasons") or "").strip()
        if existing:
            reason_labels.append(existing)
        else:
            reason_labels.append(
                "; ".join(item["label"] for item in breakdown if item.get("label"))
            )

    out["score_breakdown"] = breakdowns
    out["fraud_reasons"] = reason_labels
    return out


def ml_fraud_detection(
    df: pd.DataFrame,
    *,
    model_path: Path | str | None = None,
) -> pd.DataFrame:
    """Flag with the trained model; explain flagged rows with readable SHAP breakdowns."""
    if df.empty:
        return _empty_result(df)

    pipeline = get_pipeline(model_path or DEFAULT_MODEL_PATH)
    decisions, featured = _model_decisions(pipeline, df)
    heuristic = simple_fraud_detection(df)

    result = heuristic.copy()
    keyed = decisions.set_index("transaction_id")
    for column in ML_DECISION_COLUMNS:
        result[column] = result["transaction_id"].astype(str).map(keyed[column])

    result = _apply_shap_explanations(result, pipeline, featured, heuristic)
    result = _rebuild_card_amount_series(result)
    return _apply_model_fallback_reasons(result, pipeline.threshold)
