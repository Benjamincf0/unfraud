"""ML fraud scorer for production CSVs — loads a trained ``FraudDetectionPipeline`` artifact."""

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
    build_shap_explainer,
    ensure_inference_columns,
    prepare_features,
    shrink,
)

class ModelNotAvailableError(RuntimeError):
    """Raised when ``use_model=True`` but no trained artifact exists."""


_pipeline: Optional[FraudDetectionPipeline] = None


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
    return _pipeline


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


def _parse_rule_codes(raw: Any) -> List[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        if raw.startswith("["):
            return json.loads(raw)
        return [raw] if raw else []
    return [str(raw)]


def _build_score_breakdown(row: pd.Series) -> List[Dict[str, Any]]:
    breakdown: List[Dict[str, Any]] = []
    model_score = float(row.get("model_score", row.get("fraud_score", 0.0)))
    combined_score = float(row.get("fraud_score", model_score))

    if bool(row.get("flagged_by_model")):
        breakdown.append(
            {
                "code": "model_score",
                "label": "Model score",
                "detail": f"LightGBM fraud probability {model_score:.2f}.",
                "weight": round(model_score, 4),
                "signal_type": "model",
                "value": round(model_score, 4),
                "baseline": round(float(row.get("threshold", 0.5)), 4),
            }
        )

    rule_weight = max(0.0, combined_score - model_score)
    for index, rule in enumerate(_parse_rule_codes(row.get("rule_reason_codes"))):
        contribution = rule_weight / max(len(_parse_rule_codes(row.get("rule_reason_codes"))), 1)
        breakdown.append(
            {
                "code": f"rule_{index}",
                "label": "Rule guardrail",
                "detail": rule,
                "weight": round(contribution, 4),
                "signal_type": "rule",
            }
        )

    for index, shap_reason in enumerate(row.get("shap_reason_codes") or []):
        breakdown.append(
            {
                "code": f"shap_{index}",
                "label": "Model feature",
                "detail": str(shap_reason),
                "weight": round(max(0.03, model_score / 4), 4),
                "signal_type": "model",
            }
        )

    if not breakdown and combined_score > 0:
        breakdown.append(
            {
                "code": "composite",
                "label": "Composite anomaly score",
                "detail": str(row.get("alert_reason") or "Elevated hybrid fraud score."),
                "weight": round(combined_score, 4),
                "signal_type": "composite",
                "value": round(combined_score, 4),
                "baseline": 0.0,
            }
        )

    return sorted(breakdown, key=lambda item: item.get("weight", 0), reverse=True)


def _build_card_amount_series(scored: pd.DataFrame, points: int = 12) -> List[str]:
    series_column: List[str] = []
    card_counters: Dict[str, int] = {}
    card_series_map: Dict[str, List[List[Dict[str, Any]]]] = {}

    for card_id, card_df in scored.groupby("card_id", sort=False):
        card_df = card_df.sort_values("timestamp")
        running: List[Dict[str, Any]] = []
        snapshots: List[List[Dict[str, Any]]] = []
        for _, row in card_df.iterrows():
            ts = row["timestamp"]
            timestamp = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            running.append(
                {
                    "timestamp": timestamp,
                    "amount": float(row["amount"]),
                    "risk_score": round(float(row["fraud_score"]), 4),
                }
            )
            snapshots.append(running[-points:].copy())
        card_series_map[str(card_id)] = snapshots

    for _, row in scored.iterrows():
        card_id = str(row["card_id"])
        position = card_counters.get(card_id, 0)
        series_column.append(json.dumps(card_series_map[card_id][position]))
        card_counters[card_id] = position + 1
    return series_column


def _adapt_scored_frame(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = scored.copy()
    out["threshold"] = threshold
    breakdowns: List[str] = []
    baselines: List[str] = []
    cross_card: List[str] = []
    graph_features: List[str] = []
    reason_labels: List[str] = []

    for _, row in out.iterrows():
        breakdown = _build_score_breakdown(row)
        breakdowns.append(json.dumps(breakdown))
        reason_labels.append(
            "; ".join(item["label"] for item in breakdown if item.get("label"))
            or str(row.get("alert_reason") or "")
        )
        baselines.append(
            json.dumps(
                {
                    "history_count": int(row.get("tx_of_day", 0) or 0),
                    "typical_amount": round(float(row.get("card_amt_mean_so_far", row["amount"])), 2),
                    "amount_ratio": round(
                        float(row.get("amount", 0))
                        / max(float(row.get("card_amt_mean_so_far", row["amount"]) or 0), 1e-9),
                        4,
                    ),
                    "amount_zscore": round(float(row.get("amt_z_vs_card", 0.0)), 4),
                    "category_typical_amount": round(float(row.get("card_amt_mean_so_far", row["amount"])), 2),
                    "amount_ratio_category": round(
                        float(row.get("amount", 0))
                        / max(float(row.get("card_amt_mean_so_far", row["amount"]) or 0), 1e-9),
                        4,
                    ),
                    "amount_zscore_category": round(float(row.get("amt_z_vs_category", 0.0)), 4),
                    "usual_categories": [],
                    "usual_countries": [],
                    "usual_devices": [],
                    "usual_ips": [],
                }
            )
        )
        cross_card.append(
            json.dumps(
                {
                    "device_card_fanout": int(row.get("device_card_fanout_24h", 1) or 1),
                    "ip_card_fanout": int(row.get("ip_card_fanout_24h", 1) or 1),
                    "merchant_tx_30m": int(row.get("tx_5min", 0) or 0),
                    "merchant_unique_cards_2h": int(row.get("distinct_categories_24h", 0) or 0),
                }
            )
        )
        graph_features.append(
            json.dumps(
                {
                    "amount_ratio": round(
                        float(row.get("amount", 0))
                        / max(float(row.get("card_amt_mean_so_far", row["amount"]) or 0), 1e-9),
                        4,
                    ),
                    "amount_zscore": round(float(row.get("amt_z_vs_card", 0.0)), 4),
                    "amount_ratio_category": round(
                        float(row.get("amount", 0))
                        / max(float(row.get("card_amt_mean_so_far", row["amount"]) or 0), 1e-9),
                        4,
                    ),
                    "amount_zscore_category": round(float(row.get("amt_z_vs_category", 0.0)), 4),
                    "card_tx_index": float(row.get("tx_of_day", 0) or 0),
                    "device_card_fanout": float(row.get("device_card_fanout_24h", 1) or 1),
                    "ip_card_fanout": float(row.get("ip_card_fanout_24h", 1) or 1),
                    "merchant_tx_30m": float(row.get("tx_5min", 0) or 0),
                    "merchant_unique_cards_2h": float(row.get("distinct_categories_24h", 0) or 0),
                }
            )
        )

    out["fraud_reasons"] = reason_labels
    out["score_breakdown"] = breakdowns
    out["card_baseline_json"] = baselines
    out["cross_card_signals_json"] = cross_card
    out["graph_features_json"] = graph_features
    out["card_amount_series_json"] = _build_card_amount_series(out)
    return out


def ml_fraud_detection(
    df: pd.DataFrame,
    *,
    model_path: Path | str | None = None,
) -> pd.DataFrame:
    """Score unlabeled transactions with the trained hybrid ML pipeline."""
    working = _prepare_upload_frame(df)
    if working.empty:
        return _empty_result(df)

    pipeline = get_pipeline(model_path or DEFAULT_MODEL_PATH)
    if pipeline.explainer is None and pipeline.model is not None:
        sample = working.head(min(len(working), 200))
        g_sample = apply_rule_guardrails(build_features(shrink(sample.copy())))
        pipeline.explainer = build_shap_explainer(pipeline.model, prepare_features(g_sample))

    featured = apply_rule_guardrails(build_features(shrink(working.copy())))
    scored = pipeline.predict(featured, prepare_features(featured))
    adapted = _adapt_scored_frame(scored, pipeline.threshold)
    score_columns = [
        "is_fraud",
        "fraud_score",
        "fraud_reasons",
        "score_breakdown",
        "card_baseline_json",
        "cross_card_signals_json",
        "graph_features_json",
        "card_amount_series_json",
        "model_score",
        "flagged_by_model",
        "flagged_by_rules",
        "alert_reason",
    ]
    keyed = adapted.set_index("transaction_id")
    result = df.copy()
    for column in score_columns:
        if column in keyed.columns:
            result[column] = result["transaction_id"].map(keyed[column])
    return result
