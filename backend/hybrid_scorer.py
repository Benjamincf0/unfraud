"""Hybrid ML + strong heuristic scoring for the canonical review queue."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from algo.algo import DEFAULT_MODEL_PATH
from fraud_scorer import simple_fraud_detection
from ml_fraud_scorer import ModelNotAvailableError, get_pipeline, ml_fraud_detection

STRONG_HEURISTIC_SCORE_THRESHOLD = 0.55
ML_SIGNAL_SCORE_FLOOR = 0.55


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def apply_hybrid_decision(analyzed: pd.DataFrame) -> pd.DataFrame:
    """Flag rows on ML alert signals or a strong heuristic score (>= 0.55)."""
    out = analyzed.copy()
    model_signal = _bool_series(out, "flagged_by_model") | _bool_series(
        out, "flagged_by_rules"
    )
    heuristic_score = pd.to_numeric(
        out.get("heuristic_fraud_score", 0.0), errors="coerce"
    ).fillna(0.0)
    strong_heuristic = heuristic_score >= STRONG_HEURISTIC_SCORE_THRESHOLD
    score_columns = [
        column
        for column in ("model_fraud_score", "model_score", "heuristic_fraud_score")
        if column in out.columns
    ]
    if score_columns:
        final_score = out[score_columns].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0).max(axis=1)
    else:
        final_score = pd.Series(0.0, index=out.index)
    final_score = final_score.where(
        ~model_signal,
        final_score.clip(lower=ML_SIGNAL_SCORE_FLOOR),
    )

    final_flag = model_signal | strong_heuristic
    out["fraud_score"] = final_score.round(4)
    out["is_fraud"] = final_flag
    out["hybrid_decision_reason"] = "not_flagged"
    out.loc[model_signal & ~strong_heuristic, "hybrid_decision_reason"] = (
        "ml_or_alert_signal"
    )
    out.loc[~model_signal & strong_heuristic, "hybrid_decision_reason"] = (
        "strong_heuristic_score"
    )
    out.loc[model_signal & strong_heuristic, "hybrid_decision_reason"] = (
        "ml_or_alert_signal_and_strong_heuristic_score"
    )

    reasons = (
        out.get("fraud_reasons", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
    )
    heuristic_reasons = (
        out.get("heuristic_fraud_reasons", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
    )
    missing_reason = final_flag & reasons.str.strip().eq("")
    out.loc[missing_reason & heuristic_reasons.str.strip().ne(""), "fraud_reasons"] = (
        heuristic_reasons
    )
    still_missing = (
        final_flag
        & out["fraud_reasons"].fillna("").astype(str).str.strip().eq("")
    )
    out.loc[
        still_missing & _bool_series(out, "flagged_by_model"),
        "fraud_reasons",
    ] = "ML model risk above threshold"
    out.loc[
        still_missing & _bool_series(out, "flagged_by_rules"),
        "fraud_reasons",
    ] = "Strict ML alert guardrail"
    out.loc[
        still_missing & strong_heuristic,
        "fraud_reasons",
    ] = "Strong heuristic fraud score"
    return out


def build_hybrid_scored_transactions(
    transactions: pd.DataFrame,
    *,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """Score with ML, attach heuristic columns, and apply the hybrid queue rule."""
    path = DEFAULT_MODEL_PATH if model_path is None else model_path
    heuristic = simple_fraud_detection(transactions)
    model = ml_fraud_detection(transactions, model_path=path)
    pipeline = get_pipeline(path)

    heuristic_by_tx = heuristic.set_index(heuristic["transaction_id"].astype(str))
    analyzed = model.copy()
    analyzed["model_fraud_score"] = analyzed["fraud_score"]
    tx_ids = analyzed["transaction_id"].astype(str)
    analyzed["heuristic_is_fraud"] = tx_ids.map(heuristic_by_tx["is_fraud"]).fillna(
        False
    )
    analyzed["heuristic_fraud_score"] = tx_ids.map(
        heuristic_by_tx["fraud_score"]
    ).fillna(0.0)
    analyzed["heuristic_fraud_reasons"] = tx_ids.map(
        heuristic_by_tx["fraud_reasons"]
    ).fillna("")
    analyzed["flagged_by_alert"] = _bool_series(analyzed, "flagged_by_rules")
    analyzed["model_threshold"] = round(float(pipeline.threshold), 4)
    analyzed["strong_heuristic_score_threshold"] = STRONG_HEURISTIC_SCORE_THRESHOLD
    analyzed["ml_signal_score_floor"] = ML_SIGNAL_SCORE_FLOOR
    return apply_hybrid_decision(analyzed)


def hybrid_summary_stats(hybrid_df: pd.DataFrame) -> dict[str, int]:
    """Break down hybrid queue rows by ML cause and heuristic-only boosts."""
    if hybrid_df.empty or "is_fraud" not in hybrid_df.columns:
        return {
            "model_only_count": 0,
            "alert_only_count": 0,
            "model_alert_both_count": 0,
            "soft_rule_only_count": 0,
            "heuristic_boost_count": 0,
        }

    flagged = hybrid_df["is_fraud"].astype(bool)
    by_model = hybrid_df.get(
        "flagged_by_model", pd.Series(False, index=hybrid_df.index)
    ).astype(bool)
    by_alert = hybrid_df.get(
        "flagged_by_rules", pd.Series(False, index=hybrid_df.index)
    ).astype(bool)
    soft = hybrid_df.get(
        "rule_guardrail", pd.Series(False, index=hybrid_df.index)
    ).astype(bool)
    return {
        "model_only_count": int((by_model & ~by_alert).sum()),
        "alert_only_count": int((by_alert & ~by_model).sum()),
        "model_alert_both_count": int((by_model & by_alert).sum()),
        "soft_rule_only_count": int((soft & ~flagged).sum()),
        "heuristic_boost_count": int((flagged & ~by_model & ~by_alert).sum()),
    }
