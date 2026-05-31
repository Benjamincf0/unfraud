from pathlib import Path

import pandas as pd

from algo.algo import DEFAULT_MODEL_PATH
from fraud_scorer import simple_fraud_detection
from ml_fraud_scorer import get_pipeline, ml_fraud_detection


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "transactions.csv"
OUTPUT_PATH = ROOT / "analyzed_transactions.csv"
STRONG_HEURISTIC_SCORE_THRESHOLD = 0.55
ML_SIGNAL_SCORE_FLOOR = 0.55


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def _apply_hybrid_decision(analyzed: pd.DataFrame) -> pd.DataFrame:
    """Use ML/strict-alert signals plus strong heuristic scores for the export flag."""
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


def build_analyzed_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    heuristic = simple_fraud_detection(transactions)
    model = ml_fraud_detection(transactions, model_path=DEFAULT_MODEL_PATH)
    pipeline = get_pipeline(DEFAULT_MODEL_PATH)

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
    return _apply_hybrid_decision(analyzed)


def main() -> None:
    transactions = pd.read_csv(INPUT_PATH)
    analyzed = build_analyzed_transactions(transactions)
    analyzed["review_decision"] = ""
    analyzed["reviewer_notes"] = ""
    analyzed["reviewed_at"] = ""
    analyzed.to_csv(OUTPUT_PATH, index=False)
    flagged_count = int(analyzed["is_fraud"].sum())
    print(
        f"Wrote {OUTPUT_PATH.relative_to(ROOT)} with {len(analyzed)} rows "
        f"({flagged_count} flagged)"
    )


if __name__ == "__main__":
    main()
