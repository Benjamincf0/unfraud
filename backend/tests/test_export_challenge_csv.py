import os

import pandas as pd
import pytest

from algo.algo import DEFAULT_MODEL_PATH
from export_challenge_csv import (
    ROOT,
    _apply_hybrid_decision,
    build_analyzed_transactions,
)
from ml_fraud_scorer import is_model_available


REAL_TRAIN_PATH = ROOT / "backend" / "fraudTrain_part1.csv"


def _classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float | int]:
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def test_tight_hybrid_decision_accepts_ml_signal_or_strong_heuristic_score():
    analyzed = pd.DataFrame(
        {
            "transaction_id": [
                "both",
                "strong_heuristic_only",
                "weak_heuristic_only",
                "model_signal",
                "alert_signal",
            ],
            "is_fraud": [True, False, True, True, True],
            "heuristic_is_fraud": [True, True, True, False, False],
            "heuristic_fraud_score": [0.60, 0.70, 0.30, 0.10, 0.10],
            "flagged_by_model": [True, False, False, True, False],
            "flagged_by_rules": [False, False, False, False, True],
            "model_score": [0.30, 0.10, 0.10, 0.30, 0.20],
            "model_fraud_score": [0.40, 0.10, 0.10, 0.30, 0.35],
        }
    )

    result = _apply_hybrid_decision(analyzed).set_index("transaction_id")

    assert bool(result.loc["both", "is_fraud"]) is True
    assert (
        result.loc["both", "hybrid_decision_reason"]
        == "ml_or_alert_signal_and_strong_heuristic_score"
    )
    assert bool(result.loc["strong_heuristic_only", "is_fraud"]) is True
    assert (
        result.loc["strong_heuristic_only", "hybrid_decision_reason"]
        == "strong_heuristic_score"
    )
    assert bool(result.loc["weak_heuristic_only", "is_fraud"]) is False
    assert bool(result.loc["model_signal", "is_fraud"]) is True
    assert result.loc["model_signal", "hybrid_decision_reason"] == "ml_or_alert_signal"
    assert result.loc["model_signal", "fraud_score"] == 0.55
    assert bool(result.loc["alert_signal", "is_fraud"]) is True
    assert result.loc["alert_signal", "fraud_score"] == 0.55


@pytest.mark.skipif(
    not REAL_TRAIN_PATH.exists() or not is_model_available(DEFAULT_MODEL_PATH),
    reason="fraudTrain_part1.csv and the trained model artifact are required",
)
def test_detector_policy_on_real_labeled_training_slice():
    # Chronological slice keeps the default suite practical while preserving
    # real merchant/category/card distributions and real labels.
    real_data = pd.read_csv(REAL_TRAIN_PATH).head(10_000)

    scored = build_analyzed_transactions(real_data)
    metrics = _classification_metrics(real_data["is_fraud"], scored["is_fraud"])
    fraud_rate = float(real_data["is_fraud"].mean())
    flag_rate = float(scored["is_fraud"].mean())

    assert len(scored) == len(real_data)
    assert int(real_data["is_fraud"].sum()) >= 40
    assert 0.03 <= flag_rate <= 0.10
    assert metrics["recall"] >= 0.90
    assert metrics["precision"] >= fraud_rate * 10
    assert metrics["f1"] >= 0.10
    assert scored.loc[scored["is_fraud"], "fraud_reasons"].fillna("").ne("").all()


@pytest.mark.skipif(
    not REAL_TRAIN_PATH.exists() or not is_model_available(DEFAULT_MODEL_PATH),
    reason="fraudTrain_part1.csv and the trained model artifact are required",
)
@pytest.mark.skipif(
    not bool(os.environ.get("RUN_FULL_REAL_DATA_DETECTOR_TEST")),
    reason="set RUN_FULL_REAL_DATA_DETECTOR_TEST=1 to run the 130k-row detector test",
)
def test_detector_policy_on_full_real_labeled_training_data():
    # This is the true full-dataset detector regression. It is opt-in because
    # it runs the full hybrid export policy across 130k labeled rows.
    real_data = pd.read_csv(REAL_TRAIN_PATH)

    scored = build_analyzed_transactions(real_data)
    metrics = _classification_metrics(real_data["is_fraud"], scored["is_fraud"])

    assert len(scored) == len(real_data)
    assert metrics["recall"] >= 0.90
    assert metrics["precision"] >= 0.15
    assert metrics["f1"] >= 0.25
