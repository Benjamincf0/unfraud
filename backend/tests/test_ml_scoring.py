import json
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from algo.algo import FraudDetectionPipeline
from fraud_scorer import simple_fraud_detection
from ml_fraud_scorer import ml_fraud_detection
from tests.test_algo import _sample_labeled_df


def _review_csv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["tx_001", "tx_002", "tx_003", "tx_004"],
            "timestamp": [
                "2026-04-25T00:00:00",
                "2026-04-25T00:01:00",
                "2026-04-25T00:02:00",
                "2026-04-25T00:03:00",
            ],
            "card_id": ["card_001", "card_001", "card_001", "card_001"],
            "amount": [12.0, 11.0, 14.0, 120.0],
            "merchant_name": ["Store A", "Store A", "Store A", "Store A"],
            "merchant_category": ["grocery", "grocery", "grocery", "electronics"],
            "channel": ["online", "online", "online", "online"],
            "cardholder_country": ["US", "US", "US", "US"],
            "merchant_country": ["US", "US", "US", "GB"],
            "device_id": ["dev_1", "dev_1", "dev_1", "dev_1"],
            "ip_address": ["10.0.0.1", "10.0.0.1", "10.0.0.1", "10.0.0.1"],
        }
    )


def test_ml_scoring_uses_shap_explanations_for_flagged(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "train.csv")
        model_path = os.path.join(tmp, "fraud_model.pkl")
        _sample_labeled_df().to_csv(csv_path, index=False)
        pipeline = FraudDetectionPipeline(model_threshold=0.3)
        pipeline.fit(csv_path, train_frac=0.7)
        pipeline.save(model_path)

        import ml_fraud_scorer as scorer_module

        monkeypatch.setattr(scorer_module, "DEFAULT_MODEL_PATH", model_path)
        monkeypatch.setattr(scorer_module, "_pipeline", None)

        df = _review_csv()
        heuristic = simple_fraud_detection(df)
        scored = ml_fraud_detection(df, model_path=model_path)

        assert scored["card_baseline_json"].tolist() == heuristic["card_baseline_json"].tolist()
        assert scored["cross_card_signals_json"].tolist() == heuristic["cross_card_signals_json"].tolist()

        outlier = scored[scored["transaction_id"] == "tx_004"].iloc[0]
        assert bool(outlier["is_fraud"])
        breakdown = json.loads(outlier["score_breakdown"])
        assert breakdown
        assert all("weight" in item and "detail" in item for item in breakdown)
        assert "sigma" not in json.dumps(breakdown).lower()
        assert "σ" not in json.dumps(breakdown)
        labels = {item.get("label", "") for item in breakdown}
        assert labels
        assert breakdown != json.loads(
            heuristic.loc[heuristic["transaction_id"] == "tx_004", "score_breakdown"].iloc[0]
        ) or any(item.get("code") in {"amt_z_vs_card", "amt_z_vs_category"} for item in breakdown)

        benign = scored[scored["transaction_id"] == "tx_001"].iloc[0]
        assert json.loads(benign["score_breakdown"]) == []


def test_ml_scoring_decisions_come_from_model(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "train.csv")
        model_path = os.path.join(tmp, "fraud_model.pkl")
        _sample_labeled_df().to_csv(csv_path, index=False)
        pipeline = FraudDetectionPipeline(model_threshold=0.3)
        pipeline.fit(csv_path, train_frac=0.7)
        pipeline.save(model_path)

        import ml_fraud_scorer as scorer_module

        monkeypatch.setattr(scorer_module, "DEFAULT_MODEL_PATH", model_path)
        monkeypatch.setattr(scorer_module, "_pipeline", None)

        df = _review_csv()
        heuristic = simple_fraud_detection(df)
        scored = ml_fraud_detection(df, model_path=model_path)

        assert not scored["is_fraud"].equals(heuristic["is_fraud"]) or (
            scored["fraud_score"].round(4).tolist() != heuristic["fraud_score"].round(4).tolist()
        )
        assert "model_score" in scored.columns
        assert scored["fraud_score"].between(0, 1).all()


def test_ml_card_amount_series_uses_model_score(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "fraud_model.pkl")
        csv_path = os.path.join(tmp, "train.csv")
        _sample_labeled_df().to_csv(csv_path, index=False)
        pipeline = FraudDetectionPipeline(model_threshold=0.3)
        pipeline.fit(csv_path, train_frac=0.7)
        pipeline.save(model_path)

        import ml_fraud_scorer as scorer_module

        monkeypatch.setattr(scorer_module, "DEFAULT_MODEL_PATH", model_path)
        monkeypatch.setattr(scorer_module, "_pipeline", None)

        scored = ml_fraud_detection(_review_csv(), model_path=model_path)
        row = scored[scored["transaction_id"] == "tx_004"].iloc[0]
        series = json.loads(row["card_amount_series_json"])
        assert series[-1]["risk_score"] == pytest.approx(float(row["fraud_score"]))
