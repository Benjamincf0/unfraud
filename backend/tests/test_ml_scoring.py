import os
import sys
import tempfile

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from algo.algo import FraudDetectionPipeline
from main import app
from tests.test_algo import _sample_labeled_df

client = TestClient(app)


def _upload_sample_csv() -> str:
    csv_data = """transaction_id,timestamp,card_id,amount,merchant_name,merchant_category,channel,cardholder_country,merchant_country,device_id,ip_address
tx_001,2026-04-25T00:00:00,card_001,12.0,Store A,grocery,online,US,US,dev_1,10.0.0.1
tx_002,2026-04-25T00:01:00,card_001,11.0,Store A,grocery,online,US,US,dev_1,10.0.0.1
tx_003,2026-04-25T00:02:00,card_001,14.0,Store A,grocery,online,US,US,dev_1,10.0.0.1
tx_004,2026-04-25T00:03:00,card_001,120.0,Store A,electronics,online,US,GB,dev_1,10.0.0.1
"""
    response = client.post(
        "/upload",
        files={"file": ("test.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    return response.json()["file_hash"]


def test_use_model_requires_trained_artifact():
    file_hash = _upload_sample_csv()
    response = client.get(f"/analysis/all/{file_hash}?use_model=true")
    if response.status_code == 503:
        assert "artifact" in response.json()["detail"].lower()
        return

    analysis = response.json()
    assert len(analysis) == 4
    assert "score_breakdown" in analysis[0]


def test_use_model_scoring_with_temporary_artifact(monkeypatch):
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

        file_hash = _upload_sample_csv()
        response = client.get(f"/analysis/all/{file_hash}?use_model=true")
        assert response.status_code == 200
        analysis = response.json()
        assert len(analysis) == 4
        assert all(item["fraud_score"] >= 0 for item in analysis)

        heuristic = client.get(f"/analysis/all/{file_hash}?use_model=false").json()
        assert len(heuristic) == len(analysis)
