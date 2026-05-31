import json
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from algo.algo import (
    DriftMonitor,
    FraudDetectionPipeline,
    apply_rule_guardrails,
    build_features,
    format_alert_reason,
    load,
    shrink,
    shap_reason_codes,
    train_model,
    build_shap_explainer,
    prepare_matrix,
)


def _sample_labeled_df() -> pd.DataFrame:
    rows = []
    shared_ip = "9.9.9.9"
    for i in range(40):
        card = f"card_{i % 5:03d}"
        rows.append(
            {
                "transaction_id": f"tx_{i:03d}",
                "timestamp": f"2026-04-25T{10 + (i // 6):02d}:{(i * 7) % 60:02d}:00",
                "card_id": card,
                "amount": 12.0 + (i % 4),
                "merchant_name": "Store A",
                "merchant_category": "grocery",
                "channel": "online",
                "cardholder_country": "US",
                "merchant_country": "US",
                "device_id": f"dev_{i % 3:03d}",
                "ip_address": shared_ip if i % 2 == 0 else f"10.0.0.{i % 4}",
                "user_age": 35,
                "city_pop": 100000,
                "is_fraud": 0,
            }
        )
    rows.append(
        {
            "transaction_id": "tx_fraud",
            "timestamp": "2026-04-26T03:15:00",
            "card_id": "card_000",
            "amount": 950.0,
            "merchant_name": "QuickPay",
            "merchant_category": "online_retail",
            "channel": "online",
            "cardholder_country": "US",
            "merchant_country": "GB",
            "device_id": "dev_shared",
            "ip_address": shared_ip,
            "user_age": 35,
            "city_pop": 100000,
            "is_fraud": 1,
        }
    )
    return pd.DataFrame(rows)


def _feature_frame():
    return apply_rule_guardrails(build_features(shrink(load(_write_sample_csv()))))


def _write_sample_csv() -> str:
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "unfraud_algo_test.csv")
    _sample_labeled_df().to_csv(path, index=False)
    return path


def test_rule_guardrails_trigger_on_anomaly():
    g = _feature_frame()
    fraud_row = g[g["transaction_id"] == "tx_fraud"].iloc[0]
    assert fraud_row["rule_guardrail"]
    codes = " ".join(fraud_row["rule_reason_codes"])
    assert "amount" in codes or "cards on this IP" in codes or "geo hop" in codes


def test_format_alert_reason():
    text = format_alert_reason(
        ["amount 6σ above card norm"],
        ["9 cards on this IP"],
        model_score=0.91,
        rule_guardrail=True,
    )
    assert text.startswith("flagged")
    assert "6σ above card norm" in text
    assert "9 cards on this IP" in text


def test_shap_reason_codes():
    g = _feature_frame()
    X, y = prepare_matrix(g)
    model = train_model(X, y)
    explainer = build_shap_explainer(model, X)
    row = g.iloc[-1]
    codes = shap_reason_codes(explainer, row, top_k=3)
    assert isinstance(codes, list)


def test_drift_monitor_weekly_and_retrain_schedule():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "metrics.json")
        monitor = DriftMonitor(path)
        monitor.record_weekly(0.72, 100, week="2026-W20")
        monitor.record_weekly(0.70, 100, week="2026-W21")
        drift, _ = monitor.drift_detected()
        assert drift is False
        monitor.record_weekly(0.60, 100, week="2026-W22")
        drift, msg = monitor.drift_detected()
        assert drift is True
        assert "PR-AUC dropped" in msg
        monitor.mark_retrained(0.60)
        data = json.loads(open(path, encoding="utf-8").read())
        assert data["last_retrain_week"] == "2026-W22"


def test_pipeline_hybrid_predict():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "train.csv")
        _sample_labeled_df().to_csv(csv_path, index=False)
        pipeline = FraudDetectionPipeline(model_threshold=0.3)
        pipeline.fit(csv_path, train_frac=0.7)
        scored = pipeline.predict(pipeline._last_test, pipeline._last_X_te)
        assert "alert_reason" in scored.columns
        assert "flagged_by_rules" in scored.columns
        assert scored["fraud_score"].between(0, 1).all()
