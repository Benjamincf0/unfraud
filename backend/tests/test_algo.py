import json
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from algo.algo import (
    DriftMonitor,
    FraudDetectionPipeline,
    _feature_reason_detail,
    _feature_signal_type,
    apply_rule_guardrails,
    assess_feature_separation,
    build_features,
    format_alert_reason,
    load,
    shrink,
    shap_reason_codes,
    shap_score_breakdown_for_rows,
    train_model,
    build_shap_explainer,
    prepare_matrix,
    temporal_split,
    validate_dataset_labels,
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


def test_high_priority_features_present_and_sane():
    g = build_features(shrink(load(_write_sample_csv())))
    for col in (
        "amt_z_vs_category",
        "distinct_categories_24h",
        "hour_rarity_for_card",
        "hour_never_seen_for_card",
        "merchant_tx_30m",
        "merchant_unique_cards_2h",
    ):
        assert col in g.columns
    fraud_row = g[g["transaction_id"] == "tx_fraud"].iloc[0]
    assert fraud_row["hour_never_seen_for_card"] == 1 or fraud_row["hour_rarity_for_card"] >= 0.85
    assert fraud_row["amt_z_vs_card"] > 0


def _merchant_burst_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["tx_m1", "tx_m2", "tx_m3", "tx_m4", "tx_late"],
            "timestamp": pd.to_datetime(
                [
                    "2026-04-25T10:00:00",
                    "2026-04-25T10:05:00",
                    "2026-04-25T10:10:00",
                    "2026-04-25T10:20:00",
                    "2026-04-25T12:30:00",
                ]
            ),
            "card_id": ["card_001", "card_001", "card_002", "card_003", "card_004"],
            "amount": [11.0, 12.0, 10.0, 13.0, 12.0],
            "merchant_name": ["Burst Shop"] * 5,
            "merchant_category": ["online_retail"] * 5,
            "channel": ["online"] * 5,
            "cardholder_country": ["US"] * 5,
            "merchant_country": ["US"] * 5,
            "device_id": ["dev_1", "dev_1", "dev_2", "dev_3", "dev_4"],
            "ip_address": ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
        }
    )


def test_merchant_cross_card_features_use_merchant_windows():
    g = build_features(shrink(_merchant_burst_df()))
    burst_row = g[g["transaction_id"] == "tx_m4"].iloc[0]
    late_row = g[g["transaction_id"] == "tx_late"].iloc[0]

    assert burst_row["merchant_tx_30m"] == 4
    assert burst_row["merchant_unique_cards_2h"] == 3
    assert late_row["merchant_tx_30m"] == 1
    assert late_row["merchant_unique_cards_2h"] == 1


def test_merchant_burst_guardrail_requires_multiple_cards():
    g = apply_rule_guardrails(build_features(shrink(_merchant_burst_df())))
    repeated_same_card = g[g["transaction_id"] == "tx_m2"].iloc[0]
    burst_row = g[g["transaction_id"] == "tx_m4"].iloc[0]

    assert not repeated_same_card["rule_merchant_burst"]
    assert burst_row["rule_merchant_burst"]
    assert "merchant burst" in " ".join(burst_row["rule_reason_codes"])


def test_merchant_burst_reason_is_cross_card():
    row = build_features(shrink(_merchant_burst_df()))
    burst_row = row[row["transaction_id"] == "tx_m4"].iloc[0]

    assert _feature_signal_type("merchant_unique_cards_2h") == "cross_card"
    assert _feature_reason_detail("merchant_unique_cards_2h", burst_row) == (
        "Merchant appears on 3 distinct cards in the last 2 hours."
    )


def test_amt_z_vs_category_uses_prior_category_history():
    df = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "timestamp": pd.to_datetime(
                ["2026-04-25T10:00:00", "2026-04-25T10:30:00", "2026-04-25T11:00:00"]
            ),
            "card_id": ["c1", "c2", "c3"],
            "amount": [10.0, 12.0, 500.0],
            "merchant_name": ["A", "B", "C"],
            "merchant_category": ["grocery", "grocery", "grocery"],
            "channel": "online",
            "cardholder_country": "US",
            "merchant_country": "US",
            "device_id": ["d1", "d2", "d3"],
            "ip_address": ["1.1.1.1", "2.2.2.2", "3.3.3.3"],
        }
    )
    g = build_features(shrink(df))
    assert g.iloc[2]["amt_z_vs_category"] > 2.0


def test_temporal_split_train_val_test():
    g = _feature_frame()
    train, val, test = temporal_split(g, train_frac=0.6, val_frac=0.2)
    assert len(train) + len(val) + len(test) == len(g)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()
    assert len(train) > 0 and len(val) > 0 and len(test) > 0


def test_pipeline_threshold_picked_on_validation_not_test():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "train.csv")
        _sample_labeled_df().to_csv(csv_path, index=False)
        pipeline = FraudDetectionPipeline()
        pipeline.fit(csv_path, train_frac=0.6, val_frac=0.2)
        threshold_after_fit = pipeline.threshold
        assert pipeline._threshold_tuned_on_val
        pipeline.evaluate()
        assert pipeline.threshold == threshold_after_fit


def test_validate_dataset_labels():
    df = _sample_labeled_df()
    result = validate_dataset_labels(df, verbose=False)
    assert "amount" in result["features"]
    assert result["features"]["amount"]["n_fraud"] >= 1
    assert result["features"]["amount"]["n_legit"] >= 1


def test_assess_feature_separation_disjoint():
    df = pd.DataFrame(
        {
            "is_fraud": [0, 0, 0, 1, 1, 1],
            "amount": [1.0, 2.0, 3.0, 100.0, 110.0, 120.0],
        }
    )
    stats = assess_feature_separation(df, "amount")
    assert stats["cleanly_separated"]


def test_rule_guardrails_trigger_on_anomaly():
    g = _feature_frame()
    fraud_row = g[g["transaction_id"] == "tx_fraud"].iloc[0]
    assert fraud_row["rule_guardrail"]
    codes = " ".join(fraud_row["rule_reason_codes"])
    assert "amount" in codes or "cards on this IP" in codes or "geo hop" in codes


def test_format_alert_reason():
    text = format_alert_reason(
        ["Amount anomaly"],
        ["9 cards on this IP"],
        model_score=0.91,
        rule_guardrail=True,
    )
    assert text.startswith("flagged")
    assert "Amount anomaly" in text
    assert "9 cards on this IP" in text


def test_shap_score_breakdown_readable():
    g = _feature_frame()
    X, y = prepare_matrix(g)
    model = train_model(X, y)
    explainer = build_shap_explainer(model, X)
    breakdown = shap_score_breakdown_for_rows(explainer, g.iloc[[-1]])[0]
    assert isinstance(breakdown, list)
    if breakdown:
        assert "label" in breakdown[0]
        assert "detail" in breakdown[0]
        assert "sigma" not in breakdown[0]["detail"].lower()
        assert "Model elevated risk" not in breakdown[0]["detail"]
        assert sum(item["weight"] for item in breakdown) == pytest.approx(1.0, abs=0.02)
        assert all(0 < item["weight"] <= 1 for item in breakdown)


def test_shap_breakdown_spend_24h_detail():
    g = _feature_frame()
    row = g.iloc[-1].copy()
    row["spend_24h"] = 2500.0
    detail = _feature_reason_detail("spend_24h", row)
    assert "$2,500.00" in detail or "$2500.00" in detail
    assert "24 hours" in detail


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


def test_pipeline_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "train.csv")
        model_path = os.path.join(tmp, "fraud_model.pkl")
        _sample_labeled_df().to_csv(csv_path, index=False)
        pipeline = FraudDetectionPipeline(model_threshold=0.3)
        pipeline.fit(csv_path, train_frac=0.7)
        pipeline.save(model_path)
        loaded = FraudDetectionPipeline.load(model_path)
        assert loaded.threshold == pipeline.threshold
        scores = loaded.model_scores(pipeline._last_X_te)
        assert len(scores) == len(pipeline._last_X_te)
