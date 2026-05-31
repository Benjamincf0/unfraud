import pandas as pd
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fraud_scorer import simple_fraud_detection

def test_simple_fraud_detection():
    """Test per-card + cross-card explainable scoring."""
    data = {
        'transaction_id': ['tx_001', 'tx_002', 'tx_003', 'tx_004', 'tx_005', 'tx_006'],
        'timestamp': [
            '2026-04-25T00:00:00',
            '2026-04-25T00:05:00',
            '2026-04-25T00:10:00',
            '2026-04-25T00:15:00',
            '2026-04-25T00:20:00',
            '2026-04-25T00:25:00',
        ],
        'card_id': ['card_001', 'card_001', 'card_001', 'card_001', 'card_002', 'card_003'],
        'amount': [12.0, 11.0, 13.0, 140.0, 85.0, 95.0],
        'merchant_name': ['Store A', 'Store A', 'Store A', 'Store A', 'Store A', 'Store A'],
        'merchant_category': ['grocery', 'grocery', 'grocery', 'electronics', 'electronics', 'electronics'],
        'channel': ['online', 'online', 'online', 'online', 'online', 'online'],
        'cardholder_country': ['US', 'US', 'US', 'US', 'US', 'US'],
        'merchant_country': ['US', 'US', 'US', 'GB', 'GB', 'GB'],
        'device_id': ['dev_shared', 'dev_shared', 'dev_shared', 'dev_shared', 'dev_shared', 'dev_shared'],
        'ip_address': ['9.9.9.9', '9.9.9.9', '9.9.9.9', '9.9.9.9', '9.9.9.9', '9.9.9.9']
    }
    df = pd.DataFrame(data)

    result_df = simple_fraud_detection(df)

    assert len(result_df) == 6
    assert result_df["fraud_score"].between(0.0, 1.0).all()

    # Per-card anomaly: big jump for card_001
    outlier = result_df[result_df["transaction_id"] == "tx_004"].iloc[0]
    assert outlier["fraud_score"] > 0.55
    assert bool(outlier["is_fraud"]) == True
    assert "Amount anomaly" in outlier["fraud_reasons"] or "Category amount anomaly" in outlier["fraud_reasons"]

    # Cross-card aggregation: same device/ip reused across multiple cards
    cross_card_row = result_df[result_df["transaction_id"] == "tx_006"].iloc[0]
    assert int(cross_card_row["device_card_fanout"]) >= 3
    assert int(cross_card_row["ip_card_fanout"]) >= 3
    assert "Device shared across cards" in cross_card_row["fraud_reasons"] or \
           "IP shared across cards" in cross_card_row["fraud_reasons"]

    # Explainable payload columns are present
    assert isinstance(cross_card_row["score_breakdown"], str)
    assert isinstance(cross_card_row["card_baseline_json"], str)
    assert isinstance(cross_card_row["cross_card_signals_json"], str)
    assert isinstance(cross_card_row["graph_features_json"], str)
    assert isinstance(cross_card_row["card_amount_series_json"], str)

def test_edge_cases():
    """Test edge cases"""
    # Empty dataframe
    empty_df = pd.DataFrame(columns=[
        'transaction_id', 'timestamp', 'card_id', 'amount', 'merchant_name',
        'merchant_category', 'channel', 'cardholder_country', 'merchant_country',
        'device_id', 'ip_address'
    ])
    result = simple_fraud_detection(empty_df)
    assert len(result) == 0
    assert "score_breakdown" in result.columns
    assert "card_baseline_json" in result.columns
    
    # Single row with no fraud indicators
    single_df = pd.DataFrame([{
        'transaction_id': 'tx_001',
        'timestamp': '2026-04-25T00:00:00',
        'card_id': 'card_001',
        'amount': 10.0,
        'merchant_name': 'Store A',
        'merchant_category': 'grocery',
        'channel': 'in_person',
        'cardholder_country': 'US',
        'merchant_country': 'US',
        'device_id': None,
        'ip_address': None
    }])
    result = simple_fraud_detection(single_df)
    assert result.loc[0, 'is_fraud'] == False
    assert result.loc[0, 'fraud_score'] <= 0.55
    
    # Test with zero median (all amounts zero)
    zero_df = pd.DataFrame([{
        'transaction_id': 'tx_001',
        'timestamp': '2026-04-25T00:00:00',
        'card_id': 'card_001',
        'amount': 0.0,
        'merchant_name': 'Store A',
        'merchant_category': 'grocery',
        'channel': 'in_person',
        'cardholder_country': 'US',
        'merchant_country': 'US',
        'device_id': 'dev_123',
        'ip_address': '1.1.1.1'
    }])
    result = simple_fraud_detection(zero_df)
    assert result.loc[0, 'amount_ratio'] == 1.0
    assert result.loc[0, 'is_fraud'] == False


def _reason_codes(row):
    return {reason["code"] for reason in json.loads(row["score_breakdown"])}


def test_heuristic_covers_velocity_and_card_testing_patterns():
    df = pd.DataFrame({
        "transaction_id": ["tx_001", "tx_002", "tx_003", "tx_004", "tx_005"],
        "timestamp": [
            "2026-04-25T10:00:00",
            "2026-04-25T10:01:00",
            "2026-04-25T10:02:00",
            "2026-04-25T10:03:00",
            "2026-04-25T10:04:00",
        ],
        "card_id": ["card_velocity"] * 5,
        "amount": [20.0, 21.0, 19.0, 20.0, 5.0],
        "merchant_name": ["Store A", "Store A", "Store A", "Store A", "QuickPay Online"],
        "merchant_category": ["grocery", "grocery", "grocery", "grocery", "online_retail"],
        "channel": ["online"] * 5,
        "cardholder_country": ["US"] * 5,
        "merchant_country": ["US"] * 5,
        "device_id": ["dev_1", "dev_1", "dev_1", "dev_1", "dev_new"],
        "ip_address": ["10.0.0.1", "10.0.0.1", "10.0.0.1", "10.0.0.1", "10.0.0.2"],
    })

    result = simple_fraud_detection(df)
    row = result[result["transaction_id"] == "tx_005"].iloc[0]
    codes = _reason_codes(row)

    assert bool(row["is_fraud"]) is True
    assert "velocity_spike" in codes
    assert "card_testing_amount" in codes
    assert int(row["tx_5min"]) == 5


def test_heuristic_covers_time_and_geo_hop_patterns():
    df = pd.DataFrame({
        "transaction_id": ["tx_001", "tx_002", "tx_003", "tx_004"],
        "timestamp": [
            "2026-04-22T10:00:00",
            "2026-04-23T10:00:00",
            "2026-04-24T10:00:00",
            "2026-04-24T10:30:00",
        ],
        "card_id": ["card_geo"] * 4,
        "amount": [30.0, 31.0, 29.0, 160.0],
        "merchant_name": ["Store A", "Store A", "Store A", "Travel Site"],
        "merchant_category": ["grocery", "grocery", "grocery", "travel"],
        "channel": ["online"] * 4,
        "cardholder_country": ["US"] * 4,
        "merchant_country": ["US", "US", "US", "GB"],
        "device_id": ["dev_1"] * 4,
        "ip_address": ["10.0.0.1"] * 4,
    })

    result = simple_fraud_detection(df)
    row = result[result["transaction_id"] == "tx_004"].iloc[0]
    codes = _reason_codes(row)

    assert bool(row["is_fraud"]) is True
    assert "rapid_geo_hop" in codes
    assert int(row["fast_country_hop"]) == 1


def test_heuristic_covers_threshold_and_missing_identity_patterns():
    df = pd.DataFrame({
        "transaction_id": ["tx_001", "tx_002", "tx_003", "tx_004"],
        "timestamp": [
            "2026-04-22T09:00:00",
            "2026-04-23T09:00:00",
            "2026-04-24T09:00:00",
            "2026-04-25T09:00:00",
        ],
        "card_id": ["card_identity"] * 4,
        "amount": [12.0, 13.0, 11.0, 99.75],
        "merchant_name": ["Store A", "Store A", "Store A", "QuickPay Online"],
        "merchant_category": ["grocery", "grocery", "grocery", "online_retail"],
        "channel": ["online"] * 4,
        "cardholder_country": ["US"] * 4,
        "merchant_country": ["US"] * 4,
        "device_id": ["dev_1", "dev_1", "dev_1", None],
        "ip_address": ["10.0.0.1", "10.0.0.1", "10.0.0.1", None],
    })

    result = simple_fraud_detection(df)
    row = result[result["transaction_id"] == "tx_004"].iloc[0]
    codes = _reason_codes(row)

    assert bool(row["is_fraud"]) is True
    assert "threshold_probe_amount" in codes
    assert "identity_missing" in codes
    assert int(row["device_missing"]) == 1
    assert int(row["ip_missing"]) == 1

if __name__ == '__main__':
    test_simple_fraud_detection()
    test_edge_cases()
    print("All fraud detection tests passed!")
