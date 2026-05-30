import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import simple_fraud_detection

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
    assert "Amount anomaly" in outlier["fraud_reasons"]

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

if __name__ == '__main__':
    test_simple_fraud_detection()
    test_edge_cases()
    print("All fraud detection tests passed!")
