import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import simple_fraud_detection

def test_simple_fraud_detection():
    """Test the fraud detection function with known cases"""
    # Create test data where fraud conditions will definitely trigger
    # We need to use None/NaN for missing values since the function checks .isna()
    import numpy as np
    data = {
        'transaction_id': ['tx_001', 'tx_002', 'tx_003', 'tx_004'],
        'timestamp': ['2026-04-25T00:00:00'] * 4,
        'card_id': ['card_001', 'card_001', 'card_001', 'card_001'],
        'amount': [10.0, 10.0, 10.0, 100.0],  # Median = 10.0, 3*median = 30.0
        'merchant_name': ['Store A', 'Store B', 'Store C', 'Store D'],
        'merchant_category': ['grocery', 'grocery', 'grocery', 'grocery'],
        'channel': ['online', 'online', 'in_person', 'online'],
        'cardholder_country': ['US', 'CA', 'US', 'US'],  # Foreign for tx_002
        'merchant_country': ['US', 'US', 'US', 'US'],
        'device_id': ['dev_123', None, 'dev_456', None],  # None for online tx_002 and tx_004
        'ip_address': ['1.1.1.1', None, '2.2.2.2', None]
    }
    df = pd.DataFrame(data)
    
    # Apply fraud detection
    result_df = simple_fraud_detection(df)
    
    # Check results
    # tx_001: normal transaction
    assert result_df.loc[0, 'is_fraud'] == False
    
    # tx_002: foreign + missing device/IP for online (but not high amount)
    assert result_df.loc[1, 'is_fraud'] == True  # Should be flagged for foreign and missing device/IP
    assert result_df.loc[1, 'fraud_score'] > 0.0
    assert 'Foreign transaction' in result_df.loc[1, 'fraud_reasons']
    assert 'Missing device/IP for online' in result_df.loc[1, 'fraud_reasons']
    
    # tx_003: normal in-person transaction
    assert result_df.loc[2, 'is_fraud'] == False
    
    # tx_004: high amount (100.0 > 3*10.0=30.0) + missing device/IP for online
    assert result_df.loc[3, 'is_fraud'] == True
    assert result_df.loc[3, 'fraud_score'] > 0.5  # Should be 0.8 (high amount) + 0.4 (missing device/IP) = 1.2 -> capped at 1.0
    assert 'High amount for card' in result_df.loc[3, 'fraud_reasons']
    assert 'Missing device/IP for online' in result_df.loc[3, 'fraud_reasons']

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
    # Amount flag should not trigger when median is 0
    assert result.loc[0, 'is_fraud'] == False

if __name__ == '__main__':
    test_simple_fraud_detection()
    test_edge_cases()
    print("All fraud detection tests passed!")