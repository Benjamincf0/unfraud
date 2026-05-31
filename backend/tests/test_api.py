import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import app
from fastapi.testclient import TestClient
import pandas as pd

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_file_upload_and_analysis():
    """Test complete upload -> analysis workflow"""
    csv_data = """transaction_id,timestamp,card_id,amount,merchant_name,merchant_category,channel,cardholder_country,merchant_country,device_id,ip_address
tx_001,2026-04-25T00:00:00,card_001,12.0,Store A,grocery,online,US,US,dev_1,10.0.0.1
tx_002,2026-04-25T00:01:00,card_001,11.0,Store A,grocery,online,US,US,dev_1,10.0.0.1
tx_003,2026-04-25T00:02:00,card_001,14.0,Store A,grocery,online,US,US,dev_1,10.0.0.1
tx_004,2026-04-25T00:03:00,card_001,120.0,Store A,electronics,online,US,GB,dev_1,10.0.0.1
tx_005,2026-04-25T00:04:00,card_002,60.0,Store A,electronics,online,US,GB,dev_1,10.0.0.1
tx_006,2026-04-25T00:05:00,card_003,62.0,Store A,electronics,online,US,GB,dev_1,10.0.0.1
"""
    
    # Upload file
    response = client.post(
        "/upload",
        files={"file": ("test.csv", csv_data, "text/csv")}
    )
    assert response.status_code == 200
    upload_data = response.json()
    file_hash = upload_data["file_hash"]
    assert "message" in upload_data
    
    # Test duplicate upload
    response = client.post(
        "/upload",
        files={"file": ("test.csv", csv_data, "text/csv")}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "File already uploaded"
    
    # Test all analysis
    response = client.get(f"/analysis/all/{file_hash}")
    assert response.status_code == 200
    analysis = response.json()
    assert len(analysis) == 6

    # Check that enriched schema is returned
    sample = analysis[0]
    assert "score_breakdown" in sample
    assert "card_baseline" in sample
    assert "cross_card_signals" in sample
    assert "graph_features" in sample
    assert "card_amount_series" in sample

    # Check at least one flagged transaction
    fraud_cases = [a for a in analysis if a["is_fraud"]]
    assert len(fraud_cases) >= 1

    # tx_004 should score high due to per-card jump
    tx_004_analysis = next((a for a in analysis if a["transaction_id"] == "tx_004"), None)
    assert tx_004_analysis is not None
    assert tx_004_analysis["fraud_score"] > 0.5
    assert any("Amount anomaly" in r for r in tx_004_analysis["reasons"])

    # tx_006 should include cross-card fanout evidence
    tx_006_analysis = next((a for a in analysis if a["transaction_id"] == "tx_006"), None)
    assert tx_006_analysis is not None
    assert tx_006_analysis["cross_card_signals"]["device_card_fanout"] >= 3
    assert tx_006_analysis["cross_card_signals"]["ip_card_fanout"] >= 3
    assert any(
        "cross" in reason["signal_type"]
        for reason in tx_006_analysis["score_breakdown"]
    )
    
    # Test user analysis
    response = client.get(f"/analysis/user/{file_hash}/card_001")
    assert response.status_code == 200
    user_analysis = response.json()
    assert len(user_analysis) == 4
    
    # Test IP analysis
    response = client.get(f"/analysis/ip/{file_hash}/1.1.1.1")  # Non-existent IP
    assert response.status_code == 404  # Should not find this IP
    
    # Test export
    response = client.get(f"/export/{file_hash}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert "attachment;" in response.headers["content-disposition"]
    
    # Check CSV content
    csv_content = response.text
    lines = csv_content.strip().split('\n')
    assert len(lines) == 7  # Header + 6 rows
    header = lines[0]
    assert "is_fraud" in header
    assert "fraud_score" in header
    assert "fraud_reasons" in header
    
    assert "score_breakdown" in header
    assert "card_baseline_json" in header
    assert "cross_card_signals_json" in header
    assert "review_decision" in header
    assert "reviewer_notes" in header
    assert "reviewed_at" in header

def test_review_endpoints():
    """Test review endpoints"""
    # Upload a file first
    csv_data = """transaction_id,timestamp,card_id,amount,merchant_name,merchant_category,channel,cardholder_country,merchant_country,device_id,ip_address
tx_001,2026-04-25T00:00:00,card_001,10.0,Store A,grocery,in_person,US,US,,
tx_002,2026-04-25T00:01:00,card_001,50.0,Store B,grocery,online,CA,US,dev_123,1.1.1.1"""
    
    response = client.post("/upload", files={"file": ("test.csv", csv_data, "text/csv")})
    file_hash = response.json()["file_hash"]
    
    # Get a transaction ID
    response = client.get(f"/analysis/all/{file_hash}")
    tx_id = response.json()[0]["transaction_id"]
    
    # Test approve action
    response = client.post(
        f"/review/{file_hash}/{tx_id}/approve",
        json={"action": "approve", "reviewer_notes": "Looks legit"}
    )
    assert response.status_code == 200
    assert "approve" in response.json()["message"]
    
    # Test dismiss action
    response = client.post(
        f"/review/{file_hash}/{tx_id}/dismiss",
        json={"action": "dismiss", "reviewer_notes": "False positive"}
    )
    assert response.status_code == 200
    assert "dismiss" in response.json()["message"]
    
    # Test escalate action
    response = client.post(
        f"/review/{file_hash}/{tx_id}/escalate",
        json={"action": "escalate", "reviewer_notes": "Needs investigation"}
    )
    assert response.status_code == 200
    assert "escalate" in response.json()["message"]

    response = client.get(f"/review/{file_hash}/audit")
    assert response.status_code == 200
    audit = response.json()
    assert len(audit) == 1
    assert audit[0]["transaction_id"] == tx_id
    assert audit[0]["action"] == "escalate"
    assert audit[0]["reviewed_at"]

    response = client.get(f"/export/{file_hash}")
    assert response.status_code == 200
    assert "review_decision" in response.text
    assert "escalate" in response.text

    response = client.post(
        f"/review/{file_hash}/{tx_id}/pending",
        json={"action": "pending"}
    )
    assert response.status_code == 200

    response = client.get(f"/review/{file_hash}/audit")
    assert response.status_code == 200
    assert response.json() == []
    
    # Test invalid action
    response = client.post(
        f"/review/{file_hash}/{tx_id}/invalid",
        json={"action": "invalid", "reviewer_notes": "Test"}
    )
    assert response.status_code == 400

    response = client.post(
        f"/review/{file_hash}/{tx_id}/approve",
        json={"action": "dismiss", "reviewer_notes": "Mismatch"}
    )
    assert response.status_code == 400

def test_error_conditions():
    """Test error handling"""
    # Test with non-existent file hash
    response = client.get("/analysis/all/nonexistenthash")
    assert response.status_code == 404
    
    # Test upload with missing columns
    bad_csv = """transaction_id,timestamp,amount
tx_001,2026-04-25T00:00:00,10.0"""
    
    response = client.post("/upload", files={"file": ("bad.csv", bad_csv, "text/csv")})
    assert response.status_code == 400
    assert "Missing required columns" in response.json()["detail"]
