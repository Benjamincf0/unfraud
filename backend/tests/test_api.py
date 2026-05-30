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
    # Create a small test CSV that will trigger our fraud detection rules
    # For card_001: amounts [10.0, 10.0] -> median 10.0 -> 3*median = 30.0 (none will trigger high amount)
    # For card_002: amounts [20.0, 20.0] -> median 20.0 -> 3*median = 60.0 (none will trigger high amount)
    # So we should expect fraud only from foreign transactions and missing device/IP for online
    csv_data = """transaction_id,timestamp,card_id,amount,merchant_name,merchant_category,channel,cardholder_country,merchant_country,device_id,ip_address
tx_001,2026-04-25T00:00:00,card_001,10.0,Store A,grocery,in_person,US,US,,
tx_002,2026-04-25T00:01:00,card_001,10.0,Store B,grocery,online,CA,US,,
tx_003,2026-04-25T00:02:00,card_002,20.0,Store C,restaurant,in_person,US,US,,
tx_004,2026-04-25T00:03:00,card_002,20.0,Store D,restaurant,online,US,US,,
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
    assert len(analysis) == 4
    
    # Check specific fraud cases based on our actual logic with this data
    fraud_cases = [a for a in analysis if a["is_fraud"]]
    # tx_002: foreign (US->CA) + missing device/IP for online
    # tx_004: missing device/IP for online (but not foreign, US->US)
    assert len(fraud_cases) == 2  # Only tx_002 and tx_004 should be fraud
    
    # Find specific transactions
    tx_002_analysis = next((a for a in analysis if a["transaction_id"] == "tx_002"), None)
    assert tx_002_analysis is not None
    assert tx_002_analysis["is_fraud"] == True
    assert any("Foreign" in r for r in tx_002_analysis["reasons"])
    assert any("Missing device/IP for online" in r for r in tx_002_analysis["reasons"])
    # Should NOT have high amount flag since amounts are too low
    assert not any("High amount for card" in r for r in tx_002_analysis["reasons"])
    
    tx_004_analysis = next((a for a in analysis if a["transaction_id"] == "tx_004"), None)
    assert tx_004_analysis is not None
    assert tx_004_analysis["is_fraud"] == True
    assert any("Missing device/IP for online" in r for r in tx_004_analysis["reasons"])
    # Should NOT have foreign flag since it's US->US
    assert not any("Foreign" in r for r in tx_004_analysis["reasons"])
    # Should NOT have high amount flag since amounts are too low
    assert not any("High amount for card" in r for r in tx_004_analysis["reasons"])
    
    # Test user analysis
    response = client.get(f"/analysis/user/{file_hash}/card_001")
    assert response.status_code == 200
    user_analysis = response.json()
    assert len(user_analysis) == 2  # tx_001 and tx_002
    # tx_001 should not be fraud, tx_002 should be fraud
    fraud_in_user = [a for a in user_analysis if a["is_fraud"]]
    assert len(fraud_in_user) == 1
    assert fraud_in_user[0]["transaction_id"] == "tx_002"
    
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
    assert len(lines) == 5  # Header + 4 rows
    header = lines[0]
    assert "is_fraud" in header
    assert "fraud_score" in header
    assert "fraud_reasons" in header
    
    # Check first data row (tx_001) - should not be fraud
    first_row = lines[1].split(',')
    is_fraud_idx = header.split(',').index('is_fraud')
    assert first_row[is_fraud_idx] == 'False'
    
    # Check second data row (tx_002) - should be fraud
    second_row = lines[2].split(',')
    assert second_row[is_fraud_idx] == 'True'

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
    
    # Test invalid action
    response = client.post(
        f"/review/{file_hash}/{tx_id}/invalid",
        json={"action": "invalid", "reviewer_notes": "Test"}
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