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


def test_scoring_status_endpoint():
    response = client.get("/scoring/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["heuristic"] is True
    assert "ml_model_available" in payload

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

def test_analysis_summary_and_queue_endpoints():
    csv_data = """transaction_id,timestamp,card_id,amount,merchant_name,merchant_category,channel,cardholder_country,merchant_country,device_id,ip_address
tx_001,2026-04-25T00:00:00,card_001,12.0,Store A,grocery,online,US,US,dev_1,10.0.0.1
tx_002,2026-04-25T00:01:00,card_001,11.0,Store A,grocery,online,US,US,dev_1,10.0.0.1
tx_003,2026-04-25T00:02:00,card_001,14.0,Store A,grocery,online,US,US,dev_1,10.0.0.1
tx_004,2026-04-25T00:03:00,card_001,120.0,Store A,electronics,online,US,GB,dev_1,10.0.0.1
tx_005,2026-04-25T00:04:00,card_002,60.0,Store A,electronics,online,US,GB,dev_1,10.0.0.1
tx_006,2026-04-25T00:05:00,card_003,62.0,Store A,electronics,online,US,GB,dev_1,10.0.0.1
"""

    response = client.post(
        "/upload",
        files={"file": ("test.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    file_hash = response.json()["file_hash"]

    response = client.get(f"/analysis/summary/{file_hash}")
    assert response.status_code == 200
    summary = response.json()
    assert summary["total_transactions"] == 6
    assert summary["flagged_count"] >= 1
    assert "model_flagged_count" in summary
    assert "ml_model_available" in summary
    assert summary["flagged_queue_stats"]["pending"] >= 1
    assert (
        summary["flagged_queue_stats"]["pending"]
        + summary["flagged_queue_stats"]["approved"]
        + summary["flagged_queue_stats"]["dismissed"]
        + summary["flagged_queue_stats"]["escalated"]
        == summary["flagged_count"]
    )

    response = client.get(f"/analysis/queue/{file_hash}?limit=2")
    assert response.status_code == 200
    queue = response.json()
    assert queue["total"] >= 1
    assert len(queue["items"]) <= 2
    assert "transaction_id" in queue["items"][0]
    assert "fraud_score" in queue["items"][0]
    assert "card_baseline" in queue["items"][0]

    response = client.get(f"/analysis/queue/{file_hash}?limit=2&slim=true")
    assert response.status_code == 200
    slim_queue = response.json()
    assert slim_queue["items"][0]["card_baseline"] == {}

    tx_id = queue["items"][0]["transaction_id"]
    response = client.get(
        f"/analysis/queue/{file_hash}?transaction_id={tx_id}",
    )
    assert response.status_code == 200
    assert len(response.json()["items"]) == 1

    response = client.get(f"/analysis/transaction/{file_hash}/{tx_id}")
    assert response.status_code == 200
    detail = response.json()
    assert detail["transaction_id"] == tx_id
    assert "heuristic" in detail
    assert "score_breakdown" in detail["heuristic"]

    response = client.get(f"/analysis/related/{file_hash}/{tx_id}")
    assert response.status_code == 200
    related = response.json()
    assert len(related["items"]) >= 1

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


def test_escalated_review_ip_updates_scoring():
    csv_data = """transaction_id,timestamp,card_id,amount,merchant_name,merchant_category,channel,cardholder_country,merchant_country,device_id,ip_address
tx_feedback_001,2026-04-25T00:00:00,card_feedback_001,10.0,Corner Shop,grocery,online,US,US,dev_feedback_001,8.8.8.8
tx_feedback_002,2026-04-25T00:01:00,card_feedback_002,12.0,Corner Shop,grocery,online,US,US,dev_feedback_002,8.8.8.8
tx_feedback_003,2026-04-25T00:02:00,card_feedback_003,11.0,Corner Shop,grocery,online,US,US,dev_feedback_003,7.7.7.7
"""

    response = client.post(
        "/upload",
        files={"file": ("feedback.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    file_hash = response.json()["file_hash"]

    response = client.get(
        f"/analysis/transaction/{file_hash}/tx_feedback_002",
    )
    assert response.status_code == 200
    baseline = response.json()["heuristic"]
    baseline_score = baseline["fraud_score"]
    assert "Previously escalated IP" not in baseline["reasons"]

    response = client.post(
        f"/review/{file_hash}/tx_feedback_001/escalate",
        json={"action": "escalate", "reviewer_notes": "Suspicious IP"},
    )
    assert response.status_code == 200

    response = client.get(
        f"/analysis/transaction/{file_hash}/tx_feedback_002",
    )
    assert response.status_code == 200
    updated = response.json()["heuristic"]
    assert updated["is_fraud"] is True
    assert updated["fraud_score"] > baseline_score
    assert "Previously escalated IP" in updated["reasons"]
    assert any(
        reason["code"] == "review_escalated_ip"
        for reason in updated["score_breakdown"]
    )
    assert (
        updated["cross_card_signals"]["review_escalated_ip_transactions"] == 1
    )

    response = client.post(
        f"/review/{file_hash}/tx_feedback_001/pending",
        json={"action": "pending"},
    )
    assert response.status_code == 200

    response = client.get(
        f"/analysis/transaction/{file_hash}/tx_feedback_002",
    )
    assert response.status_code == 200
    reverted = response.json()["heuristic"]
    assert "Previously escalated IP" not in reverted["reasons"]
    assert reverted["fraud_score"] == baseline_score


def test_review_reason_tunes_heuristic_weights():
    csv_data = """transaction_id,timestamp,card_id,amount,merchant_name,merchant_category,channel,cardholder_country,merchant_country,device_id,ip_address
tx_tune_001,2026-04-25T00:00:00,card_tune_001,10.0,Store A,grocery,online,US,US,dev_tune_001,6.6.6.1
tx_tune_002,2026-04-25T00:01:00,card_tune_001,11.0,Store A,grocery,online,US,US,dev_tune_001,6.6.6.1
tx_tune_003,2026-04-25T00:02:00,card_tune_001,12.0,Store A,grocery,online,US,US,dev_tune_001,6.6.6.1
tx_tune_004,2026-04-25T00:03:00,card_tune_001,80.0,Store B,electronics,online,US,US,dev_tune_002,6.6.6.2
"""

    response = client.post(
        "/upload",
        files={"file": ("tuning.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    file_hash = response.json()["file_hash"]

    response = client.get(f"/analysis/transaction/{file_hash}/tx_tune_004")
    assert response.status_code == 200
    baseline = response.json()["heuristic"]
    baseline_score = baseline["fraud_score"]
    assert any(
        signal["code"] == "amount_outlier"
        for signal in baseline["score_breakdown"]
    )

    response = client.post(
        f"/review/{file_hash}/tx_tune_004/dismiss",
        json={
            "action": "dismiss",
            "reviewer_notes": "Cardholder confirmed this purchase",
            "feedback_reason_codes": ["amount_outlier"],
            "feedback_reasoning": "Amount spikes are normal for this card.",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["feedback_reason_codes"] == ["amount_outlier"]

    response = client.get(f"/analysis/transaction/{file_hash}/tx_tune_004")
    assert response.status_code == 200
    tuned = response.json()["heuristic"]
    assert tuned["fraud_score"] < baseline_score

    response = client.get(f"/review-log/{file_hash}")
    assert response.status_code == 200
    entry = response.json()[0]
    assert entry["feedback_reason_codes"] == ["amount_outlier"]
    assert entry["feedback_reasoning"] == "Amount spikes are normal for this card."


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
