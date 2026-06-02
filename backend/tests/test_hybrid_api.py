import io
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algo.algo import DEFAULT_MODEL_PATH
from export_challenge_csv import build_analyzed_transactions
from main import app
from ml_fraud_scorer import is_model_available

client = TestClient(app)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRANSACTIONS_PATH = os.path.join(ROOT, "transactions.csv")


@pytest.mark.skipif(
    not os.path.exists(TRANSACTIONS_PATH) or not is_model_available(DEFAULT_MODEL_PATH),
    reason="Challenge CSV and trained model artifact are required",
)
def test_api_hybrid_queue_matches_offline_export():
    csv_bytes = open(TRANSACTIONS_PATH, "rb").read()
    upload = client.post(
        "/upload",
        files={"file": ("transactions.csv", csv_bytes, "text/csv")},
    )
    assert upload.status_code == 200
    file_hash = upload.json()["file_hash"]

    offline = build_analyzed_transactions(pd.read_csv(TRANSACTIONS_PATH))
    offline_flagged = set(
        offline.loc[offline["is_fraud"], "transaction_id"].astype(str)
    )

    summary = client.get(f"/analysis/summary/{file_hash}").json()
    assert summary["model_flagged_count"] == len(offline_flagged)
    assert summary["heuristic_boost_count"] == 7

    queue = client.get(
        f"/analysis/queue/{file_hash}",
        params={"use_model": "true", "limit": 5000},
    ).json()
    api_flagged = {item["transaction_id"] for item in queue["items"]}
    assert api_flagged == offline_flagged
    assert len(api_flagged) == 70

    export = client.get(f"/export/{file_hash}", params={"use_model": "true"})
    assert export.status_code == 200
    exported = pd.read_csv(io.StringIO(export.text))
    export_flagged = set(
        exported.loc[exported["is_fraud"], "transaction_id"].astype(str)
    )
    assert export_flagged == offline_flagged
