import hashlib
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

# In-memory storage for uploaded files. This is intentionally process-local for
# the challenge demo; exported CSVs include review decisions for handoff.
uploaded_files: Dict[str, pd.DataFrame] = {}
analysis_cache: Dict[str, pd.DataFrame] = {}
review_log: Dict[str, Dict[str, Any]] = {}

# Pydantic models
class UploadResponse(BaseModel):
    file_hash: str
    message: str

class TransactionBase(BaseModel):
    transaction_id: str
    timestamp: str
    card_id: str
    amount: float
    merchant_name: str
    merchant_category: str
    channel: str
    cardholder_country: str
    merchant_country: str
    device_id: Optional[str] = None
    ip_address: Optional[str] = None

class FraudAnalysis(BaseModel):
    transaction_id: str
    timestamp: str
    card_id: str
    amount: float
    merchant_name: str
    merchant_category: str
    channel: str
    is_fraud: bool
    fraud_score: float
    reasons: List[str]
    score_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    card_baseline: Dict[str, Any] = Field(default_factory=dict)
    cross_card_signals: Dict[str, Any] = Field(default_factory=dict)
    graph_features: Dict[str, float] = Field(default_factory=dict)
    card_amount_series: List[Dict[str, Any]] = Field(default_factory=list)

class AnalyzedTransaction(TransactionBase):
    is_fraud: bool
    fraud_score: float
    fraud_reasons: List[str]
    reasons: List[str] = Field(default_factory=list)

class ReviewAction(BaseModel):
    action: str  # approve, dismiss, escalate, pending
    reviewer_notes: Optional[str] = None

class ReviewLogEntry(BaseModel):
    transaction_id: str
    action: str
    reviewer_notes: Optional[str] = None
    reviewed_at: str

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    # Read file content
    content = await file.read()
    
    # Generate hash of the blob
    file_hash = hashlib.sha256(content).hexdigest()
    
    # Check if file already uploaded
    if file_hash in uploaded_files:
        review_log.setdefault(file_hash, {})
        return UploadResponse(
            file_hash=file_hash,
            message="File already uploaded"
        )
    
    # Parse CSV into DataFrame
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        # Validate required columns
        required_columns = ['transaction_id', 'timestamp', 'card_id', 'amount', 
                          'merchant_name', 'merchant_category', 'channel',
                          'cardholder_country', 'merchant_country']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Normalize optional columns for downstream logic
        if "device_id" not in df.columns:
            df["device_id"] = pd.NA
        if "ip_address" not in df.columns:
            df["ip_address"] = pd.NA

        # Store DataFrame
        uploaded_files[file_hash] = df
        analysis_cache.pop(file_hash, None)
        review_log[file_hash] = {}
        
        return UploadResponse(
            file_hash=file_hash,
            message="File uploaded successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing CSV: {str(e)}"
        )

def _top_values(series: pd.Series, limit: int = 4) -> List[str]:
    values = series.dropna().astype(str)
    if values.empty:
        return []
    return values.value_counts().head(limit).index.tolist()


def _build_card_amount_series(df: pd.DataFrame, points: int = 12) -> Dict[str, List[List[Dict[str, Any]]]]:
    series_map: Dict[str, List[List[Dict[str, Any]]]] = {}
    for card_id, card_df in df.groupby("card_id", sort=False):
        card_df = card_df.sort_values("timestamp_dt")
        running: List[Dict[str, Any]] = []
        snapshots: List[List[Dict[str, Any]]] = []
        for _, row in card_df.iterrows():
            running.append(
                {
                    "timestamp": row["timestamp_dt"].isoformat(),
                    "amount": float(row["amount"]),
                    "risk_score": round(float(row["fraud_score"]), 4),
                }
            )
            snapshots.append(running[-points:].copy())
        series_map[str(card_id)] = snapshots
    return series_map


def simple_fraud_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-card anomaly scoring + cross-card aggregation + explainable output.
    """
    working = df.copy()
    if working.empty:
        for column, default_value in [
            ("is_fraud", False),
            ("fraud_score", 0.0),
            ("fraud_reasons", ""),
            ("score_breakdown", "[]"),
            ("card_baseline_json", "{}"),
            ("cross_card_signals_json", "{}"),
            ("graph_features_json", "{}"),
            ("card_amount_series_json", "[]"),
        ]:
            working[column] = default_value
        return working

    if "device_id" not in working.columns:
        working["device_id"] = pd.NA
    if "ip_address" not in working.columns:
        working["ip_address"] = pd.NA

    working["device_id"] = working["device_id"].replace("", pd.NA)
    working["ip_address"] = working["ip_address"].replace("", pd.NA)
    working["timestamp_dt"] = pd.to_datetime(working["timestamp"], errors="coerce")
    if working["timestamp_dt"].isna().any():
        raise HTTPException(status_code=400, detail="Invalid timestamp format in CSV")

    working = working.sort_values(["timestamp_dt", "transaction_id"]).reset_index(drop=True)

    card_group = working.groupby("card_id", sort=False)
    working["card_tx_index"] = card_group.cumcount()
    working["card_median_before"] = card_group["amount"].transform(
        lambda s: s.shift().expanding().median()
    )
    working["card_mean_before"] = card_group["amount"].transform(
        lambda s: s.shift().expanding().mean()
    )
    working["card_std_before"] = card_group["amount"].transform(
        lambda s: s.shift().expanding().std()
    )
    working["card_median_before"] = working["card_median_before"].fillna(working["amount"])
    working["card_mean_before"] = working["card_mean_before"].fillna(working["amount"])
    working["card_std_before"] = working["card_std_before"].fillna(0.0)

    working["amount_ratio"] = (
        working["amount"] / working["card_median_before"].replace(0, pd.NA)
    ).fillna(1.0)
    working["amount_zscore"] = (
        (working["amount"] - working["card_mean_before"])
        / working["card_std_before"].replace(0, pd.NA)
    ).fillna(0.0)

    category_seen_before = working.groupby(["card_id", "merchant_category"], sort=False).cumcount()
    device_seen_before = working.groupby(["card_id", "device_id"], sort=False).cumcount()
    ip_seen_before = working.groupby(["card_id", "ip_address"], sort=False).cumcount()

    history_nonzero = working["card_tx_index"].replace(0, pd.NA)
    working["category_seen_rate"] = (category_seen_before / history_nonzero).fillna(1.0)
    working["novel_category"] = (
        (working["card_tx_index"] >= 3) & (category_seen_before == 0)
    ).astype(int)
    working["novel_device"] = (
        (working["channel"] == "online")
        & (working["card_tx_index"] >= 3)
        & working["device_id"].notna()
        & (device_seen_before == 0)
    ).astype(int)
    working["novel_ip"] = (
        (working["channel"] == "online")
        & (working["card_tx_index"] >= 3)
        & working["ip_address"].notna()
        & (ip_seen_before == 0)
    ).astype(int)

    working["foreign_country"] = (
        (working["cardholder_country"] != working["merchant_country"])
        & working["cardholder_country"].notna()
        & working["merchant_country"].notna()
    ).astype(int)
    country_seen_before = working.groupby(["card_id", "merchant_country"], sort=False).cumcount()
    working["novel_country"] = (
        (working["card_tx_index"] >= 4) & (country_seen_before == 0)
    ).astype(int)

    # Cross-card signals (global in file scope)
    working["device_card_fanout"] = (
        working.groupby("device_id")["card_id"].transform("nunique").fillna(1)
    )
    working.loc[working["device_id"].isna(), "device_card_fanout"] = 1
    working["ip_card_fanout"] = (
        working.groupby("ip_address")["card_id"].transform("nunique").fillna(1)
    )
    working.loc[working["ip_address"].isna(), "ip_card_fanout"] = 1

    merchant_tx_30m = pd.Series(1.0, index=working.index)
    merchant_unique_cards_2h = pd.Series(1.0, index=working.index)
    card_codes = pd.Categorical(working["card_id"]).codes.astype(float)
    working["card_code"] = card_codes
    for merchant, merchant_df in working.groupby("merchant_name", sort=False):
        sorted_merchant = merchant_df.sort_values("timestamp_dt")
        merchant_view = sorted_merchant.set_index("timestamp_dt")
        merchant_tx_30m.loc[sorted_merchant.index] = (
            merchant_view["transaction_id"].rolling("30min").count().values
        )
        merchant_unique_cards_2h.loc[sorted_merchant.index] = merchant_view["card_code"].rolling("2h").apply(
            lambda values: len(set(values)), raw=True
        ).values
        _ = merchant

    working["merchant_tx_30m"] = merchant_tx_30m
    working["merchant_unique_cards_2h"] = merchant_unique_cards_2h

    # Weighted explainable score
    amount_risk = ((working["amount_ratio"] - 1.6) / 5.0).clip(lower=0, upper=1)
    amount_z_risk = ((working["amount_zscore"] - 2.2) / 3.0).clip(lower=0, upper=1)
    category_risk = (1 - working["category_seen_rate"]).clip(lower=0, upper=1) * working["novel_category"]
    device_novelty_risk = working["novel_device"] * 1.0
    ip_novelty_risk = working["novel_ip"] * 1.0
    country_risk = ((0.6 * working["foreign_country"]) + (0.4 * working["novel_country"])).clip(0, 1)
    device_reuse_risk = ((working["device_card_fanout"] - 1) / 4.0).clip(lower=0, upper=1)
    ip_reuse_risk = ((working["ip_card_fanout"] - 1) / 5.0).clip(lower=0, upper=1)
    merchant_burst_risk = (
        ((working["merchant_tx_30m"] - 3) / 8.0).clip(lower=0, upper=1) * 0.4
        + ((working["merchant_unique_cards_2h"] - 2) / 6.0).clip(lower=0, upper=1) * 0.6
    ).clip(lower=0, upper=1)

    components = {
        "amount_outlier": 0.24 * amount_risk + 0.10 * amount_z_risk,
        "category_shift": 0.11 * category_risk,
        "device_shift": 0.10 * device_novelty_risk,
        "ip_shift": 0.09 * ip_novelty_risk,
        "geo_shift": 0.11 * country_risk,
        "device_reuse_cross_card": 0.11 * device_reuse_risk,
        "ip_reuse_cross_card": 0.09 * ip_reuse_risk,
        "merchant_burst_cross_card": 0.16 * merchant_burst_risk,
    }

    score = sum(components.values()).clip(lower=0, upper=1)
    high_confidence_rule = (
        (working["amount_ratio"] >= 6.0)
        | (working["ip_card_fanout"] >= 4)
        | (working["merchant_unique_cards_2h"] >= 7)
    )
    working["fraud_score"] = score.round(4)
    working["is_fraud"] = (working["fraud_score"] >= 0.55) | high_confidence_rule

    # Card-level explainability payloads
    top_categories = {
        card_id: _top_values(group["merchant_category"])
        for card_id, group in working.groupby("card_id", sort=False)
    }
    top_countries = {
        card_id: _top_values(group["merchant_country"])
        for card_id, group in working.groupby("card_id", sort=False)
    }
    top_devices = {
        card_id: _top_values(group["device_id"])
        for card_id, group in working.groupby("card_id", sort=False)
    }
    top_ips = {
        card_id: _top_values(group["ip_address"])
        for card_id, group in working.groupby("card_id", sort=False)
    }

    rows_breakdown: List[List[Dict[str, Any]]] = []
    rows_baseline: List[Dict[str, Any]] = []
    rows_cross_card: List[Dict[str, Any]] = []
    rows_graph_features: List[Dict[str, float]] = []
    rows_reason_labels: List[str] = []

    for _, row in working.iterrows():
        breakdown: List[Dict[str, Any]] = []

        def add_reason(
            code: str,
            label: str,
            detail: str,
            contribution: float,
            signal_type: str,
            value: Optional[float] = None,
            baseline: Optional[float] = None,
        ):
            if contribution < 0.03:
                return
            breakdown.append(
                {
                    "code": code,
                    "label": label,
                    "detail": detail,
                    "weight": round(float(contribution), 4),
                    "signal_type": signal_type,
                    "value": None if value is None else round(float(value), 4),
                    "baseline": None if baseline is None else round(float(baseline), 4),
                }
            )

        add_reason(
            "amount_outlier",
            "Amount anomaly",
            (
                f"Amount is {row['amount_ratio']:.2f}× this card's historical median "
                f"({row['card_median_before']:.2f})."
            ),
            components["amount_outlier"].loc[row.name],
            "per_card",
            row["amount"],
            row["card_median_before"],
        )

        add_reason(
            "category_shift",
            "Atypical category",
            (
                f"Category '{row['merchant_category']}' is uncommon for this card "
                f"(seen rate {row['category_seen_rate']:.2f})."
            ),
            components["category_shift"].loc[row.name],
            "per_card",
            row["category_seen_rate"],
            1.0,
        )

        if row["novel_device"] == 1:
            add_reason(
                "device_shift",
                "New device for card",
                f"Online transaction from unseen device '{row['device_id']}'.",
                components["device_shift"].loc[row.name],
                "per_card",
            )
        if row["novel_ip"] == 1:
            add_reason(
                "ip_shift",
                "New IP for card",
                f"Online transaction from unseen IP '{row['ip_address']}'.",
                components["ip_shift"].loc[row.name],
                "per_card",
            )
        if row["foreign_country"] == 1 or row["novel_country"] == 1:
            add_reason(
                "geo_shift",
                "Location deviation",
                (
                    f"Merchant country '{row['merchant_country']}' differs from "
                    f"cardholder '{row['cardholder_country']}' or card history."
                ),
                components["geo_shift"].loc[row.name],
                "per_card",
            )

        add_reason(
            "device_reuse_cross_card",
            "Device shared across cards",
            f"Device appears on {int(row['device_card_fanout'])} distinct cards.",
            components["device_reuse_cross_card"].loc[row.name],
            "cross_card",
            row["device_card_fanout"],
            1.0,
        )
        add_reason(
            "ip_reuse_cross_card",
            "IP shared across cards",
            f"IP appears on {int(row['ip_card_fanout'])} distinct cards.",
            components["ip_reuse_cross_card"].loc[row.name],
            "cross_card",
            row["ip_card_fanout"],
            1.0,
        )
        add_reason(
            "merchant_burst_cross_card",
            "Merchant burst across cards",
            (
                f"Merchant has {int(row['merchant_tx_30m'])} tx in 30m and "
                f"{int(row['merchant_unique_cards_2h'])} cards in 2h."
            ),
            components["merchant_burst_cross_card"].loc[row.name],
            "cross_card",
            row["merchant_unique_cards_2h"],
            1.0,
        )

        breakdown = sorted(breakdown, key=lambda signal: signal["weight"], reverse=True)
        if not breakdown and row["fraud_score"] > 0:
            breakdown = [
                {
                    "code": "baseline_risk",
                    "label": "Composite anomaly score",
                    "detail": "Combined per-card and cross-card signals elevated risk.",
                    "weight": round(float(row["fraud_score"]), 4),
                    "signal_type": "composite",
                    "value": round(float(row["fraud_score"]), 4),
                    "baseline": 0.0,
                }
            ]

        rows_breakdown.append(breakdown)
        rows_reason_labels.append("; ".join(reason["label"] for reason in breakdown))
        rows_baseline.append(
            {
                "history_count": int(row["card_tx_index"]),
                "typical_amount": round(float(row["card_median_before"]), 2),
                "amount_ratio": round(float(row["amount_ratio"]), 4),
                "amount_zscore": round(float(row["amount_zscore"]), 4),
                "usual_categories": top_categories.get(row["card_id"], []),
                "usual_countries": top_countries.get(row["card_id"], []),
                "usual_devices": top_devices.get(row["card_id"], []),
                "usual_ips": top_ips.get(row["card_id"], []),
            }
        )
        rows_cross_card.append(
            {
                "device_card_fanout": int(row["device_card_fanout"]),
                "ip_card_fanout": int(row["ip_card_fanout"]),
                "merchant_tx_30m": int(row["merchant_tx_30m"]),
                "merchant_unique_cards_2h": int(row["merchant_unique_cards_2h"]),
            }
        )
        rows_graph_features.append(
            {
                "amount_ratio": round(float(row["amount_ratio"]), 4),
                "amount_zscore": round(float(row["amount_zscore"]), 4),
                "card_tx_index": float(row["card_tx_index"]),
                "device_card_fanout": float(row["device_card_fanout"]),
                "ip_card_fanout": float(row["ip_card_fanout"]),
                "merchant_tx_30m": float(row["merchant_tx_30m"]),
                "merchant_unique_cards_2h": float(row["merchant_unique_cards_2h"]),
            }
        )

    working["fraud_reasons"] = rows_reason_labels
    working["score_breakdown"] = [json.dumps(value) for value in rows_breakdown]
    working["card_baseline_json"] = [json.dumps(value) for value in rows_baseline]
    working["cross_card_signals_json"] = [json.dumps(value) for value in rows_cross_card]
    working["graph_features_json"] = [json.dumps(value) for value in rows_graph_features]

    card_series_map = _build_card_amount_series(working)
    card_counters: Dict[str, int] = {}
    series_column: List[str] = []
    for _, row in working.iterrows():
        card_id = str(row["card_id"])
        position = card_counters.get(card_id, 0)
        series = card_series_map[card_id][position]
        series_column.append(json.dumps(series))
        card_counters[card_id] = position + 1
    working["card_amount_series_json"] = series_column

    return working.drop(columns=["timestamp_dt", "card_code"])


def _get_or_compute_analysis(file_hash: str) -> pd.DataFrame:
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    if file_hash not in analysis_cache:
        analysis_cache[file_hash] = simple_fraud_detection(uploaded_files[file_hash])
    return analysis_cache[file_hash]


def _review_records_for_export(file_hash: str) -> pd.DataFrame:
    records = review_log.get(file_hash, {})
    if not records:
        return pd.DataFrame(
            columns=[
                "transaction_id",
                "review_decision",
                "reviewer_notes",
                "reviewed_at",
            ]
        )

    return pd.DataFrame(
        [
            {
                "transaction_id": transaction_id,
                "review_decision": record.get("action", ""),
                "reviewer_notes": record.get("reviewer_notes", ""),
                "reviewed_at": record.get("reviewed_at", ""),
            }
            for transaction_id, record in records.items()
        ]
    )


def _analysis_with_review_columns(file_hash: str) -> pd.DataFrame:
    analyzed_df = _get_or_compute_analysis(file_hash).copy()
    review_df = _review_records_for_export(file_hash)

    if review_df.empty:
        analyzed_df["review_decision"] = ""
        analyzed_df["reviewer_notes"] = ""
        analyzed_df["reviewed_at"] = ""
        return analyzed_df

    return analyzed_df.merge(review_df, on="transaction_id", how="left").fillna(
        {
            "review_decision": "",
            "reviewer_notes": "",
            "reviewed_at": "",
        }
    )


def _to_fraud_analysis(row: pd.Series) -> FraudAnalysis:
    parsed_breakdown = json.loads(row.get("score_breakdown", "[]") or "[]")
    parsed_baseline = json.loads(row.get("card_baseline_json", "{}") or "{}")
    parsed_cross_card = json.loads(row.get("cross_card_signals_json", "{}") or "{}")
    parsed_graph_features = json.loads(row.get("graph_features_json", "{}") or "{}")
    parsed_card_series = json.loads(row.get("card_amount_series_json", "[]") or "[]")
    reasons = [item["label"] for item in parsed_breakdown if "label" in item]

    return FraudAnalysis(
        transaction_id=str(row["transaction_id"]),
        timestamp=str(row["timestamp"]),
        card_id=str(row["card_id"]),
        amount=float(row["amount"]),
        merchant_name=str(row["merchant_name"]),
        merchant_category=str(row["merchant_category"]),
        channel=str(row["channel"]),
        is_fraud=bool(row["is_fraud"]),
        fraud_score=float(row["fraud_score"]),
        reasons=reasons,
        score_breakdown=parsed_breakdown,
        card_baseline=parsed_baseline,
        cross_card_signals=parsed_cross_card,
        graph_features=parsed_graph_features,
        card_amount_series=parsed_card_series,
    )

@app.get("/analysis/all/{file_hash}")
async def get_all_analysis(file_hash: str):
    analyzed_df = _get_or_compute_analysis(file_hash)
    return [_to_fraud_analysis(row) for _, row in analyzed_df.iterrows()]

def row_to_analyzed_transaction(row: Any) -> AnalyzedTransaction:
    reasons = []
    if row['fraud_reasons']:
        reasons = [r.strip() for r in row['fraud_reasons'].split(';') if r.strip()]

    return AnalyzedTransaction(
        transaction_id=row['transaction_id'],
        timestamp=row['timestamp'],
        card_id=row['card_id'],
        amount=float(row['amount']),
        merchant_name=row['merchant_name'],
        merchant_category=row['merchant_category'],
        channel=row['channel'],
        cardholder_country=row['cardholder_country'],
        merchant_country=row['merchant_country'],
        device_id=None if pd.isna(row.get('device_id')) else row.get('device_id'),
        ip_address=None if pd.isna(row.get('ip_address')) else row.get('ip_address'),
        is_fraud=bool(row['is_fraud']),
        fraud_score=float(row['fraud_score']),
        fraud_reasons=reasons,
        reasons=reasons,
    )

@app.get("/analysis/user/{file_hash}/{card_id}", response_model=List[AnalyzedTransaction])
async def get_user_analysis(file_hash: str, card_id: str):
    analyzed_df = _get_or_compute_analysis(file_hash)
    user_df = analyzed_df[analyzed_df["card_id"] == card_id]
    if user_df.empty:
        raise HTTPException(status_code=404, detail="No transactions found for this card")
    return [
        row_to_analyzed_transaction(row)
        for _, row in user_df.sort_values('timestamp').iterrows()
    ]

@app.get("/analysis/ip/{file_hash}/{ip_address}")
async def get_ip_analysis(file_hash: str, ip_address: str):
    analyzed_df = _get_or_compute_analysis(file_hash)
    ip_df = analyzed_df[analyzed_df["ip_address"] == ip_address]
    if ip_df.empty:
        raise HTTPException(status_code=404, detail="No transactions found for this IP")
    return [_to_fraud_analysis(row) for _, row in ip_df.iterrows()]

@app.get("/review/{file_hash}/audit", response_model=List[ReviewLogEntry])
async def get_review_audit(file_hash: str):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")

    records = review_log.get(file_hash, {})
    return [
        ReviewLogEntry(
            transaction_id=transaction_id,
            action=str(record["action"]),
            reviewer_notes=record.get("reviewer_notes"),
            reviewed_at=str(record["reviewed_at"]),
        )
        for transaction_id, record in sorted(
            records.items(),
            key=lambda item: item[1].get("reviewed_at", ""),
            reverse=True,
        )
    ]


@app.post("/review/{file_hash}/{transaction_id}/{action}")
async def review_transaction(file_hash: str, transaction_id: str, action: str, review_action: ReviewAction):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    if action not in ["approve", "dismiss", "escalate", "pending"]:
        raise HTTPException(status_code=400, detail="Invalid action")

    if review_action.action != action:
        raise HTTPException(status_code=400, detail="Action body does not match URL")
    
    df = uploaded_files[file_hash]
    transaction_mask = df['transaction_id'] == transaction_id
    
    if not transaction_mask.any():
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    if action == "pending":
        review_log[file_hash].pop(transaction_id, None)
        reviewed_at = datetime.now(timezone.utc).isoformat()
    else:
        review_log[file_hash][transaction_id] = {
            "action": action,
            "reviewer_notes": review_action.reviewer_notes,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        reviewed_at = str(review_log[file_hash][transaction_id]["reviewed_at"])
    
    return {
        "message": f"Transaction {transaction_id} set to {action}",
        "transaction_id": transaction_id,
        "action": action,
        "reviewer_notes": review_action.reviewer_notes,
        "reviewed_at": reviewed_at,
    }

@app.get("/review-log/{file_hash}", response_model=List[ReviewLogEntry])
async def get_review_log(file_hash: str):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")

    file_review_log = review_log.get(file_hash, {})
    entries: List[ReviewLogEntry] = []
    for transaction_id, payload in file_review_log.items():
        entries.append(
            ReviewLogEntry(
                transaction_id=transaction_id,
                action=str(payload.get("action", "")),
                reviewer_notes=payload.get("reviewer_notes"),
                reviewed_at=str(payload.get("reviewed_at", "")),
            )
        )

    entries.sort(key=lambda item: item.reviewed_at, reverse=True)
    return entries

@app.get("/export/{file_hash}")
async def export_analysis(file_hash: str):
    analyzed_df = _analysis_with_review_columns(file_hash)
    
    # Create CSV in memory
    output = io.StringIO()
    analyzed_df.to_csv(output, index=False)
    output.seek(0)
    
    # Return as streaming response
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=analyzed_transactions_{file_hash[:8]}.csv"}
    )
