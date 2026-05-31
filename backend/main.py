import hashlib
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from fraud_scorer import simple_fraud_detection
from ml_fraud_scorer import (
    DEFAULT_MODEL_PATH,
    ModelNotAvailableError,
    is_model_available,
    ml_fraud_detection,
)

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


class AnalysisSummaryResponse(BaseModel):
    total_transactions: int
    flagged_count: int
    model_flagged_count: int = 0
    ml_model_available: bool


class QueueTransactionItem(TransactionBase):
    is_fraud: bool
    fraud_score: float
    fraud_reasons: List[str] = Field(default_factory=list)
    review_decision: str = ""
    reviewer_notes: Optional[str] = None
    reviewed_at: Optional[str] = None
    card_baseline: Dict[str, Any] = Field(default_factory=dict)


class QueuePageResponse(BaseModel):
    items: List[QueueTransactionItem]
    total: int
    offset: int
    limit: Optional[int] = None


class ScorerDetail(BaseModel):
    fraud_score: float
    is_fraud: bool
    reasons: List[str] = Field(default_factory=list)
    score_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    card_baseline: Dict[str, Any] = Field(default_factory=dict)
    cross_card_signals: Dict[str, Any] = Field(default_factory=dict)
    graph_features: Dict[str, float] = Field(default_factory=dict)


class TransactionDetailResponse(TransactionBase):
    heuristic: ScorerDetail
    model: Optional[ScorerDetail] = None
    review_decision: str = ""
    reviewer_notes: Optional[str] = None
    reviewed_at: Optional[str] = None


class RelatedTransactionsResponse(BaseModel):
    items: List[QueueTransactionItem]


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.get("/scoring/status")
def scoring_status():
    return {
        "heuristic": True,
        "ml_model_path": str(DEFAULT_MODEL_PATH),
        "ml_model_available": is_model_available(),
    }


def _analysis_cache_key(file_hash: str, use_model: bool) -> str:
    return f"{file_hash}:{'ml' if use_model else 'heuristic'}"


def _clear_analysis_cache(file_hash: str) -> None:
    prefix = f"{file_hash}:"
    for key in list(analysis_cache):
        if key.startswith(prefix):
            analysis_cache.pop(key, None)


def _score_transactions(df: pd.DataFrame, use_model: bool) -> pd.DataFrame:
    if use_model:
        try:
            return ml_fraud_detection(df)
        except ModelNotAvailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return simple_fraud_detection(df)

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
        _clear_analysis_cache(file_hash)
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

def _get_or_compute_analysis(file_hash: str, use_model: bool = False) -> pd.DataFrame:
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    cache_key = _analysis_cache_key(file_hash, use_model)
    if cache_key not in analysis_cache:
        try:
            analysis_cache[cache_key] = _score_transactions(
                uploaded_files[file_hash],
                use_model=use_model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return analysis_cache[cache_key]


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


def _analysis_with_review_columns(file_hash: str, use_model: bool = False) -> pd.DataFrame:
    analyzed_df = _get_or_compute_analysis(file_hash, use_model=use_model).copy()
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


def _parse_fraud_reasons(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _parse_json_field(value: Any, fallback: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value) or json.dumps(fallback))
    except (json.JSONDecodeError, TypeError):
        return fallback


def _optional_text(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _queue_dataframe(
    file_hash: str,
    use_model: bool = False,
    flagged_only: bool = True,
) -> pd.DataFrame:
    analyzed_df = _analysis_with_review_columns(file_hash, use_model=use_model)
    if flagged_only:
        analyzed_df = analyzed_df[
            analyzed_df["is_fraud"] | (analyzed_df["fraud_score"] > 0)
        ]
    return analyzed_df.sort_values("fraud_score", ascending=False)


def _row_to_queue_item(row: pd.Series) -> QueueTransactionItem:
    return QueueTransactionItem(
        transaction_id=str(row["transaction_id"]),
        timestamp=str(row["timestamp"]),
        card_id=str(row["card_id"]),
        amount=float(row["amount"]),
        merchant_name=str(row["merchant_name"]),
        merchant_category=str(row["merchant_category"]),
        channel=str(row["channel"]),
        cardholder_country=str(row["cardholder_country"]),
        merchant_country=str(row["merchant_country"]),
        device_id=_optional_text(row.get("device_id")),
        ip_address=_optional_text(row.get("ip_address")),
        is_fraud=bool(row["is_fraud"]),
        fraud_score=float(row["fraud_score"]),
        fraud_reasons=_parse_fraud_reasons(row.get("fraud_reasons")),
        review_decision=str(row.get("review_decision") or ""),
        reviewer_notes=_optional_text(row.get("reviewer_notes")),
        reviewed_at=_optional_text(row.get("reviewed_at")),
        card_baseline=_parse_json_field(row.get("card_baseline_json"), {}),
    )


def _row_to_scorer_detail(row: pd.Series) -> ScorerDetail:
    parsed_breakdown = _parse_json_field(row.get("score_breakdown"), [])
    if not isinstance(parsed_breakdown, list):
        parsed_breakdown = []
    reasons = [
        str(item["label"])
        for item in parsed_breakdown
        if isinstance(item, dict) and item.get("label")
    ]
    if not reasons:
        reasons = _parse_fraud_reasons(row.get("fraud_reasons"))

    graph_features = _parse_json_field(row.get("graph_features_json"), {})
    if not isinstance(graph_features, dict):
        graph_features = {}

    return ScorerDetail(
        fraud_score=float(row["fraud_score"]),
        is_fraud=bool(row["is_fraud"]),
        reasons=reasons,
        score_breakdown=[
            item for item in parsed_breakdown if isinstance(item, dict)
        ],
        card_baseline=_parse_json_field(row.get("card_baseline_json"), {}),
        cross_card_signals=_parse_json_field(row.get("cross_card_signals_json"), {}),
        graph_features={
            str(key): float(value)
            for key, value in graph_features.items()
            if isinstance(value, (int, float))
        },
    )


def _lookup_analysis_row(
    file_hash: str,
    transaction_id: str,
    use_model: bool,
) -> pd.Series:
    analyzed_df = _analysis_with_review_columns(file_hash, use_model=use_model)
    matches = analyzed_df[analyzed_df["transaction_id"] == transaction_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return matches.iloc[0]


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

@app.get("/analysis/summary/{file_hash}", response_model=AnalysisSummaryResponse)
async def get_analysis_summary(file_hash: str):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")

    analyzed_df = _analysis_with_review_columns(file_hash, use_model=False)
    flagged_count = int(
        (analyzed_df["is_fraud"] | (analyzed_df["fraud_score"] > 0)).sum()
    )
    model_flagged_count = 0
    if is_model_available():
        model_df = _analysis_with_review_columns(file_hash, use_model=True)
        model_flagged_count = int(
            (model_df["is_fraud"] | (model_df["fraud_score"] > 0)).sum()
        )

    return AnalysisSummaryResponse(
        total_transactions=len(uploaded_files[file_hash]),
        flagged_count=flagged_count,
        model_flagged_count=model_flagged_count,
        ml_model_available=is_model_available(),
    )


@app.get("/analysis/queue/{file_hash}", response_model=QueuePageResponse)
async def get_analysis_queue(
    file_hash: str,
    use_model: bool = False,
    flagged_only: bool = True,
    limit: Optional[int] = None,
    offset: int = 0,
    transaction_id: Optional[str] = None,
):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be non-negative")
    if limit is not None and limit < 1:
        raise HTTPException(status_code=400, detail="Limit must be positive")

    queue_df = _queue_dataframe(file_hash, use_model=use_model, flagged_only=flagged_only)

    if transaction_id:
        analyzed_df = _analysis_with_review_columns(file_hash, use_model=use_model)
        matches = analyzed_df[analyzed_df["transaction_id"] == transaction_id]
        if matches.empty:
            raise HTTPException(status_code=404, detail="Transaction not found")
        items = [_row_to_queue_item(matches.iloc[0])]
        return QueuePageResponse(
            items=items,
            total=len(queue_df),
            offset=0,
            limit=1,
        )

    total = len(queue_df)
    if limit is None:
        page_df = queue_df.iloc[offset:]
    else:
        page_df = queue_df.iloc[offset : offset + limit]

    return QueuePageResponse(
        items=[_row_to_queue_item(row) for _, row in page_df.iterrows()],
        total=total,
        offset=offset,
        limit=limit,
    )


@app.get(
    "/analysis/transaction/{file_hash}/{transaction_id}",
    response_model=TransactionDetailResponse,
)
async def get_transaction_detail(file_hash: str, transaction_id: str):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")

    heuristic_row = _lookup_analysis_row(file_hash, transaction_id, use_model=False)
    model_detail = None
    if is_model_available():
        try:
            model_row = _lookup_analysis_row(file_hash, transaction_id, use_model=True)
            model_detail = _row_to_scorer_detail(model_row)
        except HTTPException:
            model_detail = None

    return TransactionDetailResponse(
        transaction_id=str(heuristic_row["transaction_id"]),
        timestamp=str(heuristic_row["timestamp"]),
        card_id=str(heuristic_row["card_id"]),
        amount=float(heuristic_row["amount"]),
        merchant_name=str(heuristic_row["merchant_name"]),
        merchant_category=str(heuristic_row["merchant_category"]),
        channel=str(heuristic_row["channel"]),
        cardholder_country=str(heuristic_row["cardholder_country"]),
        merchant_country=str(heuristic_row["merchant_country"]),
        device_id=_optional_text(heuristic_row.get("device_id")),
        ip_address=_optional_text(heuristic_row.get("ip_address")),
        heuristic=_row_to_scorer_detail(heuristic_row),
        model=model_detail,
        review_decision=str(heuristic_row.get("review_decision") or ""),
        reviewer_notes=_optional_text(heuristic_row.get("reviewer_notes")),
        reviewed_at=_optional_text(heuristic_row.get("reviewed_at")),
    )


@app.get(
    "/analysis/related/{file_hash}/{transaction_id}",
    response_model=RelatedTransactionsResponse,
)
async def get_related_transactions(
    file_hash: str,
    transaction_id: str,
    use_model: bool = False,
):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")

    analyzed_df = _analysis_with_review_columns(file_hash, use_model=use_model)
    active_rows = analyzed_df[analyzed_df["transaction_id"] == transaction_id]
    if active_rows.empty:
        raise HTTPException(status_code=404, detail="Transaction not found")

    active = active_rows.iloc[0]
    active_card = active["card_id"]
    active_device = active.get("device_id")
    active_ip = active.get("ip_address")

    mask = analyzed_df["card_id"] == active_card
    if _optional_text(active_device):
        mask = mask | (analyzed_df["device_id"] == active_device)
    if _optional_text(active_ip):
        mask = mask | (analyzed_df["ip_address"] == active_ip)

    related_df = analyzed_df[mask].sort_values("timestamp")
    return RelatedTransactionsResponse(
        items=[_row_to_queue_item(row) for _, row in related_df.iterrows()],
    )


@app.get("/analysis/all/{file_hash}")
async def get_all_analysis(file_hash: str, use_model: bool = False):
    analyzed_df = _get_or_compute_analysis(file_hash, use_model=use_model)
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
async def get_user_analysis(file_hash: str, card_id: str, use_model: bool = False):
    analyzed_df = _get_or_compute_analysis(file_hash, use_model=use_model)
    user_df = analyzed_df[analyzed_df["card_id"] == card_id]
    if user_df.empty:
        raise HTTPException(status_code=404, detail="No transactions found for this card")
    return [
        row_to_analyzed_transaction(row)
        for _, row in user_df.sort_values('timestamp').iterrows()
    ]

@app.get("/analysis/ip/{file_hash}/{ip_address}")
async def get_ip_analysis(file_hash: str, ip_address: str, use_model: bool = False):
    analyzed_df = _get_or_compute_analysis(file_hash, use_model=use_model)
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
async def export_analysis(file_hash: str, use_model: bool = False):
    analyzed_df = _analysis_with_review_columns(file_hash, use_model=use_model)
    
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
