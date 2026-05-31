# Fraud Detection Backend

This is the FastAPI backend for the fraud detection application.

## Endpoints

### File Upload
- **POST /upload** - Upload a CSV file and receive a hash identifier
  - Returns: `{file_hash: str, message: str}`

### Analysis Endpoints
- **GET /analysis/all/{file_hash}` - Get fraud analysis for all transactions
  - Returns: List of enriched `[FraudAnalysis]` objects with explainable score payloads
- **GET /analysis/user/{file_hash}/{card_id}` - Get fraud analysis for specific card/user
  - Returns: Per-card slice of globally computed analysis (keeps cross-card signals intact)
- **GET /analysis/ip/{file_hash}/{ip_address}` - Get fraud analysis for specific IP address
  - Returns: IP-specific slice of globally computed analysis

### Review Endpoints
- **POST /review/{file_hash}/{transaction_id}/{action}` - Review a transaction
  - Actions: `approve`, `dismiss`, `escalate`, `pending`
  - Body: `{action: str, reviewer_notes?: str}`
  - Returns: Confirmation message
- **GET /review/{file_hash}/audit` - Get review actions for the current upload
  - Returns: List of transaction decisions with timestamps

### Export Endpoint
- **GET /export/{file_hash}` - Export analyzed transactions as CSV
  - Returns: CSV file with original data plus fraud analysis columns
  - Columns added: `is_fraud`, `fraud_score`, `fraud_reasons`, review metadata

## Pydantic Models

### UploadResponse
```python
class UploadResponse(BaseModel):
    file_hash: str
    message: str
```

### FraudAnalysis
```python
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
    score_breakdown: List[Dict[str, Any]]         # weighted explainability objects
    card_baseline: Dict[str, Any]                 # per-card baseline snapshot
    cross_card_signals: Dict[str, Any]            # fanout/burst metrics
    graph_features: Dict[str, float]              # numeric frontend graph inputs
    card_amount_series: List[Dict[str, Any]]      # per-card historical points
```

### ReviewAction
```python
class ReviewAction(BaseModel):
    action: str  # approve, dismiss, escalate
    reviewer_notes: Optional[str] = None
```

## Fraud Detection Logic

The detector combines:
1. **Per-card anomaly scoring**
   - Amount deviation vs card historical median and z-score
   - Novel category/device/IP/country for that card
2. **Cross-card aggregation**
   - Device reused across multiple cards
   - IP reused across multiple cards
   - Merchant burst behavior (velocity + unique cards in short windows)
3. **Explainable weighted score**
   - Final score in `[0,1]`
   - Reason-level `weight`, `signal_type`, `value`, and `baseline` for auditability
   - `fraud_reasons` remains available for backward compatibility

## Installation

```bash
uv sync
```

## Running the Server

```bash
uvicorn main:app --reload
```

## Testing

Run the test suite:
```bash
uv run --extra test python -m pytest -q
```

## Exporting the challenge CSV

From the repo root:

```bash
make export
```
