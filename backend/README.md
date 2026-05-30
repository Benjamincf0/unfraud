# Fraud Detection Backend

This is the FastAPI backend for the fraud detection application.

## Endpoints

### File Upload
- **POST /upload** - Upload a CSV file and receive a hash identifier
  - Returns: `{file_hash: str, message: str}`

### Analysis Endpoints
- **GET /analysis/all/{file_hash}` - Get fraud analysis for all transactions
  - Returns: List of `[FraudAnalysis]` objects
- **GET /analysis/user/{file_hash}/{card_id}` - Get fraud analysis for specific card/user
  - Returns: List of `[FraudAnalysis]` objects
- **GET /analysis/ip/{file_hash}/{ip_address}` - Get fraud analysis for specific IP address
  - Returns: List of `[FraudAnalysis]` objects

### Review Endpoints
- **POST /review/{file_hash}/{transaction_id}/{action}` - Review a transaction
  - Actions: `approve`, `dismiss`, `escalate`
  - Body: `{action: str, reviewer_notes?: str}`
  - Returns: Confirmation message

### Export Endpoint
- **GET /export/{file_hash}` - Export analyzed transactions as CSV
  - Returns: CSV file with original data plus fraud analysis columns
  - Columns added: `is_fraud`, `fraud_score`, `fraud_reasons`

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
    is_fraud: bool
    fraud_score: float
    reasons: List[str]
```

### ReviewAction
```python
class ReviewAction(BaseModel):
    action: str  # approve, dismiss, escalate
    reviewer_notes: Optional[str] = None
```

## Fraud Detection Logic

The current implementation uses a simple rule-based approach:
1. **High amount for card**: Transactions > 3x the card's median amount
2. **Foreign transaction**: Cardholder country ≠ Merchant country
3. **Missing device/IP for online**: Online transactions missing device_id or ip_address

Each flag contributes to a fraud score (capped at 1.0) and provides human-readable reasons.

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
python -c "
import sys
sys.path.insert(0, '.')
from main import app
from fastapi.testclient import TestClient
client = TestClient(app)
# Add your test code here
"
```