from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import hashlib
import io
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json

app = FastAPI()

# In-memory storage for uploaded files
uploaded_files: Dict[str, pd.DataFrame] = {}

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
    is_fraud: bool
    fraud_score: float
    reasons: List[str]

class ReviewAction(BaseModel):
    action: str  # approve, dismiss, escalate
    reviewer_notes: Optional[str] = None

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
        
        # Store DataFrame
        uploaded_files[file_hash] = df
        
        return UploadResponse(
            file_hash=file_hash,
            message="File uploaded successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing CSV: {str(e)}"
        )

def simple_fraud_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple fraud detection logic - to be expanded
    For now, flags transactions with:
    - Amount > 3x the card's median amount
    - Foreign transactions (cardholder_country != merchant_country)
    - Online transactions with missing device_id or ip_address
    """
    df = df.copy()
    
    # Initialize fraud columns
    df['is_fraud'] = False
    df['fraud_score'] = 0.0
    df['fraud_reasons'] = ''
    
    # Calculate card medians for amount comparison
    card_medians = df.groupby('card_id')['amount'].transform('median')
    
    # Flag 1: Amount significantly higher than card's typical amount
    amount_flag = (df['amount'] > 3 * card_medians) & (card_medians > 0)
    df.loc[amount_flag, 'is_fraud'] = True
    df.loc[amount_flag, 'fraud_score'] = 0.8
    df.loc[amount_flag, 'fraud_reasons'] = df.loc[amount_flag, 'fraud_reasons'] + 'High amount for card; '
    
    # Flag 2: Foreign transactions
    foreign_flag = (df['cardholder_country'] != df['merchant_country']) & \
                   (df['cardholder_country'].notna()) & (df['merchant_country'].notna())
    df.loc[foreign_flag, 'is_fraud'] = True
    df.loc[foreign_flag, 'fraud_score'] = df.loc[foreign_flag, 'fraud_score'] + 0.3
    df.loc[foreign_flag, 'fraud_reasons'] = df.loc[foreign_flag, 'fraud_reasons'] + 'Foreign transaction; '
    
    # Flag 3: Online transactions missing device or IP
    online_mask = df['channel'] == 'online'
    missing_device_ip = online_mask & (df['device_id'].isna() | df['ip_address'].isna())
    df.loc[missing_device_ip, 'is_fraud'] = True
    df.loc[missing_device_ip, 'fraud_score'] = df.loc[missing_device_ip, 'fraud_score'] + 0.4
    df.loc[missing_device_ip, 'fraud_reasons'] = df.loc[missing_device_ip, 'fraud_reasons'] + 'Missing device/IP for online; '
    
    # Cap fraud score at 1.0
    df['fraud_score'] = df['fraud_score'].clip(0, 1.0)
    
    # Clean up reasons string
    df['fraud_reasons'] = df['fraud_reasons'].str.rstrip('; ')
    
    return df

@app.get("/analysis/all/{file_hash}")
async def get_all_analysis(file_hash: str):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    df = uploaded_files[file_hash]
    analyzed_df = simple_fraud_detection(df)
    
    # Convert to list of FraudAnalysis objects
    results = []
    for _, row in analyzed_df.iterrows():
        reasons = []
        if row['fraud_reasons']:
            reasons = [r.strip() for r in row['fraud_reasons'].split(';') if r.strip()]
        
        results.append(FraudAnalysis(
            transaction_id=row['transaction_id'],
            is_fraud=bool(row['is_fraud']),
            fraud_score=float(row['fraud_score']),
            reasons=reasons
        ))
    
    return results

@app.get("/analysis/user/{file_hash}/{card_id}")
async def get_user_analysis(file_hash: str, card_id: str):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    df = uploaded_files[file_hash]
    user_df = df[df['card_id'] == card_id]
    
    if user_df.empty:
        raise HTTPException(status_code=404, detail="No transactions found for this card")
    
    analyzed_df = simple_fraud_detection(user_df)
    
    # Convert to list of FraudAnalysis objects
    results = []
    for _, row in analyzed_df.iterrows():
        reasons = []
        if row['fraud_reasons']:
            reasons = [r.strip() for r in row['fraud_reasons'].split(';') if r.strip()]
        
        results.append(FraudAnalysis(
            transaction_id=row['transaction_id'],
            is_fraud=bool(row['is_fraud']),
            fraud_score=float(row['fraud_score']),
            reasons=reasons
        ))
    
    return results

@app.get("/analysis/ip/{file_hash}/{ip_address}")
async def get_ip_analysis(file_hash: str, ip_address: str):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    df = uploaded_files[file_hash]
    ip_df = df[df['ip_address'] == ip_address]
    
    if ip_df.empty:
        raise HTTPException(status_code=404, detail="No transactions found for this IP")
    
    analyzed_df = simple_fraud_detection(ip_df)
    
    # Convert to list of FraudAnalysis objects
    results = []
    for _, row in analyzed_df.iterrows():
        reasons = []
        if row['fraud_reasons']:
            reasons = [r.strip() for r in row['fraud_reasons'].split(';') if r.strip()]
        
        results.append(FraudAnalysis(
            transaction_id=row['transaction_id'],
            is_fraud=bool(row['is_fraud']),
            fraud_score=float(row['fraud_score']),
            reasons=reasons
        ))
    
    return results

@app.post("/review/{file_hash}/{transaction_id}/{action}")
async def review_transaction(file_hash: str, transaction_id: str, action: str, review_action: ReviewAction):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    if action not in ["approve", "dismiss", "escalate"]:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    df = uploaded_files[file_hash]
    transaction_mask = df['transaction_id'] == transaction_id
    
    if not transaction_mask.any():
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    # In a real application, we would update a database or separate review log
    # For now, we'll just return a confirmation
    # We could add review columns to the dataframe here if needed
    
    return {
        "message": f"Transaction {transaction_id} {action}d successfully",
        "transaction_id": transaction_id,
        "action": action,
        "reviewer_notes": review_action.reviewer_notes
    }

@app.get("/export/{file_hash}")
async def export_analysis(file_hash: str):
    if file_hash not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    df = uploaded_files[file_hash]
    analyzed_df = simple_fraud_detection(df)
    
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
