import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import numpy as np
import os
import uuid
from collections import defaultdict
import warnings


MODEL_PATH = "fraud_model.pkl"
CSV_FOR_ANALYSIS = "transactions.csv" 


user_transaction_history = defaultdict(list)


xgb_model = None
MODEL_EXPECTED_FEATURES = [] 
try:
    xgb_model = joblib.load(MODEL_PATH)
    if hasattr(xgb_model, 'feature_names_in_'):
        MODEL_EXPECTED_FEATURES = xgb_model.feature_names_in_.tolist()
    elif hasattr(xgb_model, 'get_booster'):
        MODEL_EXPECTED_FEATURES = xgb_model.get_booster().feature_names
    else:
        warnings.warn(f"Warning: Could not extract feature names directly from model '{MODEL_PATH}'. Will attempt to infer during preprocessing. This is less robust.")

    print(f"Model '{MODEL_PATH}' loaded successfully.")
    print(f"Model expects {len(MODEL_EXPECTED_FEATURES)} features.")
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please ensure 'fraud_model.pkl' is in the same directory.")
except Exception as e:
    raise RuntimeError(f"Error loading model from {MODEL_PATH}: {e}")


HARDCODED_CATEGORIES = {
    'account_id': [],
    'method': ['UPI', 'CARD', 'NEFT', 'IMPS'],
    'location': [
        'Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune',
        'Ahmedabad', 'Surat', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Visakhapatnam',
        'Indore', 'Thane', 'Bhopal', 'Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana',
        'Agra', 'Nashik', 'Faridabad', 'Meerut', 'Rajkot', 'Varanasi', 'Srinagar',
        'Aurangabad', 'Dhanbad', 'Amritsar', 'Prayagraj', 'Ranchi', 'Howrah',
        'Coimbatore', 'Jabalpur', 'Gwalior', 'Vijayawada', 'Jodhpur', 'Madurai',
        'Raipur', 'Kota', 'Guwahati', 'Chandigarh', 'Solapur', 'Hubli',
        'Tiruchirappalli', 'Bareilly', 'Moradabad', 'Mysuru', 'Gurugram', 'Aligarh',
        'Jalandhar', 'Bhubaneshwar', 'Salem', 'Warangal', 'Guntur', 'Bhiwandi',
        'Saharanpur', 'Gorakhpur', 'Bikaner', 'Amravati', 'Noida', 'Jamshedpur',
        'Bhilai', 'Cuttack', 'Firozabad', 'Kochi', 'Nellore', 'Bhavnagar',
        'Dehradun', 'Durgapur', 'Asansol', 'Nanded', 'Kolhapur', 'Ajmer',
        'Gulbarga', 'Jamnagar', 'Ujjain', 'Loni', 'Jhansi', 'Tirunelveli',
        'Jammu', 'Kashmir', 'Rourkela', 'Chandigarh', 'Assam', 'Noida'
    ],
    'device_id': [
        'DEVICE-H8', 'DEVICE-W2', 'DEVICE-K3', 'DEVICE-Z1', 'DEVICE-AD6', 'DEVICE-T6',
        'DEVICE-F7', 'DEVICE-8890', 'DEVICE-G6', 'DEVICE-8810', 'DEVICE-H1', 'DEVICE-A2',
        'DEVICE-F4', 'DEVICE-D2', 'DEVICE-D5'
    ],
    'status': ["COMPLETED", "PENDING", "FAILED", "REVERSED"]
}

global_avg_amount = 0.0
global_std_amount = 1.0
global_min_amount = 0.0
global_max_amount = 0.0

ALL_POSSIBLE_CATEGORIES = HARDCODED_CATEGORIES.copy()


try:
    df_original = pd.read_csv(CSV_FOR_ANALYSIS)
    
   
    if 'amount' in df_original.columns:
        global_avg_amount = df_original['amount'].mean()
        global_std_amount = df_original['amount'].std()
        global_min_amount = df_original['amount'].min()
        global_max_amount = df_original['amount'].max()
        print(f"Global amount stats: Avg={global_avg_amount:.2f}, Std={global_std_amount:.2f}, Min={global_min_amount:.2f}, Max={global_max_amount:.2f}")
    else:
        warnings.warn("Column 'amount' not found in 'transactions.csv'. Using default global amount stats.")

    for col in ['account_id', 'method', 'location', 'device_id', 'status']:
        if col in df_original.columns:
            unique_values = df_original[col].dropna().unique().tolist()
        
            ALL_POSSIBLE_CATEGORIES[col] = sorted(list(set(ALL_POSSIBLE_CATEGORIES.get(col, []) + unique_values)))
        else:
            warnings.warn(f"Column '{col}' not found in '{CSV_FOR_ANALYSIS}'. Using hardcoded categories for it.")

    print(f"Categorical features derived from '{CSV_FOR_ANALYSIS}': {list(ALL_POSSIBLE_CATEGORIES.keys())}")
    print(f"Updated categories (first 5 for each):")
    for k, v in ALL_POSSIBLE_CATEGORIES.items():
        print(f"  {k}: {v[:5]}...") # Print only a few for brevity

except FileNotFoundError:
    warnings.warn(f"Original CSV file '{CSV_FOR_ANALYSIS}' not found. Using hardcoded categories and default amount stats.")
except KeyError as e:
    warnings.warn(f"Error reading '{CSV_FOR_ANALYSIS}': {e}. Using hardcoded categories and default amount stats.")
except Exception as e:
    warnings.warn(f"An unexpected error occurred while processing '{CSV_FOR_ANALYSIS}': {e}. Using hardcoded categories and default amount stats.")


app = FastAPI(
    title="Fraud Detection ML API",
    description="API for real-time fraud transaction prediction using a pre-trained XGBoost model.",
    version="1.0.0",
)

class TransactionIn(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this prediction request (generated by client for correlation).")
    account_id: str = Field(..., description="Unique identifier for the account (UUID).")
    amount: float = Field(..., gt=0, description="Amount of the transaction (must be positive).")
    method: str = Field(..., description="Payment method (e.g., UPI, CARD, NEFT, IMPS).")
    location: str = Field(..., description="Location of the transaction.")
    device_id: str = Field(..., description="Device ID used for the transaction.")
    timestamp: datetime = Field(..., description="Transaction timestamp in ISO 8601 format.")
    status: str = Field(..., description="Status of the transaction (e.g., COMPLETED, PENDING, FAILED, REVERSED).")
    is_flagged: bool = Field(False, description="Whether the transaction was flagged for manual review during initial processing.") # Added is_flagged

class PredictionOut(BaseModel):
    request_id: str
    is_fraud_predicted: bool
    fraud_probability: float
    is_flagged_simulated: bool
    message: str = "Transaction processed successfully."


def preprocess_features(transaction: TransactionIn):
    """
    Generates and preprocesses features for the XGBoost model.
    Includes current transaction features and historical features.
    """
    global MODEL_EXPECTED_FEATURES 
    
    account_id = transaction.account_id
    transaction_dt = transaction.timestamp

    if account_id not in user_transaction_history:
        user_transaction_history[account_id] = []
    
    seven_days_before_current_txn = transaction_dt - timedelta(days=7)
    recent_history = [
        t for t in user_transaction_history[account_id]
        if t['timestamp'] >= seven_days_before_current_txn
    ]
    
    num_txns_7d = len(recent_history)
    total_amount_7d = sum(t['amount'] for t in recent_history)
    
    avg_txn_amount_7d = total_amount_7d / num_txns_7d if num_txns_7d > 0 else 0
    txn_per_day_7d = num_txns_7d / 7.0 if num_txns_7d > 0 else 0.0

    time_since_last_txn = 0.0
    if recent_history:
        last_txn_timestamp_in_history = max(t['timestamp'] for t in recent_history)
        time_since_last_txn = (transaction_dt - last_txn_timestamp_in_history).total_seconds()
        if time_since_last_txn < 0:
             time_since_last_txn = 0.0

    user_avg_amount = np.mean([t['amount'] for t in recent_history]) if recent_history else global_avg_amount
    user_std_amount = np.std([t['amount'] for t in recent_history]) if len(recent_history) > 1 else global_std_amount
    user_min_amount = np.min([t['amount'] for t in recent_history]) if recent_history else global_min_amount
    user_max_amount = np.max([t['amount'] for t in recent_history]) if recent_history else global_max_amount
    user_transaction_count = len(user_transaction_history[account_id])

    amount_to_avg_ratio = transaction.amount / user_avg_amount if user_avg_amount > 0 else 1.0


    features_dict = {
        'amount': transaction.amount,
        
        'hour_of_day': transaction_dt.hour,
        'day_of_week': transaction_dt.weekday(), 
        'month': transaction_dt.month,
        'day_of_month': transaction_dt.day,
        'week_of_year': transaction_dt.isocalendar().week,
        'time_since_last_txn': time_since_last_txn,
        'user_avg_amount': user_avg_amount,
        'user_std_amount': user_std_amount,
        'user_min_amount': user_min_amount,
        'user_max_amount': user_max_amount,
        'user_transaction_count': user_transaction_count,
        'amount_to_avg_ratio': amount_to_avg_ratio,
        'txn_per_day_7d': txn_per_day_7d,
        'avg_txn_amount_7d': avg_txn_amount_7d,
        
        'is_flagged': int(transaction.is_flagged) 
    }

    for cat_col, categories in ALL_POSSIBLE_CATEGORIES.items():
        
        if cat_col == 'account_id':
            value = transaction.account_id
        else:
            value = getattr(transaction, cat_col, None) 
        
        if value is None:
            warnings.warn(f"Categorical feature '{cat_col}' not provided in transaction, will be encoded as all zeros.")

        for category in categories:
            col_name = f"{cat_col}_{category}"
            features_dict[col_name] = 1 if value == category else 0
            
    df_features = pd.DataFrame([features_dict])

    if not MODEL_EXPECTED_FEATURES:
        warnings.warn("MODEL_EXPECTED_FEATURES was not populated from the model. Inferring from first processed transaction. This is less robust and can cause issues if not all feature types are seen in the first transaction.")
        MODEL_EXPECTED_FEATURES = df_features.columns.tolist()

    final_features = df_features.reindex(columns=MODEL_EXPECTED_FEATURES, fill_value=0)

    
    numerical_cols_to_convert = [
        'amount', 'hour_of_day', 'day_of_week', 'month', 'day_of_month', 'week_of_year',
        'time_since_last_txn', 'user_avg_amount', 'user_std_amount', 'user_min_amount',
        'user_max_amount', 'user_transaction_count', 'amount_to_avg_ratio',
        'txn_per_day_7d', 'avg_txn_amount_7d', 'is_flagged'
    ]
    for col in numerical_cols_to_convert:
        if col in final_features.columns:
            final_features[col] = pd.to_numeric(final_features[col], errors='coerce').fillna(0)


    return final_features


@app.post("/predict", response_model=PredictionOut)
async def predict_fraud(transaction: TransactionIn):
    """
    Receives new transaction data, preprocesses it, and predicts if it's fraudulent.
    """
    print(f"Received request_id: {transaction.request_id}, account_id: {transaction.account_id}, amount: {transaction.amount}")

    account_history = user_transaction_history[transaction.account_id]
    
    try:
        features_df = preprocess_features(transaction)
        
        if features_df.shape[1] != len(MODEL_EXPECTED_FEATURES):
            warnings.warn(f"Feature count mismatch after preprocessing. Expected {len(MODEL_EXPECTED_FEATURES)}, got {features_df.shape[1]}. This might lead to incorrect predictions.")
            print(f"Expected features (count {len(MODEL_EXPECTED_FEATURES)}): {MODEL_EXPECTED_FEATURES}")
            print(f"Actual features (count {features_df.shape[1]}): {features_df.columns.tolist()}")


        if xgb_model is None:
            raise RuntimeError("ML model is not loaded. Cannot make prediction.")

        fraud_probability = xgb_model.predict_proba(features_df)[:, 1][0]
        is_fraud_predicted = bool(fraud_probability > 0.5)

        is_flagged_simulated = bool(fraud_probability > 0.6)

    except Exception as e:
        print(f"Error during model prediction for {transaction.request_id}: {e}")
        if MODEL_EXPECTED_FEATURES:
             print(f"MODEL_EXPECTED_FEATURES (after potential inference): {MODEL_EXPECTED_FEATURES}")
        if 'features_df' in locals():
             print(f"Features DataFrame columns: {features_df.columns.tolist()}")
        raise HTTPException(status_code=500, detail=f"Error during model prediction or feature engineering: {e}")

    user_transaction_history[transaction.account_id].append({
        "timestamp": transaction.timestamp,
        "amount": transaction.amount
    })
    if len(user_transaction_history[transaction.account_id]) > 100:
        user_transaction_history[transaction.account_id] = user_transaction_history[transaction.account_id][-100:]

    return PredictionOut(
        request_id=transaction.request_id,
        is_fraud_predicted=is_fraud_predicted,
        fraud_probability=float(fraud_probability),
        is_flagged_simulated=is_flagged_simulated
    )


@app.get("/")
async def read_root():
    """
    Basic health check endpoint.
    """
    return {"message": "Fraud Detection API is running!"}
