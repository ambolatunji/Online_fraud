import pandas as pd
import joblib
import os
from pathlib import Path

def compute_features(df):
    df = df.copy()
    df['balance_diff_org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['amount_diff_org'] = df['amount'] - df['balance_diff_org']
    df['txn_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['is_sender_zero_bal'] = (df['oldbalanceOrg'] == 0).astype(int)
    df['is_receiver_zero_before'] = (df['oldbalanceDest'] == 0).astype(int)
    df['is_receiver_exact_amount'] = ((df['newbalanceDest'] - df['oldbalanceDest']) == df['amount']).astype(int)
    df['is_large_txn'] = (df['amount'] > 50000).astype(int)
    return df

def save_model(model, name):
    model_path = f"models/{name}.pkl"
    joblib.dump(model, model_path)
    return model_path

def load_model(name):
    model_path = f"models/{name}.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def list_saved_models():
    model_dir = Path("models")
    return [f.stem for f in model_dir.glob("*.pkl")]