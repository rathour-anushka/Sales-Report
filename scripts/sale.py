

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load assets
script_dir = Path(__file__).parent
model = joblib.load(script_dir / ".." / "model" / "FOODS_3_586.pkl")
model_features = joblib.load(script_dir / ".." / "model" / "model_features_food3586.pkl")
known_categories = joblib.load(script_dir / ".." / "model" / "known_categories_food3586.pkl")

def preprocess_input(input_data: dict) -> pd.DataFrame:
    # Extract date parts
    date = pd.to_datetime(input_data["date"])
    features = {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "dayofweek": date.dayofweek,
        "dayofyear": date.dayofyear,
        "weekofyear": date.isocalendar().week,
        "quarter": date.quarter,
        "snap_CA": 1 if input_data["snap_state"] == "CA" else 0,
        "snap_TX": 1 if input_data["snap_state"] == "TX" else 0,
        "snap_WI": 1 if input_data["snap_state"] == "WI" else 0,
    }

    # Base DataFrame
    df = pd.DataFrame([features])

    # One-hot encode store_id
    for store in known_categories["store_id"]:
        df[f"store_id_{store}"] = 1 if input_data["store_id"] == store else 0

    # One-hot encode event types as 'unknown' by default
    for cat_name in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        for val in known_categories[cat_name]:
            df[f"{cat_name}_{val}"] = 1 if val == "unknown" else 0  # all unknown in input

    # Align with training features
    final_df = pd.DataFrame(columns=model_features)
    for col in model_features:
        final_df[col] = df[col] if col in df.columns else 0

    return final_df

def predict_total_price(input_data: dict) -> float:
    input_df = preprocess_input(input_data)
    prediction = model.predict(input_df)[0]
    return float(prediction)
