import sys
from pathlib import Path
from enum import Enum
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to the Python path to allow for absolute imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.sale import predict_total_price

app = FastAPI(title="Sales Prediction API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Prediction API. Go to /docs for the API documentation."}

# Load known values for store_id from file
known_categories_path = project_root / "model" / "known_categories_food3586.pkl"
try:
    known_categories = joblib.load(known_categories_path)
except FileNotFoundError:
    raise RuntimeError(
        f"Could not load known categories from {known_categories_path}. "
        "Make sure the model files are in the correct location."
    ) from None
known_stores = known_categories.get("store_id", [])

# Inline Enum for snap_state validation
class SnapState(str, Enum):
    CA = "CA"
    TX = "TX"
    WI = "WI"

# Inline request model
class PredictionRequest(BaseModel):
    date: str
    snap_state: SnapState  # Only allows CA, TX, WI
    store_id: str

# Inline response model
class PredictionResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    input_data = request.model_dump()

    # ✅ Validate store_id
    if input_data["store_id"] not in known_stores:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid store_id: {input_data['store_id']}. Must be one of: {known_stores[:5]}..."
        )

    try:
        prediction = predict_total_price(input_data)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
