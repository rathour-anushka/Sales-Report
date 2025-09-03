from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Any, List, Optional

app = FastAPI(title="Sales Revenue Prediction API")

# Global variables to store loaded models and data
model = None
label_encoders = None
df_template = None

def load_models_and_data():
    """Load the trained model, encoders, and template data"""
    global model, label_encoders, df_template
    
    try:
        # Load the trained XGBoost model
        model = joblib.load("../model/xgb_regressor.joblib")
        
        # Load the label encoders
        label_encoders = joblib.load("../model/label_encoders.joblib")
        
        # Load the template data to get feature values
        df_template = pd.read_csv("../data/processed/top_15_items_df.csv", low_memory=False)
        df_template['date'] = pd.to_datetime(df_template['date'])
        
        # Create the missing quarter feature (just like in the notebook)
        df_template['quarter'] = (df_template['month'] - 1) // 3 + 1
        
        # Convert weekday from text to numeric (like in notebook: df["weekday"] = df["date"].dt.weekday)
        df_template['weekday'] = df_template['date'].dt.weekday
        
        print("Models and data loaded successfully!")
        print(f"Data range: {df_template['date'].min()} to {df_template['date'].max()}")
        
    except Exception as e:
        print(f"Error loading models and data: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load models and data when the API starts"""
    load_models_and_data()

def create_calendar_features(year: int, month: int, day: int) -> Dict[str, Any]:
    """Create calendar features for the given date"""
    try:
        target_date = datetime(year, month, day)
        weekday = target_date.weekday()
        wday = target_date.isoweekday()
        quarter = (month - 1) // 3 + 1
        week_of_year = target_date.isocalendar()[1]
        wm_yr_wk = year * 100 + week_of_year
        
        return {
            'year': year,
            'month': month,
            'day': day,
            'weekday': weekday,
            'wday': wday,
            'quarter': quarter,
            'wm_yr_wk': wm_yr_wk,
            'date': target_date
        }
    except ValueError as e:
        raise ValueError(f"Invalid date: {year}-{month}-{day}")

def encode_categorical_value(value, encoder_name):
    """Encode a categorical value using the appropriate label encoder"""
    if encoder_name in label_encoders:
        try:
            # Ensure the value is a string before transforming
            return label_encoders[encoder_name].transform([str(value)])[0]
        except ValueError:
            # If value is not in categories, return 0 (or a sensible default for OOV)
            return 0 
    return value

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sales Revenue Prediction API",
        "description": "Predict grand total revenue for a specific date",
        "endpoints": {
            "prediction": "/predict?year=YYYY&month=MM&day=DD",
            "top_items": "/top-items"
        }
    }

@app.get("/predict")
async def predict_revenue(year: int, month: int, day: int):
    """
    Predict grand total revenue for a specific date
    
    Parameters:
    - year: Year (e.g., 2015)
    - month: Month (1-12)  
    - day: Day (1-31)
    """
    
    # Validate inputs
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    
    if not (1 <= day <= 31):
        raise HTTPException(status_code=400, detail="Day must be between 1 and 31")
    
    if year < 2011 or year > 2016:
        raise HTTPException(status_code=400, detail="Year must be between 2011 and 2016 (data available range)")
    
    # Validate the actual date
    try:
        target_date = datetime(year, month, day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date: {year}-{month}-{day}")
    
    try:
        # Check if this exact date exists in our training data
        target_date_str = target_date.strftime('%Y-%m-%d')
        date_data = df_template[df_template['date'].dt.strftime('%Y-%m-%d') == target_date_str]
        
        if not date_data.empty:
            # If date exists in training data, use those exact features for prediction
            features = ['sales', 'cat_id', 'sell_price', 'item_id', 'dept_id', 'store_id',
                        'state_id', 'wm_yr_wk', 'wday', 'month', 'weekday', 'year',
                        'snap_WI', 'snap_TX', 'event_type_1', 'snap_CA', 'event_name_1',
                        'event_name_2', 'event_type_2', 'quarter']
            
            # Create a copy and encode categorical variables
            prediction_data = date_data.copy()
            
            # Encode categorical columns
            cat_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                           'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
            
            for col in cat_columns:
                if col in label_encoders:
                    # Handle potential new categories by converting to string and then transforming
                    prediction_data[col] = label_encoders[col].transform(prediction_data[col].astype(str))
            
            # Get feature matrix and predict
            X = prediction_data[features]
            predictions = model.predict(X)
            
            # Add predictions back to the original data for breakdown
            prediction_data['predicted_revenue'] = predictions
            
            # Group by item_id to get the breakdown (same as notebook logic)
            item_breakdown_readable = (
                prediction_data.groupby('item_id')['predicted_revenue']
                .sum()
                .reset_index()
            )
            
            # Create the breakdown list
            breakdown_list = []
            for _, row in item_breakdown_readable.iterrows():
                breakdown_list.append({
                    "item_id": row['item_id'],
                    "date": f"{year}-{month:02d}-{day:02d}",
                    "predicted_revenue": round(float(row['predicted_revenue']), 2)
                })
            
            total_predicted_revenue = float(predictions.sum())
            
            return {
                "date": f"{year}-{month:02d}-{day:02d}",
                "predicted_grand_total_revenue": round(total_predicted_revenue, 2),
                "currency": "USD",
                "breakdown_by_item": breakdown_list,
                "model_info": {
                    "model_type": "XGBoost Regressor",
                    "combinations_predicted": len(prediction_data),
                    "items_count": len(breakdown_list),
                    "data_source": "historical_date"
                }
            }
        else:
            calendar_features = create_calendar_features(year, month, day)
            most_recent_date = df_template['date'].max()
            days_elapsed = (target_date - most_recent_date).days
            
            # Find historical data for the same month from previous years
            same_month_historical = df_template[df_template['date'].dt.month == month]
            
            # If we have historical data for this month, use it as a better baseline
            if not same_month_historical.empty:
                # Get data from the same month in the most recent year available
                available_years = same_month_historical['date'].dt.year.unique()
                recent_year_for_month = max([y for y in available_years if y < year] if any(y < year for y in available_years) else available_years)
                
                # Find data closest to the target day in that month
                month_data = same_month_historical[same_month_historical['date'].dt.year == recent_year_for_month]
                target_day_data = month_data[month_data['date'].dt.day == day]
                
                if target_day_data.empty:
                    # Find closest day in that month
                    if not month_data.empty:
                        day_diff = abs(month_data['date'].dt.day - day)
                        closest_day = day_diff.idxmin()
                        reference_data = month_data.loc[[closest_day]]
                    else:
                        reference_data = pd.DataFrame() # No data for this month in previous years
                else:
                    reference_data = target_day_data
            else:
                # Fallback to most recent data if no same-month data exists
                reference_data = df_template[df_template['date'] == most_recent_date]
            
            item_store_combinations = df_template[['item_id', 'store_id', 'state_id', 'cat_id', 'dept_id']].drop_duplicates()
            total_predicted_revenue = 0
            breakdown_dict = {}
            
            for _, combination in item_store_combinations.iterrows():
                # Get historical pattern for this specific item-store combination
                combo_reference = reference_data[
                    (reference_data['item_id'] == combination['item_id']) & 
                    (reference_data['store_id'] == combination['store_id'])
                ]
                
                if combo_reference.empty:
                    # Fallback: get any historical data for this combination from same month
                    combo_historical = same_month_historical[
                        (same_month_historical['item_id'] == combination['item_id']) & 
                        (same_month_historical['store_id'] == combination['store_id'])
                    ]
                    
                    if not combo_historical.empty:
                        combo_reference = combo_historical.tail(1)  # Most recent data for this combo in this month
                    else:
                        # Ultimate fallback: most recent data for this combination
                        combo_data = df_template[
                            (df_template['item_id'] == combination['item_id']) & 
                            (df_template['store_id'] == combination['store_id'])
                        ]
                        if not combo_data.empty:
                            combo_reference = combo_data.sort_values('date', ascending=False).head(1)
                        else:
                            continue # Skip this combination if no historical data found
                
                if combo_reference.empty: # Check again after all fallbacks
                    continue

                # Extract features from historical reference data
                sales_value = combo_reference['sales'].iloc[0]
                price_value = combo_reference['sell_price'].iloc[0]
                snap_ca = combo_reference['snap_CA'].iloc[0]
                snap_tx = combo_reference['snap_TX'].iloc[0]
                snap_wi = combo_reference['snap_WI'].iloc[0]
                
                # Handle potential NaN values for event features by providing default empty strings
                event_type_1 = combo_reference['event_type_1'].iloc[0] if pd.notna(combo_reference['event_type_1'].iloc[0]) else ''
                event_name_1 = combo_reference['event_name_1'].iloc[0] if pd.notna(combo_reference['event_name_1'].iloc[0]) else ''
                event_name_2 = combo_reference['event_name_2'].iloc[0] if pd.notna(combo_reference['event_name_2'].iloc[0]) else ''
                event_type_2 = combo_reference['event_type_2'].iloc[0] if pd.notna(combo_reference['event_type_2'].iloc[0]) else ''
                
                # Use target date calendar features with historical business patterns
                combo_vector = np.array([[
                    float(sales_value),
                    int(encode_categorical_value(combination['cat_id'], 'cat_id')),
                    float(price_value),
                    int(encode_categorical_value(combination['item_id'], 'item_id')),
                    int(encode_categorical_value(combination['dept_id'], 'dept_id')),
                    int(encode_categorical_value(combination['store_id'], 'store_id')),
                    int(encode_categorical_value(combination['state_id'], 'state_id')),
                    float(calendar_features['wm_yr_wk']),
                    float(calendar_features['wday']),
                    float(calendar_features['month']),
                    float(calendar_features['weekday']),
                    float(calendar_features['year']),
                    float(snap_wi),
                    float(snap_tx),
                    int(encode_categorical_value(event_type_1, 'event_type_1')),
                    float(snap_ca),
                    int(encode_categorical_value(event_name_1, 'event_name_1')),
                    int(encode_categorical_value(event_name_2, 'event_name_2')),
                    int(encode_categorical_value(event_type_2, 'event_type_2')),
                    float(calendar_features['quarter'])
                ]])
                
                combo_prediction = model.predict(combo_vector)[0]
                item_prediction = max(0, float(combo_prediction))
                total_predicted_revenue += item_prediction
                
                item_id = combination['item_id']
                if item_id not in breakdown_dict:
                    breakdown_dict[item_id] = 0
                breakdown_dict[item_id] += item_prediction
            
            breakdown_list = []
            for item_id, predicted_revenue in breakdown_dict.items():
                breakdown_list.append({
                    "item_id": item_id,
                    "date": f"{year}-{month:02d}-{day:02d}",
                    "predicted_revenue": round(predicted_revenue, 2)
                })
            
            breakdown_list.sort(key=lambda x: x['item_id'])
            
            return {
                "date": f"{year}-{month:02d}-{day:02d}",
                "predicted_grand_total_revenue": round(total_predicted_revenue, 2),
                "currency": "USD",
                "breakdown_by_item": breakdown_list,
                "model_info": {
                    "model_type": "XGBoost Regressor",
                    "combinations_predicted": len(item_store_combinations),
                    "items_count": len(breakdown_list),
                    "data_source": "historical_pattern_projection",
                    "reference_info": {
                        "days_from_last_data": days_elapsed,
                        "pattern_source": "same_month_historical" if not same_month_historical.empty else "most_recent"
                    }
                }
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/top-items")
async def get_top_items(
    top_n: int = 10,
    store_id: Optional[str] = None,
    cat_id: Optional[str] = None,
    dept_id: Optional[str] = None,
    state_id: Optional[str] = None
):
    """
    Retrieve top N items based on their historical average sales revenue,
    with optional filters for store_id, cat_id, dept_id, and state_id.

    Parameters:
    - top_n: Number of top items to retrieve (default is 10).
    - store_id: Optional filter for store ID (e.g., "CA_1").
    - cat_id: Optional filter for category ID (e.g., "HOBBIES").
    - dept_id: Optional filter for department ID (e.g., "HOBBIES_1").
    - state_id: Optional filter for state ID (e.g., "CA").
    """
    if df_template is None:
        raise HTTPException(status_code=503, detail="Models and data not loaded yet. Please try again in a moment.")

    filtered_df = df_template.copy()

    if store_id:
        filtered_df = filtered_df[filtered_df['store_id'] == store_id]
    if cat_id:
        filtered_df = filtered_df[filtered_df['cat_id'] == cat_id]
    if dept_id:
        filtered_df = filtered_df[filtered_df['dept_id'] == dept_id]
    if state_id:
        filtered_df = filtered_df[filtered_df['state_id'] == state_id]

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="No data found for the given filters.")

    # Calculate average sales revenue per item
    item_sales = filtered_df.groupby('item_id')['sales'].mean().reset_index()
    item_sales.rename(columns={'sales': 'average_daily_sales'}, inplace=True)

    # Sort by average sales in descending order and get top_n
    top_items = item_sales.sort_values(by='average_daily_sales', ascending=False).head(top_n)

    # Convert to list of dictionaries for JSON response
    top_items_list = top_items.to_dict(orient='records')

    return {
        "top_n": top_n,
        "filters_applied": {
            "store_id": store_id,
            "cat_id": cat_id,
            "dept_id": dept_id,
            "state_id": state_id
        },
        "top_items": [
            {
                "item_id": item['item_id'],
                "average_daily_sales": round(float(item['average_daily_sales']), 2),
                "currency": "USD"
            }
            for item in top_items_list
        ]
    }