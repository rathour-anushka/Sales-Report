# Sales-Report

This project combines machine learning models for sales forecasting with a dynamic Power BI dashboard connected to a Flask API backend. The system provides real-time sales analytics and predictions.

## Project Structure

```
├── app/
│   ├── app.py         # Flask API implementation
│   └── main.py        # Main application entry point
├── model/
│   ├── FOODS_3_586.pkl               # Trained model data
│   ├── known_categories_food3586.pkl  # Category mappings
│   ├── label_encoders.joblib         # Label encoders
│   ├── model_features_food3586.pkl    # Feature definitions
│   └── xgb_regressor.joblib          # XGBoost model
├── notebook/
│   ├── model.ipynb    # Model development notebook
│   ├── sale.ipynb     # Sales analysis notebook
│   └── top15.ipynb    # Top 15 analysis notebook
├── scripts/
│   ├── sale.py        # Sales processing scripts
│   └── top15.py       # Top 15 analysis scripts
└── util/
    └── utills.py      # Utility functions
```

## Features

- Sales forecasting using XGBoost regression model
- Real-time data visualization through Power BI dashboard
- RESTful API for dynamic data updates
- Comprehensive data analysis notebooks
- Category-based sales analysis
- Top 15 products analysis

## Power BI Dashboard Integration

The project includes a Power BI dashboard (`sales.forecast.pbix`) that connects to the Flask API for real-time data visualization. The dashboard provides:

- Sales forecasting visualizations
- Category-wise analysis
- Top performing products
- Dynamic updates through API integration

### Setting up Power BI Connection

1. Open `sales.forecast.pbix` in Power BI Desktop
2. The dashboard is configured to connect to the local API endpoint: `http://localhost:5000`
3. Ensure the Flask API is running before refreshing the dashboard
4. Use Power BI's scheduled refresh features for automated updates

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rathour-anushka/Sales-Report.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask API:
```bash
python app/main.py
```

2. The API will be available at `http://localhost:5000`

3. Open `sales.forecast.pbix` to view the dashboard

## API Endpoints

The Flask API provides the following endpoints:

- `GET /sales/forecast`: Get sales forecasts
- `GET /sales/categories`: Get category-wise sales data
- `GET /sales/top15`: Get top 15 performing products

## Model Information

The project uses an XGBoost regression model for sales forecasting. The model is trained on historical sales data and considers various features including:

- Historical sales patterns
- Product categories
- Seasonal factors

Model files are stored in the `model/` directory and include:
- Pre-trained XGBoost model
- Feature encoders
- Category mappings

## Development

The `notebook/` directory contains Jupyter notebooks used for:
- Model development and training
- Sales data analysis
- Top 15 products analysis

## Requirements

- Python 3.7+
- Power BI Desktop
- Required Python packages listed in `requirements.txt`


## Contact

Anushka Rathour - [GitHub](https://github.com/rathour-anushka)