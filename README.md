📘 README.md — Meru Potato Yield Intelligence (Option A)
(Updated for Multi-Year Recursive Forecasting Version)
🥔 Meru Potato Yield & Cold Storage Intelligence
AI-Powered Multi-Year Yield Prediction & Storage Planning Tool (Option A)

This repository contains a Streamlit application for predicting potato crop yield in Meru County using historical datasets and machine-learning models.
It supports:

📊 Exploratory data analysis (EDA)

🤖 Yield modeling using Random Forest regression

🔄 Recursive multi-year forecasting (predict year → year → year)

🎯 User-selectable forecast horizon (next dataset year → current year)

🧊 Cold storage requirement estimation (1000MT, 500MT, 250MT chambers)

📥 Export of forecast + storage plan as JSON

This is Option A of the full architecture — predictions are made using historical datasets only (no NDVI or weather yet).

📁 Project Structure
meru-yield-intelligence/
│
├── app.py                      # Streamlit app (recursive multi-year version)
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation (this file)
│
├── data/                       # (Optional) Upload input files here
│   ├── hvstat_meru_potato.csv
│   ├── adm_meru_potato.csv
│
├── models/                     # Saved ML models (optional for future versions)
│   ├── rf_meru_yield_model.joblib
│   └── lr_meru_yield_model.joblib
│
└── outputs/                    # Forecast & cold storage recommendations
    └── meru_yield_storage_plan.json

📦 Installation & Running the App
1. Clone the repository
git clone https://github.com/<your-username>/meru-yield-intelligence.git
cd meru-yield-intelligence

2. Install dependencies
pip install -r requirements.txt

3. Run the application
streamlit run app.py

4. Upload Required Datasets

When the app loads, upload the following:

hvstat_meru_potato.csv

adm_meru_potato.csv (optional for EDA)

Format expected:

Columns: year, area, production, yield, admin_2, period_date, etc.

🧠 How the Model Works
1. Dataset Aggregation

The HVSTAT dataset is grouped by year to create a clean modeling dataset:

Total Area (ha)

Total Production (tonnes)

Yield (tonnes/hectare)

2. Feature Engineering

The model adds:

yield_lag_1 — previous year yield

yield_lag_2 — 2-years lag

yield_lag_3 — 3-years lag

yield_roll3 — 3-year rolling mean

These features capture trend + momentum in yield history.

3. Model Training

Two models are trained:

Linear Regression

Random Forest Regressor (primary model)

The last historical year is used as a test sample to validate performance.

4. Recursive Multi-Year Forecasting (NEW 🔥)

This version supports forecasting multiple years ahead:

Example:
If dataset ends at 2021, and user selects 2025:

Predict 2022 using real data

Predict 2023 using prediction for 2022

Predict 2024 using prediction for 2023

Predict 2025 using prediction for 2024

This allows forecasting up to current calendar year even if dataset ends earlier.

5. Cold Storage Engine

Based on predicted tonnage:

Calculates required storage capacity

Packs into:

1000 MT chambers

500 MT chambers

250 MT chambers

Ensures max 90% fill rate

Outputs:

Total required storage

Recommended chamber allocation

Overall utilization

📅 Forecast Output Example
Year	Predicted Yield (t/ha)	Predicted Tonnage
2022	15.2	48,120
2023	15.8	49,980
2024	16.4	51,840
2025	17.0	53,700
❄️ Cold Storage Output Example
{
  "predicted_tonnes": 53700,
  "required_capacity": 59667,
  "allocation": [
    {"size": 1000, "count": 59},
    {"size": 500, "count": 1}
  ],
  "total_allocated": 59500,
  "utilization": 0.90
}

📥 Downloadable JSON Report

The app provides a JSON download containing:

{
  "forecast_year": 2025,
  "forecast_yield_t_per_ha": 17.0,
  "predicted_tonnage": 53700,
  "storage_plan": { ... }
}


Stored automatically under:

outputs/meru_yield_storage_plan.json
