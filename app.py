import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- Streamlit Page Setup ----
st.set_page_config(page_title="Meru Potato Yield & Storage Intelligence", layout="wide")
st.title("🥔 Meru Potato Yield Forecasting (Option A - Datasets Only)")
st.markdown("""
Upload your **hvstat_meru_potato.csv** and **adm_meru_potato.csv** files to explore
EDA, yield prediction, and cold storage recommendations.
""")

# ---- File Upload Section ----
hv_file = st.file_uploader("Upload hvstat_meru_potato.csv", type=["csv"])
adm_file = st.file_uploader("Upload adm_meru_potato.csv", type=["csv"])

if hv_file is None:
    st.stop()

# Load datasets
hv = pd.read_csv(hv_file)
hv.columns = hv.columns.str.lower()

if adm_file:
    adm = pd.read_csv(adm_file)
    adm.columns = adm.columns.str.lower()

# ---- Helper Functions ----

def extract_year(df):
    if "harvest_year" in df.columns:
        return df["harvest_year"]
    if "period_date" in df.columns:
        return pd.to_datetime(df["period_date"], errors="coerce").dt.year
    if "start_date" in df.columns:
        return pd.to_datetime(df["start_date"], errors="coerce").dt.year
    return None

hv["year"] = extract_year(hv)

# ---- EDA Section ----
st.header("📊 Exploratory Data Analysis (HVSTAT)")

if "yield" in hv.columns:
    hv_year = hv.groupby("year")["yield"].mean()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(hv_year.index, hv_year.values, marker="o")
    ax.set_title("Mean Potato Yield Over Time (Meru)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (t/ha)")
    ax.grid(True)
    st.pyplot(fig)

if "area" in hv.columns and "production" in hv.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(hv["area"], hv["production"])
    ax.set_title("Area vs Production")
    ax.set_xlabel("Area (ha)")
    ax.set_ylabel("Production (tonnes)")
    ax.grid(True)
    st.pyplot(fig)

if "admin_2" in hv.columns:
    sub = hv.groupby("admin_2")["yield"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12,4))
    sub.plot(kind="bar", ax=ax)
    ax.set_title("Yield by Sub-county (Meru)")
    ax.set_ylabel("Yield (t/ha)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.markdown("### Summary Statistics")
st.write(hv[["year","area","production","yield"]].describe())

# ---- MODEL PREPARATION ----
st.header("🤖 Yield Modeling (Option A)")

grouped = hv.groupby("year").agg({
    "area": "sum",
    "production": "sum"
}).reset_index()

grouped["yield"] = grouped["production"] / grouped["area"]
grouped = grouped.sort_values("year").reset_index(drop=True)

st.write("### Aggregated Dataset")
st.dataframe(grouped)

# Feature engineering
df = grouped.copy()
for lag in [1,2,3]:
    df[f"yield_lag_{lag}"] = df["yield"].shift(lag)

df["yield_roll3"] = df["yield"].rolling(3).mean().shift(1)
df = df.dropna().reset_index(drop=True)

FEATURES = ["yield_lag_1","yield_lag_2","yield_lag_3","yield_roll3"]

# Train-test split (last year → test)
train = df.iloc[:-1]
test = df.iloc[-1:]

X_train = train[FEATURES]
y_train = train["yield"]
X_test = test[FEATURES]
y_test = test["yield"]

# Train models
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train, y_train)

lr_pred = lr.predict(X_test)[0]
rf_pred = rf.predict(X_test)[0]

def metrics(y, yhat):
    mae = mean_absolute_error([y], [yhat])
    mse = mean_squared_error([y], [yhat])
    rmse = np.sqrt(mse)  # Calculate RMSE manually
    mape = np.mean(np.abs((y - yhat) / y)) * 100
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    }

st.subheader("📈 Model Performance on Test Year")
st.write("### Linear Regression")
st.write(metrics(y_test, lr_pred))
st.write("### Random Forest")
st.write(metrics(y_test, rf_pred))

# ---- FORECAST NEXT YEAR ----
# ---- RECURSIVE FORECAST FUNCTION ----
def recursive_forecast(grouped, rf_model, final_year):
    """
    grouped: historical dataset (year, yield, area)
    rf_model: trained RandomForest model
    final_year: target year for prediction
    """
    df_hist = grouped.copy().reset_index(drop=True)
    last_year = int(df_hist["year"].iloc[-1])
    results = []

    for year in range(last_year + 1, final_year + 1):

        # Build lag features
        lag1 = df_hist["yield"].iloc[-1]
        lag2 = df_hist["yield"].iloc[-2] if len(df_hist) >= 2 else lag1
        lag3 = df_hist["yield"].iloc[-3] if len(df_hist) >= 3 else lag1
        roll3 = df_hist["yield"].iloc[-3:].mean()

        X = np.array([[lag1, lag2, lag3, roll3]])
        pred_yield = rf_model.predict(X)[0]

        # Use last known area for production prediction
        area = df_hist["area"].iloc[-1]
        pred_tonnes = pred_yield * area

        # Store prediction
        results.append({
            "year": year,
            "predicted_yield_t_ha": pred_yield,
            "predicted_tonnage": pred_tonnes
        })

        # Append prediction for next iteration
        df_hist = pd.concat([
            df_hist,
            pd.DataFrame({
                "year": [year],
                "yield": [pred_yield],
                "area": [area],
                "production": [pred_tonnes]
            })
        ], ignore_index=True)

    return pd.DataFrame(results)

# ---- MULTI-YEAR FORECASTING ----
st.header("📅 Multi-Year Forecasting")

last_year = int(grouped["year"].max())
current_year = datetime.datetime.now().year

# User selects prediction horizon
target_year = st.number_input(
    "Select the forecast target year:",
    min_value=last_year + 1,
    max_value=current_year,
    value=last_year + 1,
    step=1
)

st.write(f"Dataset ends at **{last_year}**")
st.write(f"Forecasting recursively up to **{target_year}**")

# Run recursive forecast
forecast_df = recursive_forecast(grouped, rf, target_year)

st.subheader("🔮 Year-by-Year Forecast (Recursive)")
st.dataframe(forecast_df)

# Select final year output
final_pred_yield = forecast_df.iloc[-1]["predicted_yield_t_ha"]
final_pred_tonnes = forecast_df.iloc[-1]["predicted_tonnage"]

# Display metrics
st.metric("Final Year Predicted Yield (t/ha)", f"{final_pred_yield:.2f}")
st.metric("Final Year Predicted Production (tonnes)", f"{final_pred_tonnes:,.0f}")

# Pass to storage engine
predicted_tonnage = final_pred_tonnes


# ---- STORAGE ENGINE ----
st.header("❄️ Cold Storage Requirement Engine")

def pack_storage(total_tonnes, chamber_sizes=[1000,500,250], max_fill=0.9):
    required_capacity = int(np.ceil(total_tonnes / max_fill))
    allocation = []
    remaining = required_capacity

    for size in chamber_sizes:
        count = remaining // size
        if count > 0:
            allocation.append({"size": size, "count": count})
            remaining -= count * size

    if remaining > 0:
        allocation.append({"size": chamber_sizes[-1], "count": 1})

    capacity = sum(a["size"] * a["count"] for a in allocation)
    utilization = total_tonnes / capacity

    return {
        "predicted_tonnes": total_tonnes,
        "required_capacity": required_capacity,
        "allocation": allocation,
        "total_allocated": capacity,
        "utilization": utilization
    }

plan = pack_storage(predicted_tonnage)

st.subheader("📦 Recommended Storage Allocation")
st.json(plan)

# ---- DOWNLOAD RESULTS ----
# ---- DOWNLOAD RESULTS ----
st.header("⬇ Download Results")

results = {
    "forecast_year": target_year,
    "forecast_yield_t_per_ha": final_pred_yield,
    "predicted_tonnage": predicted_tonnage,
    "storage_plan": plan
}

st.download_button(
    "Download Forecast + Storage Plan (JSON)",
    data=str(results),
    file_name="meru_yield_storage_plan.json",
    mime="application/json"
)
