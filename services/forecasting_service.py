import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursive
from sklearn.svm import SVR
from typing import Dict, List

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konstanta Nama Kolom (Agar konsisten dengan Database & CSV)
AIR_TEMPERATURE = "Air temperature"
PROCESS_TEMPERATURE = "Process temperature"
ROTATIONAL_SPEED = "Rotational speed"
TORQUE = "Torque"
TOOL_WEAR = "Tool wear"

SENSOR_COLUMN_MAP = {
    'Air temperature [K]': AIR_TEMPERATURE,
    'Process temperature [K]': PROCESS_TEMPERATURE,
    'Rotational speed [rpm]': ROTATIONAL_SPEED,
    'Torque [Nm]': TORQUE,
    'Tool wear [min]': TOOL_WEAR,
}

REQUIRED_COLS = [AIR_TEMPERATURE, PROCESS_TEMPERATURE, ROTATIONAL_SPEED, TORQUE, TOOL_WEAR]
DROPPABLE_COLS = ['UDI', 'Product ID', 'Type', 'Target', 'Failure Type', '_id']

def detect_outliers_iqr(df, column):
    """Helper function untuk deteksi outlier"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def prepare_forecast_data(readings: List[Dict]) -> pd.DataFrame:
    """Konversi data MongoDB ke DataFrame yang bersih untuk forecasting"""
    df = pd.DataFrame(readings)
    logger.info("Loaded historical readings: rows=%s, cols=%s", len(df), list(df.columns))
    return df

def generate_forecast(
    historical_data: pd.DataFrame,
    forecast_minutes: int = 300,
    machine_id: str = None, # Optional logging
    machine_type: str = None # Optional logging
) -> List[Dict]:
    """
    Generate forecast canggih menggunakan SVR dan Skforecast.
    Data diambil realtime dari MongoDB (via historical_data).
    """
    steps = int(forecast_minutes)
    df = historical_data.copy()

    if df.empty:
        logger.error("Historical data is empty; cannot generate forecast")
        raise ValueError("No historical data available for forecasting")

    # 1. Cleanup & Rename Columns
    df = df.drop(columns=[c for c in DROPPABLE_COLS if c in df.columns], errors='ignore')
    df = df.rename(columns=SENSOR_COLUMN_MAP)

    # Keep only sensor columns
    df = df[[c for c in REQUIRED_COLS if c in df.columns]]

    # Validate Columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure Numeric
    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Handle Tool Wear (Pivot Point)
    df = df.dropna(subset=[TOOL_WEAR])
    
    # Handle Duplicates in Tool Wear
    duplicate_count = df[TOOL_WEAR].duplicated().sum()
    if duplicate_count:
        df = df.groupby(TOOL_WEAR, as_index=False).mean()

    # Sort & Reindex
    df = df.sort_values(TOOL_WEAR)
    df = df.set_index(TOOL_WEAR)
    # Create full index range based on max tool wear
    full_index = pd.Index(range(int(df.index.max()) + 1), name=TOOL_WEAR)
    df = df.reindex(full_index)

    # 3. Forecasting Logic per Feature

    # --- A. Air Temperature ---
    _, lower, upper = detect_outliers_iqr(df, AIR_TEMPERATURE)
    df.loc[df[AIR_TEMPERATURE] < lower, AIR_TEMPERATURE] = lower
    df.loc[df[AIR_TEMPERATURE] > upper, AIR_TEMPERATURE] = upper
    df[AIR_TEMPERATURE] = df[AIR_TEMPERATURE].interpolate()
    
    df_reset = df.reset_index(drop=True)
    x = pd.DataFrame(list(df_reset.index), columns=["Index"])
    
    # SVR for Trend
    svr_air = SVR(kernel='rbf')
    # Fill NaNs before fitting if any remain
    y_air = df_reset[AIR_TEMPERATURE].fillna(method='bfill').fillna(method='ffill')
    svr_air.fit(x, y_air)
    y_pred_air = svr_air.predict(x)

    # Skforecast Recursive
    forecaster_air = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster_air.fit(y=pd.Series(y_pred_air, name=AIR_TEMPERATURE))
    
    pred_air = forecaster_air.predict(steps=steps)
    # Clip logical bounds
    pred_air[pred_air < 290] = 290 
    pred_air[pred_air > 310] = 310
    output_df = pd.DataFrame(pred_air.rename(AIR_TEMPERATURE))

    # --- B. Process Temperature (Physics Aware) ---
    # Process Temp biasanya berkorelasi dengan Air Temp (+10K difference assumption)
    _, lower, upper = detect_outliers_iqr(df, PROCESS_TEMPERATURE)
    df.loc[df[PROCESS_TEMPERATURE] < lower, PROCESS_TEMPERATURE] = lower
    df.loc[df[PROCESS_TEMPERATURE] > upper, PROCESS_TEMPERATURE] = upper
    
    # Hitung selisih (delta)
    proc_delta = df[PROCESS_TEMPERATURE] - df[AIR_TEMPERATURE] - 10
    proc_delta = proc_delta.interpolate().fillna(0)
    
    svr_proc = SVR(kernel='rbf')
    svr_proc.fit(x, proc_delta)
    y_pred_proc = svr_proc.predict(x)

    forecaster_proc = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster_proc.fit(y=pd.Series(y_pred_proc, name=PROCESS_TEMPERATURE))
    
    pred_proc_delta = forecaster_proc.predict(steps=steps)
    # Re-add Base Air Temp + 10
    pred_proc = pred_proc_delta.values + output_df[AIR_TEMPERATURE].values + 10
    output_df[PROCESS_TEMPERATURE] = pred_proc

    # --- C. Rotational Speed ---
    _, lower, upper = detect_outliers_iqr(df, ROTATIONAL_SPEED)
    df.loc[df[ROTATIONAL_SPEED] < lower, ROTATIONAL_SPEED] = lower
    df.loc[df[ROTATIONAL_SPEED] > upper, ROTATIONAL_SPEED] = upper
    df[ROTATIONAL_SPEED] = df[ROTATIONAL_SPEED].interpolate().fillna(method='ffill')

    forecaster_rot = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster_rot.fit(y=df[ROTATIONAL_SPEED].reset_index(drop=True))
    pred_rot = forecaster_rot.predict(steps=steps)
    output_df[ROTATIONAL_SPEED] = pred_rot.values

    # --- D. Torque ---
    _, lower, upper = detect_outliers_iqr(df, TORQUE)
    df.loc[df[TORQUE] < lower, TORQUE] = lower
    df.loc[df[TORQUE] > upper, TORQUE] = upper
    df[TORQUE] = df[TORQUE].interpolate().fillna(40) # Default torque 40 if missing

    forecaster_tor = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster_tor.fit(y=df[TORQUE].reset_index(drop=True))
    pred_tor = forecaster_tor.predict(steps=steps)
    pred_tor[pred_tor < 0] = 0 # No negative torque
    output_df[TORQUE] = pred_tor.values

    # --- E. Tool Wear (Index) ---
    # Tool wear diasumsikan naik linear seiring langkah prediksi (index baru)
    # Kita ambil index dari output_df yang merupakan kelanjutan steps
    # Tapi karena output_df indexnya range(0, steps), kita perlu map ke tool wear asli terakhir
    last_wear = df.index.max()
    if pd.isna(last_wear): last_wear = 0
    
    # Buat simulasi kenaikan tool wear (misal naik 1 setiap menit/step)
    new_wear = range(int(last_wear) + 1, int(last_wear) + 1 + steps)
    output_df[TOOL_WEAR] = new_wear

    # 4. Convert to List of Dicts
    forecast_list = []
    for i in range(steps):
        item = {
            "air_temperature": float(output_df[AIR_TEMPERATURE].iloc[i]),
            "process_temperature": float(output_df[PROCESS_TEMPERATURE].iloc[i]),
            "rotational_speed": float(output_df[ROTATIONAL_SPEED].iloc[i]),
            "torque": float(output_df[TORQUE].iloc[i]),
            "tool_wear": float(output_df[TOOL_WEAR].iloc[i]),
        }
        forecast_list.append(item)

    return forecast_list