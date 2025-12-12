import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursive
from sklearn.svm import SVR
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="skforecast")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    'air_temperature': AIR_TEMPERATURE,
    'process_temperature': PROCESS_TEMPERATURE,
    'rotational_speed': ROTATIONAL_SPEED,
    'torque': TORQUE,
    'tool_wear': TOOL_WEAR
}

REQUIRED_COLS = [AIR_TEMPERATURE, PROCESS_TEMPERATURE, ROTATIONAL_SPEED, TORQUE, TOOL_WEAR]
DROPPABLE_COLS = ['UDI', 'Product ID', 'Type', 'Target', 'Failure Type', '_id', 'machine_id', 'machine_type', 'timestamp', 'uploaded_at']

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

def _fill_missing_values(series: pd.Series, fallback_value: float = 0.0) -> pd.Series:
    """
    Helper kuat untuk mengisi NaN:
    1. Interpolate (isi tengah)
    2. Ffill (isi depan)
    3. Bfill (isi belakang - untuk data awal yang kosong)
    4. Fallback (isi default jika semua gagal)
    """
    return series.interpolate().ffill().bfill().fillna(fallback_value)

def generate_forecast(
    historical_data: pd.DataFrame,
    forecast_minutes: int = 300,
    machine_id: str = None, 
    machine_type: str = None 
) -> List[Dict]:
    """
    Generate forecast canggih menggunakan SVR dan Skforecast.
    """
    steps = int(forecast_minutes)
    if steps < 1: steps = 1
    
    df = historical_data.copy()

    if df.empty:
        logger.error("Historical data is empty; cannot generate forecast")
        raise ValueError("No historical data available for forecasting")

    df = df.rename(columns=SENSOR_COLUMN_MAP)
    df = df.drop(columns=[c for c in DROPPABLE_COLS if c in df.columns], errors='ignore')
    
    df = df[[c for c in REQUIRED_COLS if c in df.columns]]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[TOOL_WEAR])
    
    if df[TOOL_WEAR].duplicated().sum() > 0:
        df = df.groupby(TOOL_WEAR, as_index=False).mean()

    df = df.sort_values(TOOL_WEAR)
    df = df.set_index(TOOL_WEAR)
    
    if df.empty:
         raise ValueError("Dataframe empty after processing tool wear")
         
    full_index = pd.Index(range(int(df.index.max()) + 1), name=TOOL_WEAR)
    df = df.reindex(full_index)

    _, lower, upper = detect_outliers_iqr(df, AIR_TEMPERATURE)
    df.loc[df[AIR_TEMPERATURE] < lower, AIR_TEMPERATURE] = lower
    df.loc[df[AIR_TEMPERATURE] > upper, AIR_TEMPERATURE] = upper
    
    df[AIR_TEMPERATURE] = _fill_missing_values(df[AIR_TEMPERATURE], fallback_value=300.0)
    
    df_reset = df.reset_index(drop=True)
    x = pd.DataFrame(list(df_reset.index), columns=["Index"])
    y_air = df_reset[AIR_TEMPERATURE]

    svr_air = SVR(kernel='rbf')
    svr_air.fit(x, y_air)
    y_pred_air = svr_air.predict(x)

    forecaster_air = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster_air.fit(y=pd.Series(y_pred_air, name=AIR_TEMPERATURE))
    
    pred_air = forecaster_air.predict(steps=steps)
    pred_air[pred_air < 290] = 290 
    pred_air[pred_air > 310] = 310
    output_df = pd.DataFrame(pred_air.rename(AIR_TEMPERATURE))

    _, lower, upper = detect_outliers_iqr(df, PROCESS_TEMPERATURE)
    df.loc[df[PROCESS_TEMPERATURE] < lower, PROCESS_TEMPERATURE] = lower
    df.loc[df[PROCESS_TEMPERATURE] > upper, PROCESS_TEMPERATURE] = upper
    
    proc_delta = df[PROCESS_TEMPERATURE] - df[AIR_TEMPERATURE] - 10
    proc_delta = _fill_missing_values(proc_delta, fallback_value=0.0)
    
    svr_proc = SVR(kernel='rbf')
    svr_proc.fit(x, proc_delta)
    y_pred_proc = svr_proc.predict(x)

    forecaster_proc = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster_proc.fit(y=pd.Series(y_pred_proc, name=PROCESS_TEMPERATURE))
    
    pred_proc_delta = forecaster_proc.predict(steps=steps)
    pred_proc = pred_proc_delta.values + output_df[AIR_TEMPERATURE].values + 10
    output_df[PROCESS_TEMPERATURE] = pred_proc

    _, lower, upper = detect_outliers_iqr(df, ROTATIONAL_SPEED)
    df.loc[df[ROTATIONAL_SPEED] < lower, ROTATIONAL_SPEED] = lower
    df.loc[df[ROTATIONAL_SPEED] > upper, ROTATIONAL_SPEED] = upper
    
    df[ROTATIONAL_SPEED] = _fill_missing_values(df[ROTATIONAL_SPEED], fallback_value=0.0)

    forecaster_rot = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster_rot.fit(y=df[ROTATIONAL_SPEED].reset_index(drop=True))
    pred_rot = forecaster_rot.predict(steps=steps)
    output_df[ROTATIONAL_SPEED] = pred_rot.values

    _, lower, upper = detect_outliers_iqr(df, TORQUE)
    df.loc[df[TORQUE] < lower, TORQUE] = lower
    df.loc[df[TORQUE] > upper, TORQUE] = upper
    
    df[TORQUE] = _fill_missing_values(df[TORQUE], fallback_value=0.0)

    forecaster_tor = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster_tor.fit(y=df[TORQUE].reset_index(drop=True))
    pred_tor = forecaster_tor.predict(steps=steps)
    pred_tor[pred_tor < 0] = 0
    output_df[TORQUE] = pred_tor.values

    last_wear = df.index.max()
    if pd.isna(last_wear): last_wear = 0
    new_wear = range(int(last_wear) + 1, int(last_wear) + 1 + steps)
    output_df[TOOL_WEAR] = new_wear

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