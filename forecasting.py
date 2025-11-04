"""
Forecasting utilities for sensor data prediction
Using ForecasterRecursive for all parameters
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursive
from sklearn.svm import SVR
from typing import Dict, List, Optional
from datetime import datetime, timedelta


def prepare_forecast_data(readings: List[Dict]) -> pd.DataFrame:
    """Convert MongoDB readings to DataFrame for forecasting"""
    df = pd.DataFrame(readings)
    
    # Convert timestamp if needed
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    
    # Keep only numeric columns needed for forecasting
    numeric_cols = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str):
    """Return outliers (rows), lower and upper bounds using IQR method.

    This mirrors the helper used in the notebook to clamp extreme values before
    interpolation and smoothing.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    series = df[column].dropna()
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def forecast_parameter(
    data: pd.Series, 
    steps: int = 7, 
    lags: int = None,
    clip_min: float = None
) -> np.ndarray:
    """
    Generic forecasting function using ForecasterRecursive
    
    Args:
        data: Pandas Series with historical data
        steps: Number of steps to forecast
        lags: Number of lags to use (auto-calculated if None)
        clip_min: Minimum value for predictions (optional)
    
    Returns:
        Array of forecasted values
    """
    # This generic helper is retained but note: the notebook applies a slightly
    # different workflow per-parameter (SVR smoothing then ForecasterRecursive
    # with lags=1). Use specialized functions below where needed. Here we keep
    # a conservative default behaviour.
    if lags is None:
        lags = min(7, max(1, len(data) - 1))
    else:
        lags = min(max(1, lags), max(1, len(data) - 1))

    forecaster = ForecasterRecursive(
        regressor=LinearRegression(),
        lags=lags
    )
    forecaster.fit(y=data)

    predictions = forecaster.predict(steps=steps)
    predictions_array = predictions.values

    if clip_min is not None:
        predictions_array = np.maximum(predictions_array, clip_min)

    return predictions_array


def forecast_air_temperature(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """Forecast air temperature following the notebook steps:

    1. Set index to `tool_wear` (if present) and reindex to a full continuous
       integer index so interpolation works like in the notebook.
    2. Detect and clamp outliers using IQR.
    3. Interpolate missing values.
    4. Fit an SVR (RBF) on the index -> air_temperature to obtain a smoothed
       series (y_pred).
    5. Fit a ForecasterRecursive (LinearRegression) on the smoothed series with
       lags=1 (the notebook uses lags=1) and predict `steps` ahead.
    6. Apply clipping to keep predictions in a reasonable physical range.
    """
    df = historical_data.copy()

    if "air_temperature" not in df.columns:
        raise KeyError("'air_temperature' column missing from historical_data")

    # If tool_wear exists, try to use it as the index and reindex to full range
    # (0..max). If the tool_wear column contains duplicate labels (possible for
    # synthetic/per-minute data) fall back to a simple integer index.
    if "tool_wear" in df.columns:
        df = df.set_index("tool_wear")
        if df.index.has_duplicates:
            # Fall back: preserve row order and use integer index
            df = df.reset_index(drop=True)
            df.index = pd.Index(range(len(df)), name="tool_wear")
        else:
            full_index = pd.Index(range(int(df.index.max()) + 1), name="tool_wear")
            df = df.reindex(full_index)

    # Clamp outliers and interpolate
    _, lower_bound, upper_bound = detect_outliers_iqr(df, "air_temperature")
    df.loc[df["air_temperature"] < lower_bound, "air_temperature"] = lower_bound
    df.loc[df["air_temperature"] > upper_bound, "air_temperature"] = upper_bound
    df["air_temperature"] = df["air_temperature"].interpolate()

    # Prepare SVR smoothing: X is the index values
    x = pd.DataFrame(list(df.index), columns=["Index"]).astype(float)
    y = df["air_temperature"].values

    # Fit SVR to obtain smoothed values (mirrors the notebook)
    svr = SVR(kernel="rbf")
    # If there are NaNs after interpolation (edge cases), fill forward/backward
    x_vals = x.values
    y_vals = np.nan_to_num(y, nan=np.nanmean(y) if np.isfinite(np.nanmean(y)) else 0.0)
    svr.fit(x_vals, y_vals)
    y_pred = svr.predict(x_vals)

    # Train forecaster on smoothed series; notebook uses a train/test split with
    # the last 50 points reserved for testing. We'll follow that pattern if
    # enough points exist, otherwise fit on the entire smoothed series.
    if len(y_pred) > 50:
        train = y_pred[:-50]
    else:
        train = y_pred

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster.fit(y=pd.Series(train, name="Air temperature"))
    predictions = forecaster.predict(steps=steps)

    # Notebook clamps air temperature forecasts to [294, 306]
    predictions_array = predictions.values
    predictions_array = np.clip(predictions_array, 294.0, 306.0)
    return predictions_array


def forecast_process_temperature(
    historical_data: pd.DataFrame, 
    air_temp_forecast: np.ndarray,
    steps: int = 7
) -> np.ndarray:
    """Follow the notebook:

    1. Compute process_temperature_ori = Process temperature - Air temperature - 10
       (this centers the relative difference used in the notebook).
    2. Clamp outliers, interpolate, smooth and forecast the relative difference
       using SVR + ForecasterRecursive (lags=1), then add back the predicted
       air temperature + 10.
    """
    df = historical_data.copy()

    if "process_temperature" not in df.columns or "air_temperature" not in df.columns:
        raise KeyError("Required columns missing for process temperature forecasting")

    # Build the relative series and handle index/reindex if tool_wear present
    df_rel = df.copy()
    df_rel["process_rel"] = df_rel["process_temperature"] - df_rel["air_temperature"] - 10

    if "tool_wear" in df_rel.columns:
        df_rel = df_rel.set_index("tool_wear")
        if df_rel.index.has_duplicates:
            df_rel = df_rel.reset_index(drop=True)
            df_rel.index = pd.Index(range(len(df_rel)), name="tool_wear")
        else:
            full_index = pd.Index(range(int(df_rel.index.max()) + 1), name="tool_wear")
            df_rel = df_rel.reindex(full_index)

    _, lower_bound, upper_bound = detect_outliers_iqr(df_rel, "process_rel")
    df_rel.loc[df_rel["process_rel"] < lower_bound, "process_rel"] = lower_bound
    df_rel.loc[df_rel["process_rel"] > upper_bound, "process_rel"] = upper_bound
    df_rel["process_rel"] = df_rel["process_rel"].interpolate()

    x = pd.DataFrame(list(df_rel.index), columns=["Index"]).astype(float)
    y = df_rel["process_rel"].values

    svr = SVR(kernel="rbf")
    y_vals = np.nan_to_num(y, nan=np.nanmean(y) if np.isfinite(np.nanmean(y)) else 0.0)
    svr.fit(x.values, y_vals)
    y_pred = svr.predict(x.values)

    if len(y_pred) > 50:
        train = y_pred[:-50]
    else:
        train = y_pred

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster.fit(y=pd.Series(train, name="Process rel"))
    relative_forecast = forecaster.predict(steps=steps).values

    # Notebook clamps relative process forecasts to [-3, 3]
    relative_forecast = np.clip(relative_forecast, -3.0, 3.0)

    # Add back air temp forecast + 10
    predictions = relative_forecast + air_temp_forecast + 10.0
    return predictions


def forecast_rotational_speed(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """Follow notebook: use the mean as baseline then forecast that baseline
    (the notebook sets y_pred = mean(y) * ones_like(y) and uses ForecasterRecursive
    with lags=1).
    """
    if "rotational_speed" not in historical_data.columns:
        raise KeyError("'rotational_speed' column missing from historical_data")

    y = historical_data["rotational_speed"].dropna().values
    baseline = float(np.mean(y)) if len(y) > 0 else 0.0
    y_pred = baseline * np.ones_like(y)

    if len(y_pred) > 50:
        train = y_pred[:-50]
    else:
        train = y_pred

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster.fit(y=pd.Series(train, name="Rotational speed"))
    preds = forecaster.predict(steps=steps).values
    preds = np.maximum(preds, 0.0)
    return preds


def forecast_torque(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """Follow notebook: use a fixed baseline (40) like the notebook's example
    (y_pred = 40 * ones_like(y)) then forecaster with lags=1.
    """
    if "torque" not in historical_data.columns:
        raise KeyError("'torque' column missing from historical_data")

    y = historical_data["torque"].dropna().values
    # Notebook uses a constant 40 for smoothing baseline
    y_pred = 40.0 * np.ones_like(y)

    if len(y_pred) > 50:
        train = y_pred[:-50]
    else:
        train = y_pred

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster.fit(y=pd.Series(train, name="Torque"))
    preds = forecaster.predict(steps=steps).values
    preds = np.maximum(preds, 0.0)
    return preds


def forecast_tool_wear(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """
    Forecast tool wear - monotonically increasing
    Tool wear increases over time, never decreases
    """
    df = historical_data.copy()

    if "tool_wear" not in df.columns:
        raise KeyError("'tool_wear' column missing from historical_data")

    # Use the existing generic forecaster but ensure predictions never go below
    # the last observed value and that the series is monotonically increasing.
    preds = forecast_parameter(data=df["tool_wear"], steps=steps, lags=14)

    last_value = float(df["tool_wear"].dropna().iloc[-1])
    for i in range(len(preds)):
        if preds[i] < last_value:
            preds[i] = last_value
        last_value = preds[i]

    return preds


def generate_forecast(
    historical_data: pd.DataFrame,
    machine_id: str,
    machine_type: str,
    forecast_minutes: int = 300,
) -> List[Dict]:
    """
    Generate complete forecast for all sensor parameters
    
    Args:
        historical_data: DataFrame with historical sensor readings
        machine_id: Machine identifier
        machine_type: Machine type (L, M, H)
        forecast_days: Number of days to forecast
    
    Returns:
        List of forecast dictionaries with all sensor values
    """
    
    # The system is now fixed to minute-based forecasts. Each step == 1 minute.
    steps = int(forecast_minutes)

    # Forecast each parameter using ForecasterRecursive (steps above)
    air_temp_forecast = forecast_air_temperature(historical_data, steps=steps)
    process_temp_forecast = forecast_process_temperature(
        historical_data,
        air_temp_forecast,
        steps=steps,
    )
    rotational_speed_forecast = forecast_rotational_speed(historical_data, steps=steps)
    torque_forecast = forecast_torque(historical_data, steps=steps)
    tool_wear_forecast = forecast_tool_wear(historical_data, steps=steps)
    
    # Get the last timestamp from historical data
    last_timestamp = pd.to_datetime(historical_data['timestamp'].iloc[-1])
    
    # Build forecast list: if forecast_minutes is provided, increment by minutes
    # otherwise increment by days (backwards-compatible).
    forecast_list = []
    for i in range(steps):
        # Fixed minute-based forecast: increment timestamp by minutes
        future_date = last_timestamp + timedelta(minutes=i + 1)
        
        forecast_dict = {
            'timestamp': future_date.isoformat(),
            'day_ahead': i + 1,
            'machine_id': machine_id,
            'machine_type': machine_type,
            'air_temperature': float(air_temp_forecast[i]),
            'process_temperature': float(process_temp_forecast[i]),
            'rotational_speed': float(rotational_speed_forecast[i]),
            'torque': float(torque_forecast[i]),
            'tool_wear': float(tool_wear_forecast[i])
        }
        
        forecast_list.append(forecast_dict)
    
    return forecast_list


def _print_forecast_sample(forecast: List[Dict], sample_size: int = 5) -> None:
    print(f"Generated {len(forecast)} forecast rows")
    for i in range(min(sample_size, len(forecast))):
        item = forecast[i]
        print(
            i + 1,
            item.get("timestamp"),
            f"air={item.get('air_temperature'):.3f}",
            f"proc={item.get('process_temperature'):.3f}",
            f"rot={item.get('rotational_speed'):.1f}",
            f"torque={item.get('torque'):.2f}",
            f"tool_wear={item.get('tool_wear'):.2f}",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run minute-based forecast using forecasting module")
    parser.add_argument("--csv", default="d:/project/asah-github-org/Backend-Developer/dummy_sensor_data.csv", help="Path to historical CSV")
    # Fixed minute horizon (default 300). Keep the flag for convenience.
    parser.add_argument("--minutes", dest="minutes", type=int, default=300, help="Number of minutes to forecast (default 300)")
    parser.add_argument("--machine-id", default="machine_01")
    parser.add_argument("--machine-type", default="M")
    args = parser.parse_args()

    # Load CSV and run forecast
    df = pd.read_csv(args.csv)
    readings = df.to_dict("records")
    hist = prepare_forecast_data(readings)

    forecast = generate_forecast(historical_data=hist, machine_id=args.machine_id, machine_type=args.machine_type, forecast_minutes=args.minutes)
    _print_forecast_sample(forecast, sample_size=5)
