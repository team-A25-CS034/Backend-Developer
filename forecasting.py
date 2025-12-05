import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursive
from sklearn.svm import SVR
from typing import Dict, List
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


def _ensure_index(df: pd.DataFrame, index_col: str = "tool_wear") -> pd.DataFrame:
    """Set index to `index_col` when present; otherwise use a RangeIndex.

    Mirrors the notebook behaviour where `tool_wear` is used as the index and
    reindexed to a continuous range for interpolation.
    """
    if index_col in df.columns:
        df = df.set_index(index_col)

        # If there are duplicate labels, aggregate numeric columns by mean
        # before reindexing to avoid "cannot reindex on an axis with duplicate labels".
        if df.index.has_duplicates:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                df = df.groupby(df.index)[numeric_cols].mean()
            else:
                df = df.groupby(df.index).first()

        max_idx = int(df.index.max()) if len(df.index) else -1
        full_index = pd.Index(range(max_idx + 1), name=index_col)
        df = df.reindex(full_index)
    else:
        df = df.copy()
        df.index = pd.RangeIndex(len(df))
    return df


def forecast_parameter(
    data: pd.Series, 
    steps: int = 7, 
    lags: int = None,
    clip_min: float = None
) -> np.ndarray:
    """Not used in the notebook-aligned workflow."""
    raise NotImplementedError("Use the per-parameter forecasting functions instead.")


def forecast_air_temperature(historical_data: pd.DataFrame, steps: int = 300) -> np.ndarray:
    """Forecast air temperature exactly as in `forecaster new.ipynb`."""
    if "air_temperature" not in historical_data.columns:
        raise KeyError("'air_temperature' column missing from historical_data")

    df = _ensure_index(historical_data, "tool_wear")

    _, lower_bound, upper_bound = detect_outliers_iqr(df, "air_temperature")
    df.loc[df["air_temperature"] < lower_bound, "air_temperature"] = lower_bound
    df.loc[df["air_temperature"] > upper_bound, "air_temperature"] = upper_bound
    df["air_temperature"] = df["air_temperature"].interpolate()

    df = df.reset_index(drop=True)
    x = pd.DataFrame(list(df.index), columns=["Index"]).astype(float)
    y = df["air_temperature"].values

    svr = SVR(kernel="rbf")
    y_vals = np.nan_to_num(y, nan=np.nanmean(y) if np.isfinite(np.nanmean(y)) else 0.0)
    svr.fit(x.values, y_vals)
    y_pred = svr.predict(x.values)

    train = y_pred[:-50] if len(y_pred) > 50 else y_pred

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster.fit(y=pd.Series(train, name="Air temperature"))
    predictions = forecaster.predict(steps=steps).values

    predictions = np.clip(predictions, 294.0, 306.0)
    return predictions


def forecast_process_temperature(
    historical_data: pd.DataFrame, 
    air_temp_forecast: np.ndarray,
    steps: int = 300
) -> np.ndarray:
    """Forecast process temperature exactly as in `forecaster new.ipynb`."""
    if "process_temperature" not in historical_data.columns or "air_temperature" not in historical_data.columns:
        raise KeyError("Required columns missing for process temperature forecasting")

    df = _ensure_index(historical_data, "tool_wear")
    df["process_rel"] = df["process_temperature"] - df["air_temperature"] - 10

    _, lower_bound, upper_bound = detect_outliers_iqr(df, "process_rel")
    df.loc[df["process_rel"] < lower_bound, "process_rel"] = lower_bound
    df.loc[df["process_rel"] > upper_bound, "process_rel"] = upper_bound
    df["process_rel"] = df["process_rel"].interpolate()

    df = df.reset_index(drop=True)
    x = pd.DataFrame(list(df.index), columns=["Index"]).astype(float)
    y = df["process_rel"].values

    svr = SVR(kernel="rbf")
    y_vals = np.nan_to_num(y, nan=np.nanmean(y) if np.isfinite(np.nanmean(y)) else 0.0)
    svr.fit(x.values, y_vals)
    y_pred = svr.predict(x.values)

    train = y_pred[:-50] if len(y_pred) > 50 else y_pred

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster.fit(y=pd.Series(train, name="Process temperature"))
    relative_forecast = forecaster.predict(steps=steps).values

    relative_forecast = np.clip(relative_forecast, -3.0, 3.0)
    predictions = relative_forecast + air_temp_forecast + 10.0
    return predictions


def forecast_rotational_speed(historical_data: pd.DataFrame, steps: int = 300) -> np.ndarray:
    """Forecast rotational speed as in the notebook (mean baseline + forecaster)."""
    if "rotational_speed" not in historical_data.columns:
        raise KeyError("'rotational_speed' column missing from historical_data")

    df = _ensure_index(historical_data, "tool_wear")
    _, lower_bound, upper_bound = detect_outliers_iqr(df, "rotational_speed")
    df.loc[df["rotational_speed"] < lower_bound, "rotational_speed"] = lower_bound
    df.loc[df["rotational_speed"] > upper_bound, "rotational_speed"] = upper_bound
    df["rotational_speed"] = df["rotational_speed"].interpolate()

    y = df["rotational_speed"].dropna().values
    baseline = float(np.mean(y)) if len(y) else 0.0
    y_pred = baseline * np.ones_like(y)

    train = y_pred[:-50] if len(y_pred) > 50 else y_pred

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster.fit(y=pd.Series(train, name="Rotational speed"))
    preds = forecaster.predict(steps=steps).values
    return preds


def forecast_torque(historical_data: pd.DataFrame, steps: int = 300) -> np.ndarray:
    """Forecast torque as in the notebook (constant 40 baseline + forecaster)."""
    if "torque" not in historical_data.columns:
        raise KeyError("'torque' column missing from historical_data")

    df = _ensure_index(historical_data, "tool_wear")
    _, lower_bound, upper_bound = detect_outliers_iqr(df, "torque")
    df.loc[df["torque"] < lower_bound, "torque"] = lower_bound
    df.loc[df["torque"] > upper_bound, "torque"] = upper_bound
    df["torque"] = df["torque"].interpolate()

    y = df["torque"].dropna().values
    y_pred = 40.0 * np.ones_like(y)

    train = y_pred[:-50] if len(y_pred) > 50 else y_pred

    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=1)
    forecaster.fit(y=pd.Series(train, name="Torque"))
    preds = forecaster.predict(steps=steps).values
    preds[preds < 0] = 0
    return preds


def forecast_tool_wear(historical_data: pd.DataFrame, steps: int = 300) -> np.ndarray:
    """Notebook export uses the horizon index as Tool wear progression."""
    _ = historical_data  # kept for signature parity; not used in notebook flow
    return np.arange(steps, dtype=float)


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
    
    steps = int(forecast_minutes)

    air_temp_forecast = forecast_air_temperature(historical_data, steps=steps)
    process_temp_forecast = forecast_process_temperature(
        historical_data,
        air_temp_forecast,
        steps=steps,
    )
    rotational_speed_forecast = forecast_rotational_speed(historical_data, steps=steps)
    torque_forecast = forecast_torque(historical_data, steps=steps)
    tool_wear_forecast = forecast_tool_wear(historical_data, steps=steps)
    
    has_timestamp = "timestamp" in historical_data.columns and not historical_data.empty
    last_timestamp = (
        pd.to_datetime(historical_data["timestamp"].iloc[-1]) if has_timestamp else None
    )

    forecast_list = []
    for i in range(steps):
        future_ts = None
        if has_timestamp and last_timestamp is not None:
            future_ts = (last_timestamp + timedelta(minutes=i + 1)).isoformat()

        forecast_dict = {
            "timestamp": future_ts,
            "machine_id": machine_id,
            "machine_type": machine_type,
            "air_temperature": float(air_temp_forecast[i]),
            "process_temperature": float(process_temp_forecast[i]),
            "rotational_speed": float(rotational_speed_forecast[i]),
            "torque": float(torque_forecast[i]),
            "tool_wear": float(tool_wear_forecast[i]),
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
