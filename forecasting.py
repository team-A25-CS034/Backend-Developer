"""
Forecasting utilities for sensor data prediction
Based on skforecast with ForecasterRecursive
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from skforecast.recursive import ForecasterRecursive
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


def forecast_air_temperature(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """Forecast air temperature using SVR + ForecasterRecursive"""
    
    # Prepare data
    df = historical_data.copy()
    df = df.reset_index(drop=True)
    
    # Use index as x for SVR regression
    x = pd.DataFrame(list(df.index), columns=["Index"])
    y = df["air_temperature"]
    
    # SVR regression for smoothing
    svr = SVR(kernel='rbf')
    svr.fit(x, y)
    y_pred = svr.predict(x)
    
    # Forecasting with ForecasterRecursive
    train_dataset = pd.DataFrame(y_pred, columns=["air_temperature"])
    forecaster = ForecasterRecursive(
        regressor=LinearRegression(),
        lags=1
    )
    forecaster.fit(y=train_dataset["air_temperature"])
    
    predictions = forecaster.predict(steps=steps)
    return predictions.values


def forecast_process_temperature(
    historical_data: pd.DataFrame, 
    air_temp_forecast: np.ndarray,
    steps: int = 7
) -> np.ndarray:
    """Forecast process temperature (relative to air temperature)"""
    
    df = historical_data.copy()
    df = df.reset_index(drop=True)
    
    # Calculate relative process temperature (Process - Air - 10)
    process_temp_relative = df["process_temperature"] - df["air_temperature"] - 10
    
    # Use index as x for SVR regression
    x = pd.DataFrame(list(df.index), columns=["Index"])
    y = process_temp_relative
    
    # SVR regression
    svr = SVR(kernel='rbf')
    svr.fit(x, y)
    y_pred = svr.predict(x)
    
    # Forecasting
    train_dataset = pd.DataFrame(y_pred, columns=["process_temperature"])
    forecaster = ForecasterRecursive(
        regressor=LinearRegression(),
        lags=1
    )
    forecaster.fit(y=train_dataset["process_temperature"])
    
    predictions = forecaster.predict(steps=steps)
    
    # Convert back to absolute temperature (add air temp + 10)
    predictions = predictions.values + air_temp_forecast + 10
    
    return predictions


def forecast_rotational_speed(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """Forecast rotational speed"""
    
    df = historical_data.copy()
    y = df["rotational_speed"]
    
    forecaster = ForecasterRecursive(
        regressor=LinearRegression(),
        lags=min(50, len(y) - 1)  # Adjust lags if data is short
    )
    forecaster.fit(y=y)
    
    predictions = forecaster.predict(steps=steps)
    return predictions.values


def forecast_torque(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """Forecast torque"""
    
    df = historical_data.copy()
    y = df["torque"]
    
    forecaster = ForecasterRecursive(
        regressor=LinearRegression(),
        lags=min(50, len(y) - 1)  # Adjust lags if data is short
    )
    forecaster.fit(y=y)
    
    predictions = forecaster.predict(steps=steps)
    
    # Ensure no negative torque values
    predictions = np.maximum(predictions.values, 0)
    
    return predictions


def forecast_tool_wear(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """
    Forecast tool wear - simple linear progression
    For daily monitoring, tool wear increases by ~24 minutes per day
    """
    
    df = historical_data.copy()
    
    # Get the last known tool wear value
    last_tool_wear = df["tool_wear"].iloc[-1]
    
    # Calculate daily increase (assume linear wear)
    if len(df) > 1:
        # Calculate average daily increase from historical data
        first_tool_wear = df["tool_wear"].iloc[0]
        days_elapsed = len(df) - 1
        if days_elapsed > 0:
            daily_increase = (last_tool_wear - first_tool_wear) / days_elapsed
        else:
            daily_increase = 24  # Default: 24 minutes per day
    else:
        daily_increase = 24  # Default: 24 minutes per day
    
    # Project forward
    future_tool_wear = [last_tool_wear + (i + 1) * daily_increase for i in range(steps)]
    
    return np.array(future_tool_wear)


def generate_forecast(
    historical_data: pd.DataFrame,
    machine_id: str,
    machine_type: str,
    forecast_days: int = 7
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
    
    # Forecast each parameter
    air_temp_forecast = forecast_air_temperature(historical_data, steps=forecast_days)
    process_temp_forecast = forecast_process_temperature(
        historical_data, 
        air_temp_forecast, 
        steps=forecast_days
    )
    rotational_speed_forecast = forecast_rotational_speed(historical_data, steps=forecast_days)
    torque_forecast = forecast_torque(historical_data, steps=forecast_days)
    tool_wear_forecast = forecast_tool_wear(historical_data, steps=forecast_days)
    
    # Get the last timestamp from historical data
    last_timestamp = pd.to_datetime(historical_data['timestamp'].iloc[-1])
    
    # Build forecast list
    forecast_list = []
    for i in range(forecast_days):
        # Calculate future date (1 day increment for daily monitoring)
        future_date = last_timestamp + timedelta(days=i + 1)
        
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
