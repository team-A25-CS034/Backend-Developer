"""
Forecasting utilities for sensor data prediction
Using ForecasterRecursive for all parameters
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
    # Auto-calculate lags if not provided
    if lags is None:
        lags = min(7, len(data) - 1)  # Use 7 days of history or less
    else:
        lags = min(lags, len(data) - 1)
    
    # Create and fit forecaster
    forecaster = ForecasterRecursive(
        regressor=LinearRegression(),
        lags=lags
    )
    forecaster.fit(y=data)
    
    # Generate predictions
    predictions = forecaster.predict(steps=steps)
    predictions_array = predictions.values
    
    # Apply clipping if specified
    if clip_min is not None:
        predictions_array = np.maximum(predictions_array, clip_min)
    
    return predictions_array


def forecast_air_temperature(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """Forecast air temperature"""
    return forecast_parameter(
        data=historical_data["air_temperature"],
        steps=steps,
        lags=7  # Use 1 week of history
    )


def forecast_process_temperature(
    historical_data: pd.DataFrame, 
    air_temp_forecast: np.ndarray,
    steps: int = 7
) -> np.ndarray:
    """
    Forecast process temperature (relative to air temperature)
    Process temp is typically 10K higher than air temp
    """
    # Calculate relative process temperature
    df = historical_data.copy()
    process_temp_relative = df["process_temperature"] - df["air_temperature"]
    
    # Forecast the relative difference
    relative_forecast = forecast_parameter(
        data=process_temp_relative,
        steps=steps,
        lags=7
    )
    
    # Add back the forecasted air temperature
    predictions = relative_forecast + air_temp_forecast
    
    return predictions


def forecast_rotational_speed(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """Forecast rotational speed"""
    return forecast_parameter(
        data=historical_data["rotational_speed"],
        steps=steps,
        lags=30,  # Use more history for stable patterns
        clip_min=0  # Speed cannot be negative
    )


def forecast_torque(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """Forecast torque"""
    return forecast_parameter(
        data=historical_data["torque"],
        steps=steps,
        lags=30,  # Use more history for stable patterns
        clip_min=0  # Torque cannot be negative
    )


def forecast_tool_wear(historical_data: pd.DataFrame, steps: int = 7) -> np.ndarray:
    """
    Forecast tool wear - monotonically increasing
    Tool wear increases over time, never decreases
    """
    df = historical_data.copy()
    
    # Forecast using ForecasterRecursive
    predictions = forecast_parameter(
        data=df["tool_wear"],
        steps=steps,
        lags=14,  # Use 2 weeks of history
        clip_min=df["tool_wear"].iloc[-1]  # Never go below last known value
    )
    
    # Ensure monotonic increase
    last_value = df["tool_wear"].iloc[-1]
    for i in range(len(predictions)):
        if predictions[i] < last_value:
            predictions[i] = last_value
        last_value = predictions[i]
    
    return predictions


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
    
    # Forecast each parameter using ForecasterRecursive
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
