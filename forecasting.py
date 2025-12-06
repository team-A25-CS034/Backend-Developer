import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursive
from sklearn.svm import SVR
from typing import Dict, List

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
}

REQUIRED_COLS = [AIR_TEMPERATURE, PROCESS_TEMPERATURE, ROTATIONAL_SPEED, TORQUE, TOOL_WEAR]
DROPPABLE_COLS = ['UDI', 'Product ID', 'Type', 'Target', 'Failure Type', '_id']


def prepare_forecast_data(readings: List[Dict]) -> pd.DataFrame:
    """Convert MongoDB readings to DataFrame for forecasting
    
    Supports both legacy schema (air_temperature, process_temperature, etc.)
    and forecaster_input.csv schema (Air temperature [K], Process temperature [K], etc.)
    """
    df = pd.DataFrame(readings)
    logger.info("Loaded historical readings: rows=%s, cols=%s", len(df), list(df.columns))
    return df

def generate_forecast(
    historical_data: pd.DataFrame,
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

    df = historical_data.copy()

    if df.empty:
        logger.error("Historical data is empty; cannot generate forecast")
        raise ValueError("No historical data available for forecasting")

    # Drop unnecessary columns if they exist
    df = df.drop(columns=[c for c in DROPPABLE_COLS if c in df.columns], errors='ignore')

    df = df.rename(columns=SENSOR_COLUMN_MAP)

    # Keep only the sensor columns we actually forecast; this drops string/object cols that break aggregations
    df = df[[c for c in REQUIRED_COLS if c in df.columns]]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure numeric types for all required columns
    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[TOOL_WEAR])
    logger.info("Cleaned frame columns=%s", list(df.columns))

    duplicate_count = df[TOOL_WEAR].duplicated().sum()
    if duplicate_count:
        logger.warning("Duplicate Tool wear values detected: %s. Aggregating by mean to continue.", duplicate_count)
        df = df.groupby(TOOL_WEAR, as_index=False).mean()

    df = df.sort_values(TOOL_WEAR)
    df = df.set_index(TOOL_WEAR)
    full_index = pd.Index(range(int(df.index.max()) + 1), name=TOOL_WEAR)
    df = df.reindex(full_index)
    logger.info("Prepared forecasting frame: rows=%s, cols=%s", df.shape[0], df.shape[1])

    _, lower_bound, upper_bound = detect_outliers_iqr(df, AIR_TEMPERATURE)
    df.loc[df[AIR_TEMPERATURE] < lower_bound, AIR_TEMPERATURE] = lower_bound
    df.loc[df[AIR_TEMPERATURE] > upper_bound, AIR_TEMPERATURE] = upper_bound

    df[AIR_TEMPERATURE] = df[AIR_TEMPERATURE].interpolate()
    df = df.reset_index(drop=True)
    x = pd.DataFrame(list(df.index), columns=["Index"])
    y = df[AIR_TEMPERATURE]

    svr = SVR(kernel='rbf')
    svr.fit(x, y)
    y_pred = svr.predict(x)

    train = y_pred

    forecaster = ForecasterRecursive(
                    regressor = LinearRegression(),
                    lags      = 1
                )
    forecaster.fit(y=pd.Series(train, name=AIR_TEMPERATURE))

    predictions = forecaster.predict(steps=steps)
    predictions[predictions < 294] = 294
    predictions[predictions > 306] = 306
    predictions = predictions.rename(AIR_TEMPERATURE)
    output_df = pd.DataFrame(predictions)

    _, lower_bound, upper_bound = detect_outliers_iqr(df, PROCESS_TEMPERATURE)
    df.loc[df[PROCESS_TEMPERATURE] < lower_bound, PROCESS_TEMPERATURE] = lower_bound
    df.loc[df[PROCESS_TEMPERATURE] > upper_bound, PROCESS_TEMPERATURE] = upper_bound

    process_temperature_ori = df[PROCESS_TEMPERATURE] - df[AIR_TEMPERATURE] - 10
    df[PROCESS_TEMPERATURE] = process_temperature_ori.interpolate()
    x = pd.DataFrame(list(df.index), columns=["Index"])
    y = df[PROCESS_TEMPERATURE]

    svr = SVR(kernel='rbf')
    svr.fit(x, y)
    y_pred = svr.predict(x)

    train = y_pred
    forecaster = ForecasterRecursive(
                    regressor = LinearRegression(),
                    lags      = 1
                )
    forecaster.fit(y=pd.Series(train, name=PROCESS_TEMPERATURE))

    predictions = forecaster.predict(steps=steps)
    predictions[predictions < -3] = -3
    predictions[predictions > 3] = 3
    predictions = predictions.values + output_df["Air temperature"].values + 10
    predictions = pd.Series(predictions, name=PROCESS_TEMPERATURE, index=output_df.index)
    output_df = pd.concat([output_df, predictions], axis=1)

    _, lower_bound, upper_bound = detect_outliers_iqr(df, ROTATIONAL_SPEED)
    df.loc[df[ROTATIONAL_SPEED] < lower_bound, ROTATIONAL_SPEED] = lower_bound
    df.loc[df[ROTATIONAL_SPEED] > upper_bound, ROTATIONAL_SPEED] = upper_bound

    df[ROTATIONAL_SPEED] = df[ROTATIONAL_SPEED].interpolate()
    y = df[ROTATIONAL_SPEED]
    y_pred = np.mean(y) * np.ones_like(y)

    train = y_pred
    forecaster = ForecasterRecursive(
                    regressor = LinearRegression(),
                    lags      = 1
                )
    forecaster.fit(y=pd.Series(train, name=ROTATIONAL_SPEED))

    predictions = forecaster.predict(steps=steps)
    predictions = predictions.rename(ROTATIONAL_SPEED)
    output_df = pd.concat([output_df, predictions], axis=1)

    _, lower_bound, upper_bound = detect_outliers_iqr(df, TORQUE)
    df.loc[df[TORQUE] < lower_bound, TORQUE] = lower_bound
    df.loc[df[TORQUE] > upper_bound, TORQUE] = upper_bound

    df[TORQUE] = df[TORQUE].interpolate()
    y = df[TORQUE]
    y_pred = 40 * np.ones_like(y)

    train = y_pred
    forecaster = ForecasterRecursive(
                    regressor = LinearRegression(),
                    lags      = 1
                )
    forecaster.fit(y=pd.Series(train, name=TORQUE))

    predictions = forecaster.predict(steps=steps)
    predictions[predictions < 0] = 0
    predictions = predictions.rename(TORQUE)
    output_df = pd.concat([output_df, predictions], axis=1)

    indices = list(output_df.index)
    indices = pd.Series(indices, name=TOOL_WEAR, index=output_df.index)
    output_df = pd.concat([output_df, indices], axis=1)
    output_df = output_df.reset_index(drop=True)

    forecast_list = []
    for i in range(steps):
        forecast_dict = {
            "air_temperature": float(output_df[AIR_TEMPERATURE][i]),
            "process_temperature": float(output_df[PROCESS_TEMPERATURE][i]),
            "rotational_speed": float(output_df[ROTATIONAL_SPEED][i]),
            "torque": float(output_df[TORQUE][i]),
            "tool_wear": float(output_df[TOOL_WEAR][i]),
        }

        forecast_list.append(forecast_dict)

        print(forecast_dict)

    return forecast_list

def detect_outliers_iqr(df, column):    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound
