import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursive
from sklearn.svm import SVR
from typing import Dict, List
from datetime import datetime, timedelta


def prepare_forecast_data(readings: List[Dict]) -> pd.DataFrame:
    """Convert MongoDB readings to DataFrame for forecasting
    
    Supports both legacy schema (air_temperature, process_temperature, etc.)
    and forecaster_input.csv schema (Air temperature [K], Process temperature [K], etc.)
    """
    df = pd.DataFrame(readings)

    
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

    df = historical_data

    # Drop unnecessary columns
    df = df.drop(['UDI'], axis=1)
    df = df.drop(['Product ID'], axis=1)
    df = df.drop(['Type'], axis=1)
    df = df.drop(['Target'], axis=1)
    df = df.drop(['Failure Type'], axis=1)

    df = df.rename(columns={'Air temperature [K]': 'Air temperature',
                                'Process temperature [K]': 'Process temperature',
                                'Rotational speed [rpm]': 'Rotational speed',
                                'Torque [Nm]': 'Torque',
                                'Tool wear [min]': 'Tool wear',
                                })

    df = df.set_index("Tool wear")
    full_index = pd.Index(range(df.index.max() + 1), name="Tool wear")
    df = df.reindex(full_index)

    _, lower_bound, upper_bound = detect_outliers_iqr(df, "Air temperature")
    df.loc[df["Air temperature"] < lower_bound, 'Air temperature'] = lower_bound
    df.loc[df["Air temperature"] > upper_bound, 'Air temperature'] = upper_bound

    df["Air temperature"] = df["Air temperature"].interpolate()
    df = df.reset_index(drop=True)
    x = pd.DataFrame(list(df.index), columns=["Index"])
    y = df["Air temperature"]

    svr = SVR(kernel='rbf')
    svr.fit(x, y)
    y_pred = svr.predict(x)

    train = y_pred

    forecaster = ForecasterRecursive(
                    regressor = LinearRegression(),
                    lags      = 1
                )
    forecaster.fit(y=pd.Series(train, name="Air temperature"))

    predictions = forecaster.predict(steps=steps)
    predictions[predictions < 294] = 294
    predictions[predictions > 306] = 306
    predictions = predictions.rename('Air temperature')
    output_df = pd.DataFrame(predictions)

    _, lower_bound, upper_bound = detect_outliers_iqr(df, "Process temperature")
    df.loc[df["Process temperature"] < lower_bound, 'Process temperature'] = lower_bound
    df.loc[df["Process temperature"] > upper_bound, 'Process temperature'] = upper_bound

    process_temperature_ori = df["Process temperature"] - df["Air temperature"] - 10
    df["Process temperature"] = process_temperature_ori.interpolate()
    x = pd.DataFrame(list(df.index), columns=["Index"])
    y = df["Process temperature"]

    svr = SVR(kernel='rbf')
    svr.fit(x, y)
    y_pred = svr.predict(x)

    train = y_pred
    forecaster = ForecasterRecursive(
                    regressor = LinearRegression(),
                    lags      = 1
                )
    forecaster.fit(y=pd.Series(train, name="Process temperature"))

    predictions = forecaster.predict(steps=steps)
    predictions[predictions < -3] = -3
    predictions[predictions > 3] = 3
    predictions = predictions.values + output_df["Air temperature"].values + 10
    predictions = pd.Series(predictions, name='Process temperature', index=output_df.index)
    output_df = pd.concat([output_df, predictions], axis=1)

    _, lower_bound, upper_bound = detect_outliers_iqr(df, "Rotational speed")
    df.loc[df["Rotational speed"] < lower_bound, 'Rotational speed'] = lower_bound
    df.loc[df["Rotational speed"] > upper_bound, 'Rotational speed'] = upper_bound

    df["Rotational speed"] = df["Rotational speed"].interpolate()
    y = df["Rotational speed"]
    y_pred = np.mean(y) * np.ones_like(y)

    train = y_pred
    forecaster = ForecasterRecursive(
                    regressor = LinearRegression(),
                    lags      = 1
                )
    forecaster.fit(y=pd.Series(train, name="Rotational speed"))

    predictions = forecaster.predict(steps=steps)
    predictions = predictions.rename('Rotational speed')
    output_df = pd.concat([output_df, predictions], axis=1)

    _, lower_bound, upper_bound = detect_outliers_iqr(df, "Torque")
    df.loc[df["Torque"] < lower_bound, 'Torque'] = lower_bound
    df.loc[df["Torque"] > upper_bound, 'Torque'] = upper_bound

    df["Torque"] = df["Torque"].interpolate()
    y = df["Torque"]
    y_pred = 40 * np.ones_like(y)

    train = y_pred
    forecaster = ForecasterRecursive(
                    regressor = LinearRegression(),
                    lags      = 1
                )
    forecaster.fit(y=pd.Series(train, name="Torque"))

    predictions = forecaster.predict(steps=steps)
    predictions[predictions < 0] = 0
    predictions = predictions.rename('Torque')
    output_df = pd.concat([output_df, predictions], axis=1)

    indices = list(output_df.index)
    indices = pd.Series(indices, name='Tool wear', index=output_df.index)
    output_df = pd.concat([output_df, indices], axis=1)
    output_df = output_df.reset_index(drop=True)

    forecast_list = []
    for i in range(steps):
        forecast_dict = {
            "air_temperature": float(output_df['Air temperature'][i]),
            "process_temperature": float(output_df['Process temperature'][i]),
            "rotational_speed": float(output_df['Rotational speed'][i]),
            "torque": float(output_df['Torque'][i]),
            "tool_wear": float(output_df['Tool wear'][i]),
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
