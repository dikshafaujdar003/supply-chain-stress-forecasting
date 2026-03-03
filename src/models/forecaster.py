import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from loguru import logger


def forecast_stress_index(daily_sentiment: pd.DataFrame,
                           freight_df: pd.DataFrame,
                           horizon_days: int = 30) -> pd.DataFrame:
    """
    Forecast aggregate supply chain stress using stock-derived stress scores.
    Falls back to sentiment if stock data unavailable.
    """
    # ── Try loading stock-based stress time series (much richer than sentiment) ──
    stock_stress_path = Path("data") / "processed" / "master_features.parquet"
    if stock_stress_path.exists():
        logger.info("Building forecast from stock stress data (2 years of daily data)")
        return _forecast_from_stock_data(freight_df, horizon_days)

    # ── Fallback: sentiment-based ──
    logger.info("Falling back to sentiment-based forecast")
    return _forecast_from_sentiment(daily_sentiment, freight_df, horizon_days)


def _forecast_from_stock_data(freight_df: pd.DataFrame,
                               horizon_days: int = 30) -> pd.DataFrame:
    """Use aggregated stock stress scores as the time series"""
    master = pd.read_parquet(Path("data") / "processed" / "master_features.parquet")

    # Aggregate daily stress across all companies
    master.index = pd.to_datetime(master.index).tz_localize(None)
    daily = (master.groupby(master.index)["stock_stress"]
             .mean()
             .reset_index())
    # rename whatever the index column is called to "date"
    daily.columns = ["date", "stress_index"]
    daily = daily.sort_values("date").reset_index(drop=True)

    # Merge macro regressor
    freight_df = freight_df.copy()
    freight_df.index = pd.to_datetime(freight_df.index).tz_localize(None)
    macro_col = next(
        (c for c in ["producer_prices", "shipping_costs", "industrial_output"]
         if c in freight_df.columns), None
    )
    if macro_col:
        macro = freight_df[[macro_col]].reset_index()
        macro.columns = ["date", "macro"]
        macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
        daily = daily.merge(macro, on="date", how="left").ffill().bfill()
        logger.info(f"Using '{macro_col}' as macro regressor")
    else:
        daily["macro"] = 0.0

    return _run_ridge_forecast(daily, horizon_days)


def _forecast_from_sentiment(daily_sentiment: pd.DataFrame,
                              freight_df: pd.DataFrame,
                              horizon_days: int = 30) -> pd.DataFrame:
    agg = daily_sentiment[["date", "avg_sentiment"]].copy()
    agg["date"] = pd.to_datetime(agg["date"]).dt.tz_localize(None)
    agg = agg.rename(columns={"avg_sentiment": "stress_index"})
    agg["stress_index"] = -agg["stress_index"]
    agg["macro"] = 0.0
    return _run_ridge_forecast(agg.sort_values("date").reset_index(drop=True),
                               horizon_days)


def _run_ridge_forecast(daily: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Core Ridge regression forecaster with lag features"""
    daily = daily.copy()
    daily["lag_1"]         = daily["stress_index"].shift(1)
    daily["lag_7"]         = daily["stress_index"].shift(7)
    daily["lag_14"]        = daily["stress_index"].shift(14)
    daily["rolling_mean_7"] = daily["stress_index"].rolling(7).mean()
    daily["rolling_std_7"]  = daily["stress_index"].rolling(7).std()
    daily["day_of_week"]    = pd.to_datetime(daily["date"]).dt.dayofweek
    daily["day_of_month"]   = pd.to_datetime(daily["date"]).dt.day
    daily = daily.dropna()

    if len(daily) < 10:
        logger.warning("Not enough data — using flat extrapolation")
        last_val  = daily["stress_index"].iloc[-1] if len(daily) > 0 else 0.3
        last_date = pd.to_datetime(daily["date"].iloc[-1]) if len(daily) > 0 else pd.Timestamp.today()
        rows = [{
            "ds": last_date + pd.Timedelta(days=i),
            "yhat": last_val, "yhat_lower": last_val - 0.05,
            "yhat_upper": last_val + 0.05, "trend": last_val,
        } for i in range(1, horizon_days + 1)]
        return pd.DataFrame(rows)

    feature_cols = ["lag_1", "lag_7", "lag_14", "rolling_mean_7",
                    "rolling_std_7", "macro", "day_of_week", "day_of_month"]
    X = daily[feature_cols].values
    y = daily["stress_index"].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model   = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    logger.info(f"Ridge model trained on {len(daily)} days of stress data")

    # Historical fitted values
    historical = []
    for _, row in daily.iterrows():
        historical.append({
            "ds":         pd.to_datetime(row["date"]),
            "yhat":       row["stress_index"],
            "yhat_lower": row["stress_index"] - 0.02,
            "yhat_upper": row["stress_index"] + 0.02,
            "trend":      row["rolling_mean_7"],
        })

    # Future forecast
    history   = list(daily["stress_index"].values)
    last_macro = daily["macro"].iloc[-1]
    last_date  = pd.to_datetime(daily["date"].iloc[-1])
    forecast  = []

    for i in range(1, horizon_days + 1):
        future_date = last_date + pd.Timedelta(days=i)
        lag1  = history[-1]
        lag7  = history[-7]  if len(history) >= 7  else history[0]
        lag14 = history[-14] if len(history) >= 14 else history[0]
        roll_mean = float(np.mean(history[-7:]))
        roll_std  = float(np.std(history[-7:]))

        x_new   = np.array([[lag1, lag7, lag14, roll_mean, roll_std,
                              last_macro, future_date.dayofweek, future_date.day]])
        pred    = model.predict(scaler.transform(x_new))[0]
        std     = float(np.std(history[-14:])) if len(history) >= 14 else 0.05

        forecast.append({
            "ds":         future_date,
            "yhat":       float(pred),
            "yhat_lower": float(pred - 1.5 * std),
            "yhat_upper": float(pred + 1.5 * std),
            "trend":      roll_mean,
        })
        history.append(pred)

    result = pd.DataFrame(historical + forecast)
    logger.info(f"Forecast: {len(historical)} historical + {len(forecast)} future points")
    return result