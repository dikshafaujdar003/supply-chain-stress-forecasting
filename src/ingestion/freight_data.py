import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import os

load_dotenv()

FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise ValueError("FRED_API_KEY not found in .env file")

fred = Fred(api_key=FRED_KEY)

# All verified working FRED series IDs
FRED_SERIES = {
    "shipping_costs":    "WTISPLC",         # WTI crude oil price
    "import_prices":     "IR",              # Import price index
    "pmi_manufacturing": "MANEMP",          # Manufacturing employment
    "freight_transport": "FRGSHPUSM649NCIS",# Freight shipments
    "cpi_commodities":   "CUSR0000SAC",     # CPI commodities
    "producer_prices":   "PPIACO",          # Producer price index
    "real_gdp":          "GDPC1",           # Real GDP (proxy for demand)
    "industrial_output": "INDPRO",          # Industrial production index
}

def fetch_fred_indicators(start: str = "2020-01-01") -> pd.DataFrame:
    frames = {}
    for name, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start)
            frames[name] = s
            print(f"✓ Fetched FRED: {name} ({series_id})")
        except Exception as e:
            print(f"✗ Skipped FRED {name} ({series_id}): {e}")

    if not frames:
        raise RuntimeError("No FRED series fetched — check your FRED_API_KEY in .env")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.resample("D").interpolate(method="time")
    return df