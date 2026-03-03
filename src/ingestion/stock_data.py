import yfinance as yf
import pandas as pd
import time
from loguru import logger

SUPPLY_CHAIN_TICKERS = {
    "semiconductors": ["TSM", "NVDA", "INTC", "AMAT", "ASML", "MU"],
    "logistics":      ["FDX", "UPS", "ZIM"],
    "manufacturing":  ["HON", "MMM", "EMR", "GE"],
    "retail":         ["WMT", "TGT", "COST", "AMZN"],
    "automotive":     ["F", "GM", "TM", "STLA"],
    "raw_materials":  ["NUE", "FCX", "AA", "CF"],
}

def fetch_single_ticker(ticker: str, period: str = "2y", retries: int = 3) -> pd.DataFrame | None:
    """Fetch one ticker with retries and a small delay to avoid rate limiting"""
    for attempt in range(retries):
        try:
            # Use Ticker object instead of yf.download — more reliable
            t = yf.Ticker(ticker)
            df = t.history(period=period)

            if df.empty:
                logger.warning(f"Empty data for {ticker}, attempt {attempt+1}/{retries}")
                time.sleep(2)
                continue

            df["ticker"]         = ticker
            df["returns"]        = df["Close"].pct_change()
            df["volatility_20d"] = df["returns"].rolling(20).std()
            df["drawdown"]       = (
                (df["Close"] - df["Close"].rolling(252).max())
                / df["Close"].rolling(252).max()
            )
            logger.info(f"Fetched {ticker} — {len(df)} rows")
            return df

        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
            time.sleep(3)

    logger.error(f"All attempts failed for {ticker}, skipping")
    return None

def fetch_stock_data(tickers: list, period: str = "2y") -> pd.DataFrame:
    all_data = []
    for i, ticker in enumerate(tickers):
        df = fetch_single_ticker(ticker, period)
        if df is not None:
            all_data.append(df)
        # Small delay every 5 tickers to avoid rate limiting
        if (i + 1) % 5 == 0:
            logger.info(f"Pausing briefly to avoid rate limits ({i+1}/{len(tickers)} done)...")
            time.sleep(2)

    if not all_data:
        raise RuntimeError("No stock data fetched at all — check your internet connection")

    combined = pd.concat(all_data)
    logger.info(f"Total: {len(all_data)}/{len(tickers)} tickers fetched successfully")
    return combined

def get_all_tickers() -> list:
    return [t for tickers in SUPPLY_CHAIN_TICKERS.values() for t in tickers]