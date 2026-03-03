import pandas as pd
import numpy as np
from .graph_builder import build_graph, compute_centrality_features

def compute_stock_stress_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    features = []
    for ticker, grp in stock_df.groupby("ticker"):
        grp = grp.sort_index().copy()
        grp["rolling_vol_30"]    = grp["returns"].rolling(30).std()
        grp["rolling_vol_5"]     = grp["returns"].rolling(5).std()
        grp["vol_spike"]         = grp["rolling_vol_5"] / grp["rolling_vol_30"].replace(0, np.nan)
        grp["price_below_50ma"]  = (grp["Close"] < grp["Close"].rolling(50).mean()).astype(int)
        grp["price_below_200ma"] = (grp["Close"] < grp["Close"].rolling(200).mean()).astype(int)
        grp["cum_return_20d"]    = grp["Close"].pct_change(20)
        grp["ticker"] = ticker
        grp["stock_stress"] = (
            0.4  * grp["rolling_vol_30"].rank(pct=True) +
            0.3  * (-grp["cum_return_20d"]).clip(0).rank(pct=True) +
            0.15 * grp["price_below_50ma"] +
            0.15 * grp["price_below_200ma"]
        )
        features.append(grp)
    return pd.concat(features)

def strip_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Remove timezone info from DatetimeIndex — fixes tz-naive/tz-aware join errors"""
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def build_master_feature_set(stock_df, freight_df, sentiment_df) -> pd.DataFrame:
    stock_features = compute_stock_stress_features(stock_df)

    # Strip timezone from all dataframes before joining
    stock_features = strip_timezone(stock_features)
    stock_features.index = pd.to_datetime(stock_features.index)

    freight_df = strip_timezone(freight_df.copy())
    freight_df.index = pd.to_datetime(freight_df.index)

    G = build_graph()
    centrality = compute_centrality_features(G)
    frames = []

    for ticker in stock_features["ticker"].unique():
        t_df = stock_features[stock_features["ticker"] == ticker].copy()
        t_df = strip_timezone(t_df)

        # Join freight indicators
        t_df = t_df.join(freight_df, how="left").ffill()

        # Join sentiment
        s = sentiment_df.copy()
        s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None)
        s = s.set_index("date")
        t_df = t_df.join(s, how="left").ffill()

        # Join graph centrality features
        c_row = centrality[centrality["ticker"] == ticker]
        if not c_row.empty:
            for col in ["betweenness", "pagerank", "in_degree", "out_degree", "tier"]:
                t_df[col] = c_row[col].values[0]

        frames.append(t_df)

    master = pd.concat(frames).dropna(subset=["stock_stress"])
    master["stress_label"] = (master["stock_stress"] > 0.7).astype(int)
    return master