from transformers import pipeline
import pandas as pd
import torch
from loguru import logger

MODEL_NAME = "ProsusAI/finbert"

def load_sentiment_model():
    # Force CPU on Windows to avoid CUDA issues
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=device,
        truncation=True,
        max_length=512,
    )

def score_articles(df: pd.DataFrame, pipe) -> pd.DataFrame:
    texts = df["full_text"].fillna("").tolist()
    results = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        try:
            out = pipe(batch)
            results.extend(out)
        except Exception as e:
            logger.warning(f"Batch failed: {e}")
            results.extend([{"label": "neutral", "score": 0.5}] * len(batch))

    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [
        r["score"] if r["label"] == "positive"
        else -r["score"] if r["label"] == "negative"
        else 0.0
        for r in results
    ]
    return df

def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["published_at"]).dt.date
    daily = df.groupby("date").agg(
        avg_sentiment  =("sentiment_score", "mean"),
        article_count  =("sentiment_score", "count"),
        negative_ratio =("sentiment_label", lambda x: (x == "negative").mean()),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily