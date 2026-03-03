"""Run this to build everything. Re-run to refresh data."""
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

for p in ["data/raw", "data/processed", "data/embeddings"]:
    Path(p).mkdir(parents=True, exist_ok=True)

from src.ingestion.stock_data import fetch_stock_data, get_all_tickers
from src.ingestion.freight_data import fetch_fred_indicators
from src.ingestion.news_scraper import fetch_all_supply_chain_news
from src.processing.sentiment import load_sentiment_model, score_articles, aggregate_daily_sentiment
from src.processing.graph_builder import build_graph, propagate_stress, COMPANY_METADATA
from src.processing.feature_engineering import build_master_feature_set
from src.models.stress_classifier import train_stress_classifier, predict_current_stress
from src.models.forecaster import forecast_stress_index
from src.rag.embedder import articles_to_documents, analysis_to_documents, chunk_documents
from src.rag.vector_store import build_vector_store
from langchain_core.documents import Document
import pandas as pd


def run_pipeline(rebuild_rag: bool = True):
    logger.info("=== Supply Chain Stress Pipeline Starting ===")

    logger.info("Step 1/8: Fetching stock data...")
    stock_df = fetch_stock_data(get_all_tickers(), period="2y")
    stock_df.to_parquet("data/processed/stock_data.parquet")

    logger.info("Step 2/8: Fetching FRED indicators...")
    freight_df = fetch_fred_indicators()
    freight_df.to_parquet("data/processed/freight_data.parquet")

    logger.info("Step 3/8: Fetching news...")
    news_df = fetch_all_supply_chain_news(days_back=28)
    news_df.to_parquet("data/processed/news_raw.parquet")

    logger.info("Step 4/8: Running FinBERT sentiment (~5 min)...")
    pipe = load_sentiment_model()
    news_df = score_articles(news_df, pipe)
    daily_sentiment = aggregate_daily_sentiment(news_df)
    daily_sentiment.to_parquet("data/processed/daily_sentiment.parquet")

    logger.info("Step 5/8: Building features and training model...")
    master_df = build_master_feature_set(stock_df, freight_df, daily_sentiment)
    master_df.to_parquet("data/processed/master_features.parquet")
    train_stress_classifier(master_df)

    logger.info("Step 6/8: Computing stress scores + graph propagation...")
    current_stress = predict_current_stress(master_df)
    meta_df = (pd.DataFrame(COMPANY_METADATA).T
                 .reset_index()
                 .rename(columns={"index": "ticker"}))
    current_stress = current_stress.merge(meta_df, on="ticker", how="left")
    stress_dict    = dict(zip(current_stress["ticker"],
                              current_stress["stress_probability"]))
    G          = build_graph(stress_dict)
    propagated = propagate_stress(G, stress_dict)
    current_stress["propagated_stress"] = current_stress["ticker"].map(propagated)
    current_stress.to_parquet("data/processed/current_stress.parquet")

    logger.info("Step 7/8: Forecasting 30-day stress index...")
    forecast_df = forecast_stress_index(daily_sentiment, freight_df)
    forecast_df.to_parquet("data/processed/forecast.parquet")

    if rebuild_rag:
        logger.info("Step 8/8: Building RAG vector store...")
        docs  = articles_to_documents(news_df)
        docs += analysis_to_documents(current_stress)

        # Add forecast summary as a queryable document
        forecast_recent = forecast_df.tail(30)
        trend     = "rising" if forecast_recent["yhat"].iloc[-1] > forecast_recent["yhat"].iloc[0] else "falling"
        avg       = forecast_recent["yhat"].mean()
        peak      = forecast_recent["yhat"].max()
        peak_date = forecast_recent.loc[forecast_recent["yhat"].idxmax(), "ds"]
        forecast_text = (
            f"30-day supply chain stress forecast summary: "
            f"The aggregate stress index is expected to be {trend} over the next 30 days. "
            f"Average forecasted stress: {avg:.3f}. "
            f"Peak stress of {peak:.3f} expected around {str(peak_date)[:10]}. "
            f"The forecast is based on 2 years of stock price stress signals from 25 companies "
            f"across semiconductors, logistics, automotive, retail, and raw materials sectors, "
            f"combined with FRED macro indicators including producer prices and industrial output."
        )
        docs.append(Document(
            page_content=forecast_text,
            metadata={"source": "stress_forecast_model", "type": "forecast"}
        ))

        build_vector_store(chunk_documents(docs))

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    run_pipeline(rebuild_rag=True)