import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

SUPPLY_CHAIN_QUERIES = [
    "supply chain disruption",
    "port congestion shipping delay",
    "semiconductor shortage",
    "freight rates logistics",
    "trade tariff import export",
    "factory shutdown manufacturing",
    "supplier bankruptcy",
    "raw material shortage",
]

def check_api_key():
    """Verify the API key works before running all queries"""
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY not found in .env file")
    r = requests.get(
        "https://newsapi.org/v2/everything",
        params={"q": "supply chain", "pageSize": 1, "apiKey": NEWS_API_KEY}
    )
    data = r.json()
    if data.get("status") == "error":
        raise ValueError(f"NewsAPI error: {data.get('message')} — check your NEWS_API_KEY")
    logger.info(f"NewsAPI key verified — total results available: {data.get('totalResults', 0)}")

def fetch_news(query: str, days_back: int = 30) -> list:
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "relevancy",
        "language": "en",
        "pageSize": 100,
        "apiKey":   NEWS_API_KEY,
    }
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        data = r.json()
        if data.get("status") == "error":
            logger.warning(f"NewsAPI error for '{query}': {data.get('message')}")
            return []
        articles = data.get("articles", [])
        logger.info(f"Fetched {len(articles)} articles for: {query}")
        return articles
    except Exception as e:
        logger.warning(f"Request failed for '{query}': {e}")
        return []

def fetch_all_supply_chain_news(days_back: int = 28) -> pd.DataFrame:
    # Verify key first
    check_api_key()

    all_articles = []
    for query in SUPPLY_CHAIN_QUERIES:
        for a in fetch_news(query, days_back):
            all_articles.append({
                "title":        a.get("title", "") or "",
                "description":  a.get("description", "") or "",
                "content":      a.get("content", "") or "",
                "source":       a.get("source", {}).get("name", ""),
                "published_at": a.get("publishedAt", ""),
                "url":          a.get("url", ""),
                "query":        query,
                "full_text":    f"{a.get('title','')}. {a.get('description','')}. {a.get('content','')}",
            })

    if not all_articles:
        logger.warning("No news articles fetched — using placeholder data for pipeline to continue")
        # Return a minimal placeholder so the pipeline doesn't crash
        return pd.DataFrame([{
            "title":        "Supply chain monitoring active",
            "description":  "No recent articles fetched",
            "content":      "Supply chain stress monitoring system initialized",
            "source":       "system",
            "published_at": datetime.now().isoformat(),
            "url":          "",
            "query":        "supply chain",
            "full_text":    "Supply chain stress monitoring system initialized",
        }])

    df = pd.DataFrame(all_articles).drop_duplicates(subset="url")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    logger.info(f"Total unique articles: {len(df)}")
    return df

def fetch_sec_filings_text(ticker: str) -> list:
    from sec_edgar_downloader import Downloader
    import re
    base_dir = Path("data") / "raw" / "sec"
    base_dir.mkdir(parents=True, exist_ok=True)
    dl = Downloader("MyCompany", "myemail@example.com", str(base_dir))
    try:
        dl.get("10-K", ticker, limit=2)
        filing_dir = base_dir / "sec-edgar-filings" / ticker / "10-K"
        files = list(filing_dir.rglob("*.txt"))
        excerpts = []
        for f in files[:2]:
            text = f.read_text(errors="ignore")
            for p in re.split(r"\n\n+", text):
                if any(kw in p.lower() for kw in
                       ["supply chain", "supplier", "inventory", "logistics", "shortage"]):
                    excerpts.append(p[:1000])
        return excerpts[:20]
    except Exception as e:
        logger.warning(f"SEC filing fetch failed for {ticker}: {e}")
        return []