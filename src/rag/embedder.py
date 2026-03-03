from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
from loguru import logger


def articles_to_documents(news_df: pd.DataFrame) -> list:
    docs = []
    for _, row in news_df.iterrows():
        if not row.get("full_text"):
            continue
        docs.append(Document(
            page_content=row["full_text"],
            metadata={
                "source":       row.get("source", ""),
                "url":          row.get("url", ""),
                "published_at": str(row.get("published_at", "")),
                "query":        row.get("query", ""),
                "type":         "news_article",
            }
        ))
    logger.info(f"Created {len(docs)} documents from news")
    return docs


def sec_text_to_documents(excerpts: list, ticker: str) -> list:
    return [
        Document(
            page_content=text,
            metadata={"source": f"SEC 10-K {ticker}", "type": "sec_filing", "ticker": ticker}
        )
        for text in excerpts if text.strip()
    ]


def analysis_to_documents(stress_df: pd.DataFrame) -> list:
    docs = []
    for _, row in stress_df.iterrows():
        text = (
            f"Supply chain stress analysis for {row.get('ticker','N/A')} "
            f"({row.get('name','')}): stress probability is "
            f"{row.get('stress_probability',0):.1%}. "
            f"Sector: {row.get('sector','')}. Tier: {row.get('tier','')}. "
            f"Drawdown: {row.get('drawdown',0):.1%}. "
            f"Propagated network stress: {row.get('propagated_stress',0):.1%}."
        )
        docs.append(Document(
            page_content=text,
            metadata={"ticker": row.get("ticker", ""), "type": "stress_analysis"}
        ))
    return docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)