from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .vector_store import load_vector_store
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a supply chain risk analyst. Answer ONLY using the context provided below.

STRICT RULES:
- ONLY use information from the context below — never use outside knowledge
- NEVER invent, fabricate, or hallucinate sources, references, or citations
- If the context does not contain enough information, say "The available data does not cover this specifically"
- Always cite the actual source name and date from the context metadata
- Quantify claims with exact numbers from the context when available
- Distinguish between model-predicted stress scores and news-reported events

Context:
{context}

Question: {question}

Answer (only use the context above, cite real sources, no invented references):"""


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return ChatGroq(
            model="llama-3.3-70b-versatile",  # updated — 3.1 was decommissioned
            temperature=0.1,
            max_tokens=1024,
            groq_api_key=api_key,
        )
    raise ValueError("No LLM key found. Set GROQ_API_KEY in .env")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain():
    vs        = load_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    llm       = get_llm()
    prompt    = PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT,
    )
    # Use LCEL chain — works with all modern langchain versions
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info(f"QA chain ready: {type(llm).__name__}")
    return {"chain": chain, "retriever": retriever}


def ask(question: str, chain_bundle: dict) -> dict:
    chain    = chain_bundle["chain"]
    retriever = chain_bundle["retriever"]

    # Get answer
    answer = chain.invoke(question)

    # Get source documents separately
    source_docs = retriever.invoke(question)
    sources = [
        {
            "source": d.metadata.get("source", ""),
            "type":   d.metadata.get("type", ""),
            "url":    d.metadata.get("url", ""),
            "date":   d.metadata.get("published_at", ""),
        }
        for d in source_docs
    ]
    return {"answer": answer, "sources": sources}