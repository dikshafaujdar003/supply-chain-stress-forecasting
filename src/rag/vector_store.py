from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pathlib import Path
from loguru import logger

FAISS_PATH = Path("data") / "embeddings" / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(documents: list) -> FAISS:
    embeddings = get_embeddings()
    logger.info(f"Building FAISS vector store with {len(documents)} chunks...")
    vs = FAISS.from_documents(documents=documents, embedding=embeddings)
    FAISS_PATH.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(FAISS_PATH))
    logger.info(f"Vector store saved to {FAISS_PATH}")
    return vs


def load_vector_store() -> FAISS:
    embeddings = get_embeddings()
    if not FAISS_PATH.exists():
        raise FileNotFoundError(
            f"No vector store found at {FAISS_PATH}. Run ingest_pipeline.py first."
        )
    vs = FAISS.load_local(
        str(FAISS_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info(f"Loaded FAISS vector store from {FAISS_PATH}")
    return vs


def update_vector_store(new_docs: list):
    vs = load_vector_store()
    embeddings = get_embeddings()
    new_vs = FAISS.from_documents(documents=new_docs, embedding=embeddings)
    vs.merge_from(new_vs)
    vs.save_local(str(FAISS_PATH))
    logger.info(f"Added {len(new_docs)} new documents to vector store")