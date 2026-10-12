"""
Hybrid RAG retrieval: BM25 (sparse) + FAISS (dense) combined via an
EnsembleRetriever, then narrowed by a cross-encoder reranker.

One index is kept per chat thread and supports MULTIPLE documents: every
uploaded PDF is chunked, tagged with its filename + page, and added to the
thread's shared FAISS store / BM25 corpus, so a single question can draw on
(and cite) any of them.
"""
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .llm import cross_encoder, embeddings

TOP_K = 8          # candidates per retrieval leg
RERANK_TOP_N = 4   # chunks passed to the LLM
BM25_WEIGHT, FAISS_WEIGHT = 0.4, 0.6

_THREADS: Dict[str, Dict[str, Any]] = {}


def _state(thread_id: Optional[str]) -> Optional[Dict[str, Any]]:
    return _THREADS.get(str(thread_id)) if thread_id else None


def get_retriever(thread_id: Optional[str]):
    st = _state(thread_id)
    return st["retriever"] if st else None


def thread_has_document(thread_id: str) -> bool:
    return _state(thread_id) is not None


def thread_documents(thread_id: str) -> List[dict]:
    """Per-file summaries for the thread: filename, pages, chunks, uploaded_at."""
    st = _state(thread_id)
    return [dict(d) for d in st["docs"].values()] if st else []


def thread_document_metadata(thread_id: str) -> dict:
    """Aggregate summary (kept backwards compatible: 'filename' is the most
    recently added file)."""
    docs = thread_documents(thread_id)
    if not docs:
        return {}
    return {
        "filename": docs[-1]["filename"],
        "files": [d["filename"] for d in docs],
        "documents": sum(d["documents"] for d in docs),
        "chunks": sum(d["chunks"] for d in docs),
    }


def thread_chunks(thread_id: str) -> list:
    st = _state(thread_id)
    return list(st["chunks"]) if st else []


def _rebuild(st: Dict[str, Any]) -> None:
    faiss_retriever = st["vector_store"].as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )
    bm25_retriever = BM25Retriever.from_documents(st["chunks"])
    bm25_retriever.k = TOP_K

    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[BM25_WEIGHT, FAISS_WEIGHT],
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=RERANK_TOP_N)
    st.update(
        bm25=bm25_retriever,
        faiss=faiss_retriever,
        hybrid=hybrid,
        retriever=ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=hybrid
        ),
    )


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Add a PDF to the thread's hybrid index (creating it on first upload).
    Returns a summary dict for the UI. Re-adding a filename that is already
    indexed is a no-op.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    key = str(thread_id)
    st = _THREADS.get(key)
    if st and filename in st["docs"]:
        return dict(st["docs"][filename])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        display_name = os.path.basename(filename or temp_path)
        pages = PyPDFLoader(temp_path).load()
        for page in pages:
            page.metadata["file"] = display_name
            page.metadata["source"] = display_name
            page.metadata["page_label"] = int(page.metadata.get("page", 0)) + 1

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(pages)
        if not chunks:
            raise ValueError("No extractable text found in the uploaded PDF.")

        if st is None:
            st = {
                "chunks": [],
                "docs": {},
                "vector_store": FAISS.from_documents(chunks, embeddings),
            }
            _THREADS[key] = st
        else:
            st["vector_store"].add_documents(chunks)
        st["chunks"].extend(chunks)
        _rebuild(st)

        summary = {
            "filename": display_name,
            "documents": len(pages),
            "chunks": len(chunks),
            "uploaded_at": time.time(),
        }
        st["docs"][display_name] = summary
        return dict(summary)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _row(doc, score=None) -> dict:
    md = doc.metadata or {}
    return {
        "file": md.get("file") or md.get("source") or "document",
        "page": md.get("page_label"),
        "text": doc.page_content,
        "score": None if score is None else float(score),
    }


def debug_retrieve(thread_id: str, query: str, k: int = 5) -> Dict[str, List[dict]]:
    """Run each stage of the pipeline separately (for the Retrieval Lab UI):
    BM25 only, FAISS only, the weighted hybrid merge, and the final
    cross-encoder-reranked result."""
    st = _state(thread_id)
    if not st:
        return {}

    bm25_docs = st["bm25"].invoke(query)[:k]
    faiss_hits = st["vector_store"].similarity_search_with_score(query, k=k)  # L2 distance
    hybrid_docs = st["hybrid"].invoke(query)

    pairs = [(query, d.page_content) for d in hybrid_docs]
    scores = list(cross_encoder.score(pairs)) if pairs else []
    ranked = sorted(zip(hybrid_docs, scores), key=lambda x: x[1], reverse=True)[:RERANK_TOP_N]

    return {
        "bm25": [_row(d) for d in bm25_docs],
        "faiss": [_row(d, s) for d, s in faiss_hits],
        "hybrid": [_row(d) for d in hybrid_docs[:k]],
        "reranked": [_row(d, s) for d, s in ranked],
    }
