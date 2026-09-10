"""
Hybrid RAG retrieval: BM25 (sparse) + FAISS (dense) combined via an
EnsembleRetriever, then narrowed by a cross-encoder reranker. One retriever
is built and cached per chat thread, keyed by thread_id.
"""
import os
import tempfile
from typing import Any, Dict, Optional

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .llm import cross_encoder, embeddings

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a hybrid BM25 + FAISS retriever (with cross-encoder reranking) for
    the uploaded PDF and cache it for the thread. Returns a summary dict
    that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No extractable text found in the uploaded PDF.")

        # Dense retrieval leg (FAISS over sentence-transformer embeddings)
        vector_store = FAISS.from_documents(chunks, embeddings)
        faiss_retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 8}
        )

        # Sparse retrieval leg (BM25 keyword search)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 8

        # Hybrid ensemble: merge sparse + dense candidates before reranking
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6],
        )

        # Cross-encoder reranker trims the merged candidate pool down to the
        # most relevant chunks for the final context window.
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=hybrid_retriever,
        )

        thread_key = str(thread_id)
        _THREAD_RETRIEVERS[thread_key] = retriever
        _THREAD_METADATA[thread_key] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return dict(_THREAD_METADATA[thread_key])
    finally:
        # FAISS/BM25 keep their own copies of the text, so the temp file can
        # safely be removed once indexing is done.
        try:
            os.remove(temp_path)
        except OSError:
            pass
