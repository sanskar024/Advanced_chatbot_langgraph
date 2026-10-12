from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from ..retrieval import get_retriever, thread_document_metadata

_SNIPPET_CHARS = 320


@tool
def rag_tool(query: str, config: RunnableConfig, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant passages from the PDFs uploaded in this chat, using
    hybrid BM25 + FAISS search with cross-encoder reranking. Results are
    numbered; cite them in your answer as [1], [2], ... Multiple documents
    may be indexed, so each result names its source file and page.
    """
    # The runtime thread id is authoritative -- the model can't know it.
    configurable = (config or {}).get("configurable", {}) or {}
    thread_id = configurable.get("thread_id") or thread_id

    retriever = get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    docs = retriever.invoke(query)
    context, citations, metadata = [], [], []
    for i, doc in enumerate(docs, 1):
        md = doc.metadata or {}
        file = md.get("file") or md.get("source") or "document"
        page = md.get("page_label")
        where = f"{file}, p.{page}" if page else file
        context.append(f"[{i}] ({where}) {doc.page_content}")
        citations.append(
            {
                "id": i,
                "file": file,
                "page": page,
                "snippet": doc.page_content[:_SNIPPET_CHARS],
            }
        )
        metadata.append(md)

    return {
        "query": query,
        "context": context,
        "citations": citations,
        "metadata": metadata,
        "source_files": thread_document_metadata(thread_id).get("files", []),
    }
