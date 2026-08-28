import os
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
import requests
import asyncio
import threading

import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import interrupt

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool, BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

import tempfile

load_dotenv()

# -------------------
# 0. Dedicated async loop for backend tasks (used for MCP tool discovery, which
#    is only exposed as an async API upstream).
# -------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

API_KEY = os.getenv("API_KEY")
API_KEY_STOCKS = os.getenv("API_KEY_STOCKS")
POSTGRES_URI = os.getenv(
    "POSTGRES_URI", "postgresql://chatbot:chatbot@localhost:5432/chatbot"
)


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM & embeddings
# -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.2,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Shared cross-encoder used to rerank the hybrid BM25 + FAISS retrieval results.
_cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# -------------------
# 2. Hybrid RAG retrieval (BM25 + FAISS dense retrieval + cross-encoder rerank)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a hybrid BM25 + FAISS retriever (with cross-encoder reranking) for the
    uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
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

        # Hybrid ensemble: combine sparse + dense candidates before reranking
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6],
        )

        # Cross-encoder reranker trims the merged candidate pool down to the
        # most relevant chunks for the final context window.
        reranker = CrossEncoderReranker(model=_cross_encoder, top_n=4)
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=hybrid_retriever,
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS/BM25 stores keep copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
_ddg_wrapper = DuckDuckGoSearchAPIWrapper(region="us-en")
search_tool = DuckDuckGoSearchRun(api_wrapper=_ddg_wrapper)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') using
    Alpha Vantage. Pauses the graph with a human-in-the-loop confirmation before
    calling the external API, since it costs an API call and returns financial
    data that a human should sign off on.
    """
    decision = interrupt(
        {
            "type": "stock_price_confirmation",
            "message": f"Confirm fetching the live stock price for '{symbol}'?",
            "symbol": symbol,
        }
    )

    approved = bool(decision) and bool(decision.get("approved", False))
    if not approved:
        return {
            "status": "rejected",
            "symbol": symbol,
            "message": "The user declined the live stock price lookup.",
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY_STOCKS}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"error": str(e), "symbol": symbol}


client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://remarkable-gold-bedbug.fastmcp.app/mcp",
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread
    using the hybrid BM25 + FAISS + cross-encoder-reranked retriever.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, calculator, rag_tool, *mcp_tools]
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 6. Checkpointer (PostgreSQL-backed LangGraph memory)
# -------------------
# A single, long-lived psycopg connection backs the checkpointer for the life
# of the process. PostgresSaver is synchronous, matching the synchronous
# graph.stream()/graph.invoke() calls used by the Streamlit frontend (mixing a
# sync graph with an async checkpointer, as the original SQLite-based version
# did, raises at call time). autocommit + dict_row are required by PostgresSaver
# when a connection is constructed manually rather than via from_conn_string().
_pg_conn = psycopg.connect(
    POSTGRES_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row
)
checkpointer = PostgresSaver(_pg_conn)
checkpointer.setup()

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
