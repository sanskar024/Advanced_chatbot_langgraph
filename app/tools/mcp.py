"""
MCP tool discovery. `MultiServerMCPClient.get_tools()` is only exposed as an
async API, but the rest of the app (graph, Streamlit) is synchronous, so
discovery runs on a small dedicated background event loop.
"""
import asyncio
import threading

from langchain_mcp_adapters.client import MultiServerMCPClient

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP).result()


_client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://remarkable-gold-bedbug.fastmcp.app/mcp",
        }
    }
)


def load_mcp_tools() -> list:
    try:
        return run_async(_client.get_tools())
    except Exception:
        return []
