"""
Stock price lookup tool combining three production-hardening layers:

  1. Semantic cache checked first -- a cache hit needs neither human approval
     nor a live API call, since no new external request is made.
  2. Human-in-the-loop approval (interrupt()) required before any *fresh*
     external call, since it costs an API call and returns financial data a
     human should sign off on.
  3. A fixed-window rate limiter applied right before the live API call.
"""
import requests
from langchain_core.tools import tool
from langgraph.types import interrupt

from .. import config
from ..cache import check_rate_limit, semantic_cache_lookup, semantic_cache_store
from ..llm import embeddings

_NAMESPACE = "stock_price"


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage. A cached recent lookup is returned immediately; a
    fresh lookup pauses the conversation for human approval first.
    """
    cached = semantic_cache_lookup(_NAMESPACE, symbol, embeddings.embed_query)
    if cached is not None:
        return {**cached, "cache_hit": True}

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

    if not check_rate_limit(_NAMESPACE):
        return {
            "error": "Stock price lookups are temporarily rate-limited.",
            "symbol": symbol,
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={config.API_KEY_STOCKS}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        result = r.json()
    except requests.RequestException as e:
        return {"error": str(e), "symbol": symbol}

    semantic_cache_store(_NAMESPACE, symbol, embeddings.embed_query, result)
    return result
