"""
Live web search, wrapping DuckDuckGoSearchRun with:
  - a Redis-backed semantic cache, so semantically similar queries (not just
    exact-string repeats) are served without a new network call
  - a fixed-window rate limiter, so one thread can't flood the tool
"""
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool

from ..cache import check_rate_limit, semantic_cache_lookup, semantic_cache_store
from ..llm import embeddings

_ddg_wrapper = DuckDuckGoSearchAPIWrapper(region="us-en")
_raw_search_tool = DuckDuckGoSearchRun(api_wrapper=_ddg_wrapper)

_NAMESPACE = "web_search"


@tool
def web_search(query: str) -> str:
    """
    Search the live web for current information (news, facts, anything not
    contained in the uploaded document). Results for semantically similar
    queries are served from cache for a few hours to avoid redundant calls.
    """
    cached = semantic_cache_lookup(_NAMESPACE, query, embeddings.embed_query)
    if cached is not None:
        return cached

    if not check_rate_limit(_NAMESPACE):
        return "Web search is temporarily rate-limited; please try again shortly."

    result = _raw_search_tool.invoke(query)
    semantic_cache_store(_NAMESPACE, query, embeddings.embed_query, result)
    return result
