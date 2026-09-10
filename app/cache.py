"""
Redis-backed semantic cache and rate limiter shared by tools that hit
external APIs (web search, stock price lookups).

Both fail open / degrade to a no-op if Redis is unreachable, so the app still
works in local dev without a Redis instance -- caching and rate limiting are
a production optimization layered on top of the tools, not a hard dependency
for correctness.
"""
import json
import logging
import time
from typing import Any, Callable, Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)

_redis_client = None
_redis_unavailable = False


def _get_client():
    """Lazily connect to Redis, remembering permanently if it's unreachable
    so we don't retry a slow connection on every tool call."""
    global _redis_client, _redis_unavailable

    if _redis_unavailable:
        return None
    if _redis_client is not None:
        return _redis_client

    try:
        import redis

        client = redis.from_url(
            config.REDIS_URL, decode_responses=True, socket_connect_timeout=1
        )
        client.ping()
        _redis_client = client
        return _redis_client
    except Exception as exc:  # noqa: BLE001 - any Redis/connection error
        logger.warning("Redis unavailable (%s); caching/rate limiting disabled.", exc)
        _redis_unavailable = True
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    vec_a, vec_b = np.array(a), np.array(b)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def semantic_cache_lookup(
    namespace: str, query: str, embed_fn: Callable[[str], list[float]]
) -> Optional[Any]:
    """
    Return a cached result for `query` in `namespace` if a semantically
    similar query (cosine similarity >= SEMANTIC_CACHE_THRESHOLD) was cached
    recently. Returns None on a miss or when Redis is unavailable.
    """
    client = _get_client()
    if client is None:
        return None

    key = f"semantic_cache:{namespace}"
    try:
        raw_entries = client.lrange(key, 0, config.SEMANTIC_CACHE_MAX_ENTRIES - 1)
    except Exception:
        logger.exception("Redis read failed for %s; treating as a cache miss.", key)
        return None

    if not raw_entries:
        return None

    query_embedding = embed_fn(query)
    best_score, best_result = 0.0, None
    for raw in raw_entries:
        try:
            entry = json.loads(raw)
            score = _cosine_similarity(query_embedding, entry["embedding"])
        except (ValueError, KeyError):
            continue
        if score > best_score:
            best_score, best_result = score, entry["result"]

    if best_score >= config.SEMANTIC_CACHE_THRESHOLD:
        logger.info("Semantic cache hit for '%s' (score=%.3f)", namespace, best_score)
        return best_result
    return None


def semantic_cache_store(
    namespace: str,
    query: str,
    embed_fn: Callable[[str], list[float]],
    result: Any,
) -> None:
    """Cache `result` for `query` under `namespace`, trimmed to the most
    recent SEMANTIC_CACHE_MAX_ENTRIES entries with a TTL."""
    client = _get_client()
    if client is None:
        return

    key = f"semantic_cache:{namespace}"
    entry = json.dumps({"query": query, "embedding": embed_fn(query), "result": result})
    try:
        pipe = client.pipeline()
        pipe.lpush(key, entry)
        pipe.ltrim(key, 0, config.SEMANTIC_CACHE_MAX_ENTRIES - 1)
        pipe.expire(key, config.SEMANTIC_CACHE_TTL_SECONDS)
        pipe.execute()
    except Exception:
        logger.exception("Redis write failed for %s; skipping cache store.", key)


def check_rate_limit(
    namespace: str,
    limit: Optional[int] = None,
    window_seconds: Optional[int] = None,
) -> bool:
    """
    Fixed-window rate limiter. Returns True if the call is allowed, False if
    the caller should back off. Fails open (allows the call) whenever Redis
    is unavailable, so a missing cache layer never breaks core functionality.
    """
    client = _get_client()
    if client is None:
        return True

    limit = limit or config.RATE_LIMIT_MAX_CALLS
    window_seconds = window_seconds or config.RATE_LIMIT_WINDOW_SECONDS
    window = int(time.time() // window_seconds)
    key = f"rate_limit:{namespace}:{window}"

    try:
        count = client.incr(key)
        if count == 1:
            client.expire(key, window_seconds)
        return count <= limit
    except Exception:
        logger.exception("Redis rate-limit check failed for %s; allowing call.", key)
        return True
