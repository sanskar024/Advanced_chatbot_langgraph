"""Live health / configuration snapshot for the Observability dashboard."""
from . import cache, config


def _first_line(text: str) -> str:
    return (text or "").strip().splitlines()[0] if (text or "").strip() else ""


def system_status() -> dict:
    from .checkpointer import _pg_conn
    from .tools import all_tools, mcp_tools

    try:
        _pg_conn.execute("SELECT 1")
        postgres = {"ok": True, "detail": "Connected"}
    except Exception as exc:  # noqa: BLE001
        postgres = {"ok": False, "detail": str(exc)[:160]}

    mcp_names = {t.name for t in mcp_tools}
    tools = [
        {
            "name": t.name,
            "description": _first_line(t.description)[:140],
            "source": "mcp" if t.name in mcp_names else "native",
        }
        for t in all_tools
    ]

    return {
        "postgres": postgres,
        "redis": cache.get_stats(),
        "langsmith": {
            "enabled": bool(config.LANGSMITH_API_KEY),
            "project": config.LANGSMITH_PROJECT,
            "url": "https://smith.langchain.com",
        },
        "models": {"fast": config.FAST_MODEL_NAME, "strong": config.STRONG_MODEL_NAME},
        "routing": {"heavy_chars": config.HEAVY_TASK_CHAR_THRESHOLD},
        "limits": {
            "rate_calls": config.RATE_LIMIT_MAX_CALLS,
            "rate_window": config.RATE_LIMIT_WINDOW_SECONDS,
            "cache_threshold": config.SEMANTIC_CACHE_THRESHOLD,
            "cache_ttl": config.SEMANTIC_CACHE_TTL_SECONDS,
            "max_corrections": config.MAX_SELF_CORRECTIONS,
        },
        "tools": tools,
        "mcp_count": len(mcp_tools),
    }
