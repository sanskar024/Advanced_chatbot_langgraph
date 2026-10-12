"""
PostgreSQL-backed LangGraph checkpoint memory.

A synchronous PostgresSaver is used deliberately: the Streamlit frontend
drives the graph with the synchronous chatbot.stream()/invoke(), and mixing
that with an async checkpointer (e.g. AsyncSqliteSaver) raises at call time.
"""
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row

from . import config

_pg_conn = psycopg.connect(
    config.POSTGRES_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row
)
checkpointer = PostgresSaver(_pg_conn)
checkpointer.setup()


def retrieve_all_threads() -> list[str]:
    """Unique thread ids, most recently active first."""
    seen: dict[str, None] = {}
    for checkpoint in checkpointer.list(None):
        seen.setdefault(checkpoint.config["configurable"]["thread_id"], None)
    return list(seen)
