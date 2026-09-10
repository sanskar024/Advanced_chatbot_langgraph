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
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
