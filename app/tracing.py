"""
LangSmith tracing bootstrap.

Importing this module (before building the graph or invoking any
LangChain/LangGraph runnable) turns on full request tracing: per-node
latency, token usage, and tool call inputs/outputs, all visible in the
LangSmith dashboard for the configured project.

Tracing degrades gracefully -- if no API key is configured, this is a no-op
and the rest of the app behaves exactly as before.
"""
import logging
import os

from . import config

logger = logging.getLogger(__name__)


def setup_tracing() -> None:
    if not config.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
        logger.info("LangSmith tracing disabled (no API key configured).")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT
    if config.LANGSMITH_ENDPOINT:
        os.environ["LANGCHAIN_ENDPOINT"] = config.LANGSMITH_ENDPOINT

    logger.info("LangSmith tracing enabled for project '%s'.", config.LANGSMITH_PROJECT)


setup_tracing()
