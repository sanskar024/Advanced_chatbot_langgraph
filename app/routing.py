"""
Dynamic model routing: pick the fast or the strong model for each LLM call.

  mode "fast" / "strong"  -> forced by the user (sidebar control)
  mode "auto"             -> heuristic below
"""
import re
from typing import Sequence, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from . import config
from .utils import message_text

REVISION_PREFIX = "Your previous answer may not be fully supported"

_COMPLEX = re.compile(
    r"\b(analy[sz]e|analysis|compare|contrast|evaluate|critique|step[- ]by[- ]step|"
    r"in detail|detailed|derive|prove|explain why|pros and cons|trade-?offs?|"
    r"architecture|design|implement|debug|optimi[sz]e|reason about|"
    r"summari[sz]e (the )?(whole|entire|full|all)|across (all )?(the )?documents)\b",
    re.IGNORECASE,
)


def select_model(messages: Sequence[BaseMessage], mode: str = "auto") -> Tuple[str, str]:
    """Return (tier, reason) where tier is "fast" or "strong"."""
    if mode == "fast":
        return "fast", "forced: fast"
    if mode == "strong":
        return "strong", "forced: strong"

    last = messages[-1] if messages else None
    if isinstance(last, HumanMessage) and message_text(last.content).startswith(REVISION_PREFIX):
        return "strong", "self-correction pass"

    tool_results = 0
    last_human = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            text = message_text(message.content)
            if not text.startswith(REVISION_PREFIX):
                last_human = text
                break
        elif isinstance(message, ToolMessage):
            tool_results += 1

    if len(last_human) > config.HEAVY_TASK_CHAR_THRESHOLD:
        return "strong", f"long prompt (>{config.HEAVY_TASK_CHAR_THRESHOLD} chars)"
    if _COMPLEX.search(last_human):
        return "strong", "complex reasoning request"
    if tool_results >= 3:
        return "strong", "multi-tool orchestration"
    return "fast", "simple query"
