"""
LangGraph agent definition.

    START -> chat_node -> [has tool_calls?] -> tools -> chat_node (loop)
                        -> [final answer]    -> critic -> [not grounded &
                                                            attempts left?]
                                                          -> chat_node (loop)
                                                        -> END

`critic` implements a lightweight self-correction / reflection loop: whenever
the most recent tool used was `rag_tool`, it checks whether the model's final
answer is actually supported by the retrieved context and asks for one
revision (bounded by config.MAX_SELF_CORRECTIONS) if it isn't. This is the
same "faithfulness" idea evaluate_rag.py measures offline with DeepEval,
applied at runtime.

Importing `app.tracing` here (for its side effect) turns on LangSmith
tracing for every node/tool call in this graph, as long as an API key is
configured.
"""
import ast
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from . import config, tracing  # noqa: F401  (tracing enabled as an import side effect)
from .checkpointer import checkpointer, retrieve_all_threads  # noqa: F401  (re-exported)
from .llm import llm
from .state import ChatState
from .tools import all_tools

logger = logging.getLogger(__name__)

llm_with_tools = llm.bind_tools(all_tools) if all_tools else llm
tool_node = ToolNode(all_tools) if all_tools else None


def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def _extract_rag_context(messages) -> list[str]:
    """Best-effort extraction of the context chunks from the most recent
    rag_tool call in the message history, for the self-correction check."""
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and getattr(message, "name", None) == "rag_tool":
            try:
                payload = ast.literal_eval(message.content)
                return payload.get("context", []) or []
            except (ValueError, SyntaxError):
                return []
    return []


def _is_grounded(answer: str, context_chunks: list[str]) -> bool:
    context_text = "\n\n".join(context_chunks)
    prompt = (
        "Context:\n"
        f"{context_text}\n\n"
        "Answer:\n"
        f"{answer}\n\n"
        "Reply with only YES if the answer is fully supported by the context "
        "above, or NO if the answer makes claims that are not present in the "
        "context."
    )
    verdict = llm.invoke(prompt).content.strip().upper()
    return verdict.startswith("YES")


def critic_node(state: ChatState):
    """Self-correction step: verify the final answer is grounded in
    retrieved context (when RAG was used) before it reaches the user, and
    request one revision if it isn't."""
    messages = state["messages"]
    attempts = state.get("correction_attempts", 0)
    last_message = messages[-1]

    context_chunks = _extract_rag_context(messages)
    if not context_chunks or attempts >= config.MAX_SELF_CORRECTIONS:
        return {"correction_attempts": attempts}

    if not isinstance(last_message, AIMessage) or not last_message.content:
        return {"correction_attempts": attempts}

    try:
        grounded = _is_grounded(last_message.content, context_chunks)
    except Exception:
        logger.exception("Self-correction grounding check failed; skipping.")
        return {"correction_attempts": attempts}

    if grounded:
        return {"correction_attempts": attempts}

    logger.info("Self-correction triggered: answer not grounded in retrieved context.")
    revision_request = HumanMessage(
        content=(
            "Your previous answer may not be fully supported by the retrieved "
            "document context. Re-answer using only the retrieved context, and "
            "explicitly say you don't know if the context doesn't contain the "
            "answer."
        )
    )
    return {"messages": [revision_request], "correction_attempts": attempts + 1}


def route_after_chat(state: ChatState) -> str:
    last_message = state["messages"][-1]
    if tool_node and getattr(last_message, "tool_calls", None):
        return "tools"
    return "critic"


def route_after_critic(state: ChatState) -> str:
    # critic_node appends a HumanMessage only when it wants another pass.
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return "chat_node"
    return END


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("critic", critic_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges(
        "chat_node", route_after_chat, {"tools": "tools", "critic": "critic"}
    )
    graph.add_edge("tools", "chat_node")
else:
    graph.add_conditional_edges("chat_node", route_after_chat, {"critic": "critic"})

graph.add_conditional_edges(
    "critic", route_after_critic, {"chat_node": "chat_node", END: END}
)

chatbot = graph.compile(checkpointer=checkpointer)
