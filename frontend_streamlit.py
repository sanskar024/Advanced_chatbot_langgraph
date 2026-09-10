import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app.checkpointer import retrieve_all_threads
from app.graph import chatbot
from app.retrieval import ingest_pdf, thread_document_metadata


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["pending_interrupt"] = None


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def get_pending_interrupts(config):
    """Return any pending human-in-the-loop interrupts for this thread."""
    state = chatbot.get_state(config=config)
    pending = []
    for task in state.tasks:
        pending.extend(getattr(task, "interrupts", None) or [])
    return pending


def run_graph_turn(payload, config):
    """
    Drive one turn of the graph (either a fresh human message or a resumed
    Command), streaming AI content into the chat while surfacing tool usage.
    """
    status_holder = {"box": None}

    def ai_only_stream():
        for message_chunk, _ in chatbot.stream(
            payload, config=config, stream_mode="messages"
        ):
            if isinstance(message_chunk, ToolMessage):
                tool_name = getattr(message_chunk, "name", "tool")
                if status_holder["box"] is None:
                    status_holder["box"] = st.status(
                        f"🔧 Using `{tool_name}` …", expanded=True
                    )
                else:
                    status_holder["box"].update(
                        label=f"🔧 Using `{tool_name}` …",
                        state="running",
                        expanded=True,
                    )

            if isinstance(message_chunk, AIMessage):
                yield message_chunk.content

    with st.chat_message("assistant"):
        ai_message = st.write_stream(ai_only_stream())
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    return ai_message


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

CONFIG = {
    "configurable": {"thread_id": thread_key},
    "metadata": {"thread_id": thread_key},
    "run_name": "chat_turn",
}

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status(
            "Indexing PDF (BM25 + FAISS + reranker)…", expanded=True
        ) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input(
    "Ask about your document or use tools",
    disabled=st.session_state["pending_interrupt"] is not None,
)

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # correction_attempts is reset to 0 here (per new user turn) rather than
    # inside the graph, so the self-correction budget doesn't accumulate
    # across separate turns in the same thread.
    ai_message = run_graph_turn(
        {
            "messages": [HumanMessage(content=user_input)],
            "correction_attempts": 0,
        },
        CONFIG,
    )

    pending = get_pending_interrupts(CONFIG)
    if pending:
        # A tool (e.g. get_stock_price) paused for human approval before
        # continuing; stash the interrupt payload and render a confirmation
        # UI instead of appending a final assistant message yet.
        st.session_state["pending_interrupt"] = {
            "thread": thread_key,
            "payload": pending[0].value,
        }
    else:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

# ================= Human-in-the-loop stock confirmation ===========
pending_interrupt = st.session_state.get("pending_interrupt")
if pending_interrupt and pending_interrupt.get("thread") == thread_key:
    payload = pending_interrupt["payload"]
    st.warning(
        f"🔒 **Approval needed:** {payload.get('message', 'The assistant wants to call an external tool.')}"
    )
    col_approve, col_deny = st.columns(2)
    approve_clicked = col_approve.button("✅ Approve", use_container_width=True)
    deny_clicked = col_deny.button("🚫 Deny", use_container_width=True)

    if approve_clicked or deny_clicked:
        resume_value = {"approved": approve_clicked}
        # Command(resume=...) continues the paused turn -- correction_attempts
        # is intentionally NOT reset here, since this is still the same turn.
        ai_message = run_graph_turn(Command(resume=resume_value), CONFIG)

        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )
        st.session_state["pending_interrupt"] = None
        st.rerun()

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    st.session_state["pending_interrupt"] = None
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
