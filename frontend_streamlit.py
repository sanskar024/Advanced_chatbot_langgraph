"""
Advanced AI Chatbot -- Streamlit frontend.

Views:
  Chat           streaming agent chat, citations, tool trace, HITL approval
  Observability  PostgreSQL / Redis / LangSmith / MCP status, agent graph, telemetry
  DeepEval       RAG quality metrics on the documents indexed in this chat
  Retrieval Lab  BM25 vs FAISS vs hybrid vs cross-encoder-reranked, side by side
"""
import time
import uuid

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Advanced AI Chatbot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui_theme import chip, esc, h, inject_css, score_class, stat_card  # noqa: E402

inject_css()

# ----------------------------------------------------------------------------
# Backend import (fails loudly but nicely if Postgres / keys are misconfigured)
# ----------------------------------------------------------------------------
try:
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        HumanMessage,
        ToolMessage,
    )
    from langgraph.types import Command

    from app import cache as app_cache
    from app.checkpointer import retrieve_all_threads
    from app.graph import chatbot
    from app.retrieval import (
        debug_retrieve,
        ingest_pdf,
        thread_document_metadata,
        thread_documents,
    )
    from app.routing import REVISION_PREFIX
    from app.status import system_status
    from app.utils import message_text, parse_tool_payload
except Exception as exc:  # noqa: BLE001
    h(
        f"""
        <div class="hero"><h1>Backend failed to start</h1>
        <p>{esc(type(exc).__name__)}: {esc(str(exc)[:400])}</p></div>
        """
    )
    st.info(
        "Common causes: `POSTGRES_URI` unreachable (use the Supabase **Session pooler** "
        "URI), a missing `API_KEY`, or a missing package (`pip install -r requirements.txt`)."
    )
    st.stop()

# Set to True to bring back the Observability / DeepEval / Retrieval Lab views.
# Their code below is intact; they are simply not reachable while this is False.
SHOW_EXTRA_VIEWS = False

AI_AVATAR = None    # avatars are hidden in CSS; a text label is shown instead
USER_AVATAR = None
NAV = ["Chat", "Observability", "DeepEval", "Retrieval Lab"]

TOOL_LABELS = {
    "rag_tool": "Document search",
    "web_search": "Web search",
    "get_stock_price": "Stock price",
    "calculator": "Calculator",
}

SUGGESTIONS = [
    ("", "Summarize my documents", "Summarize the key points of the uploaded document(s) with citations."),
    ("", "Check a stock price", "What is the latest stock price of AAPL?"),
    ("", "Do some math", "What is 1284 multiplied by 37, then divided by 12?"),
    ("", "Search the web", "What are the latest developments in agentic AI frameworks?"),
]


# ============================================================================
# Session state
# ============================================================================
def _init_state() -> None:
    ss = st.session_state
    if "thread_id" not in ss:
        ss["thread_id"] = str(uuid.uuid4())
        ss["message_history"] = []
        ss["pending_interrupt"] = None
        ss["chat_threads"] = [str(t) for t in retrieve_all_threads()]
        ss["thread_titles"] = {}
        ss["turn_log"] = []
        ss["model_mode"] = "auto"
        ss["view"] = NAV[0]
        ss["eval_pairs"] = [{"question": "", "ground_truth": ""}]
        ss["eval_rows"] = None
    if ss["thread_id"] not in ss["chat_threads"]:
        ss["chat_threads"].insert(0, ss["thread_id"])


_init_state()
thread_key: str = st.session_state["thread_id"]


def run_config() -> dict:
    return {
        "configurable": {
            "thread_id": thread_key,
            "model_mode": st.session_state["model_mode"],
        },
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }


# ============================================================================
# Conversation <-> history helpers
# ============================================================================
def get_pending_interrupts(cfg: dict) -> list:
    state = chatbot.get_state(config=cfg)
    pending = []
    for task in state.tasks:
        pending.extend(getattr(task, "interrupts", None) or [])
    return pending


def _new_meta() -> dict:
    return {
        "tools": [], "citations": [], "model": None, "tier": None, "reason": None,
        "grounded": None, "self_corrected": False, "cache_hit": False,
        "rate_limited": False, "latency": None, "error": None,
    }


def _record_tool_result(meta: dict, call: dict, msg: ToolMessage) -> None:
    body = message_text(msg.content)
    payload = parse_tool_payload(msg.content) or {}
    failed = getattr(msg, "status", None) == "error" or "error" in payload
    entry = {
        "name": call.get("name") or getattr(msg, "name", "tool"),
        "args": call.get("args") or {},
        "result": body[:600],
        "failed": failed,
    }
    meta["tools"].append(entry)
    if payload.get("citations"):
        meta["citations"].extend(payload["citations"])
    if payload.get("cache_hit"):
        meta["cache_hit"] = True
    if "rate-limited" in body.lower() or "rate limited" in body.lower():
        meta["rate_limited"] = True


def history_from_messages(messages: list) -> list:
    """Rebuild chat bubbles (with tool/citation metadata) from checkpointed
    LangGraph messages, hiding tool plumbing and critic revision prompts."""
    history, cur, calls = [], None, {}
    for m in messages:
        if isinstance(m, HumanMessage):
            text = message_text(m.content)
            if text.startswith(REVISION_PREFIX):
                if cur:
                    cur["meta"]["self_corrected"] = True
                continue
            history.append({"role": "user", "content": text})
            cur = None
        elif isinstance(m, AIMessage):
            if cur is None:
                cur = {"role": "assistant", "content": "", "meta": _new_meta()}
                history.append(cur)
            extra = m.additional_kwargs or {}
            if extra.get("routed_model"):
                cur["meta"].update(
                    model=extra["routed_model"],
                    tier=extra.get("route_tier"),
                    reason=extra.get("route_reason"),
                )
            for tc in m.tool_calls or []:
                calls[tc["id"]] = tc
            text = message_text(m.content)
            if text.strip():
                cur["content"] = text
        elif isinstance(m, ToolMessage) and cur is not None:
            _record_tool_result(cur["meta"], calls.get(m.tool_call_id, {}), m)
    return [x for x in history if x["role"] == "user" or x["content"].strip()]


def thread_title(tid: str, compute: bool) -> str:
    titles = st.session_state["thread_titles"]
    if tid in titles:
        return titles[tid]
    if tid == thread_key and not st.session_state["message_history"]:
        return "New chat"
    if not compute:
        return f"Chat · {tid[:8]}"
    try:
        msgs = chatbot.get_state({"configurable": {"thread_id": tid}}).values.get("messages", [])
        first = next((message_text(m.content) for m in msgs if isinstance(m, HumanMessage)), "")
        titles[tid] = (first.strip().replace("\n", " ")[:38] or f"Chat · {tid[:8]}")
    except Exception:  # noqa: BLE001
        titles[tid] = f"Chat · {tid[:8]}"
    return titles[tid]


def switch_thread(tid: str) -> None:
    ss = st.session_state
    ss["thread_id"] = tid
    ss["view"] = NAV[0]
    ss["pending_interrupt"] = None
    ss["message_history"] = []
    try:
        cfg = {"configurable": {"thread_id": tid}}
        ss["message_history"] = history_from_messages(
            chatbot.get_state(config=cfg).values.get("messages", [])
        )
        pending = get_pending_interrupts(cfg)
        if pending:
            ss["pending_interrupt"] = {"thread": tid, "payload": pending[0].value, "calls": {}}
    except Exception as exc:  # noqa: BLE001
        st.toast(f"Couldn't load conversation: {exc}", icon=None)


def new_chat() -> None:
    ss = st.session_state
    ss["thread_id"] = str(uuid.uuid4())
    ss["message_history"] = []
    ss["pending_interrupt"] = None
    ss["view"] = NAV[0]


@st.cache_data(ttl=10, show_spinner=False)
def cached_status() -> dict:
    return system_status()


# ============================================================================
# Sidebar
# ============================================================================
def sidebar() -> None:
    sb = st.sidebar
    sb.markdown(
        """<div class="brand-name">Advanced AI Chatbot</div>
        <div class="brand-sub">Document Q&amp;A, web search and tools</div>""",
        unsafe_allow_html=True,
    )
    with sb.container():
        st.markdown('<div class="newchat">', unsafe_allow_html=True)
        st.button("New chat", key="new_chat", on_click=new_chat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- model routing ---
    sb.markdown('<div class="side-label">Model routing</div>', unsafe_allow_html=True)
    sb.radio(
        "Model routing",
        ["auto", "fast", "strong"],
        key="model_mode",
        horizontal=True,
        label_visibility="collapsed",
        format_func=lambda x: {"auto": "Auto", "fast": "Fast", "strong": "Strong"}[x],
        help="Auto sends simple queries to the fast model and long, analytical, "
        "multi-tool or self-correction passes to the strong model.",
    )

    # --- documents ---
    sb.markdown('<div class="side-label">Documents</div>', unsafe_allow_html=True)
    uploads = sb.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader-{thread_key}",
        label_visibility="collapsed",
        help="Upload one or more PDFs. They're indexed together (BM25 + FAISS + reranker) "
        "so questions can span all of them.",
    )
    existing = {d["filename"] for d in thread_documents(thread_key)}
    new_files = [f for f in uploads or [] if f.name not in existing]
    if new_files:
        with sb.status(f"Indexing {len(new_files)} PDF(s)…", expanded=True) as box:
            for f in new_files:
                try:
                    box.write(f.name)
                    ingest_pdf(f.getvalue(), thread_id=thread_key, filename=f.name)
                except Exception as exc:  # noqa: BLE001
                    box.write(f"{f.name}: failed ({exc})")
            box.update(label="Documents indexed", state="complete", expanded=False)
    for d in thread_documents(thread_key):
        sb.markdown(
            f"""<div class="doc-chip">
            <div class="n">{esc(d['filename'])}</div>
            <div class="m">{d['documents']} pages · {d['chunks']} chunks</div></div>""",
            unsafe_allow_html=True,
        )
    if not thread_documents(thread_key) and not new_files:
        sb.caption("No documents in this chat yet.")

    # --- sessions ---
    threads = st.session_state["chat_threads"]
    sb.markdown(
        f'<div class="side-label">Sessions · {len(threads)}</div>', unsafe_allow_html=True
    )
    box = sb.container(height=260, border=False)
    with box:
        for i, tid in enumerate(threads):
            st.button(
                thread_title(tid, compute=i < 20),
                key=f"th-{tid}",
                on_click=switch_thread,
                args=(tid,),
                type="primary" if tid == thread_key else "secondary",
                use_container_width=True,
            )

    # --- system status (extra views only) ---
    s = None
    if SHOW_EXTRA_VIEWS:
        try:
            s = cached_status()
        except Exception:  # noqa: BLE001
            s = None
    if s:
        def row(label, ok, text):
            cls = "ok" if ok else "off"
            return (
                f'<div class="sysrow"><span><span class="dot {cls}"></span>{label}</span>'
                f"<span>{esc(text)}</span></div>"
            )

        sb.markdown('<div class="side-label">System</div>', unsafe_allow_html=True)
        sb.markdown(
            row("PostgreSQL", s["postgres"]["ok"], "memory" if s["postgres"]["ok"] else "down")
            + row("Redis", s["redis"].get("available"), "cache" if s["redis"].get("available") else "offline")
            + row("LangSmith", s["langsmith"]["enabled"], "tracing" if s["langsmith"]["enabled"] else "off")
            + row("MCP", s["mcp_count"] > 0, f"{s['mcp_count']} tools"),
            unsafe_allow_html=True,
        )


sidebar()


# ============================================================================
# Chat view
# ============================================================================
def render_meta(meta: dict, key: str) -> None:
    """Chips, sources and agent trace under an assistant message."""
    if not meta:
        return
    chips = []
    if meta.get("model"):
        tier = meta.get("tier") or "fast"
        chips.append(chip(f"{tier} · {meta['model']}", tier))
    seen = []
    for t in meta.get("tools", []):
        if t["name"] in seen:
            continue
        seen.append(t["name"])
        label = TOOL_LABELS.get(t["name"], t["name"])
        chips.append(chip(label, "bad" if t["failed"] else "tool"))
    if meta.get("citations"):
        chips.append(chip(f"{len(meta['citations'])} sources", "good"))
    if meta.get("grounded") is True:
        chips.append(chip("grounded", "good"))
    if meta.get("self_corrected"):
        chips.append(chip("self-corrected", "warn"))
    if meta.get("cache_hit"):
        chips.append(chip("cache hit", "good"))
    if meta.get("rate_limited"):
        chips.append(chip("rate-limited", "warn"))
    if meta.get("latency"):
        chips.append(chip(f"{meta['latency']:.1f}s"))
    if chips:
        h(f'<div class="chips">{"".join(chips)}</div>')

    if meta.get("citations"):
        with st.expander(f"Sources ({len(meta['citations'])})"):
            for c in meta["citations"]:
                page = f" · p.{c['page']}" if c.get("page") else ""
                h(
                    f"""<div class="cite"><div class="h"><span class="num">{esc(c.get('id'))}</span>
                    <span>{esc(c.get('file'))}{esc(page)}</span></div>
                    <div class="t">{esc(c.get('snippet'))}…</div></div>"""
                )
    if meta.get("tools"):
        with st.expander(f"Agent trace ({len(meta['tools'])} tool call(s))"):
            if meta.get("reason"):
                st.caption(f"Router: {meta['reason']}")
            for t in meta["tools"]:
                st.markdown(f"**{TOOL_LABELS.get(t['name'], t['name'])}** · `{t['name']}`")
                if t["args"]:
                    st.json(t["args"], expanded=False)
                st.markdown(
                    f'<div class="trace">{esc(t["result"][:300])}'
                    f'{"…" if len(t["result"]) > 300 else ""}</div>',
                    unsafe_allow_html=True,
                )


def role_label(role: str) -> None:
    name = "You" if role == "user" else "Assistant"
    h(f'<div class="role role-{role}">{name}</div>')


def render_history() -> None:
    for i, msg in enumerate(st.session_state["message_history"]):
        with st.chat_message(msg["role"]):
            role_label(msg["role"])
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_meta(msg.get("meta"), f"h{i}")


def run_turn(payload, cfg: dict, carry_calls: dict | None = None):
    """Run one graph turn, streaming into a chat bubble. Returns (text, meta)."""
    t0 = time.perf_counter()
    before = app_cache.get_stats()
    meta, calls, buffer, streamed = _new_meta(), dict(carry_calls or {}), "", set()

    with st.chat_message("assistant"):
        role_label("assistant")
        status = st.status("Working…", expanded=False)
        slot = st.empty()
        slot.markdown('<div class="thinking">Thinking…</div>', unsafe_allow_html=True)
        try:
            for mode, data in chatbot.stream(
                payload, config=cfg, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    chunk, md = data
                    if md.get("langgraph_node") != "chat_node" or not isinstance(chunk, AIMessage):
                        continue
                    if isinstance(chunk, AIMessageChunk):
                        streamed.add(chunk.id)
                    elif chunk.id in streamed:
                        continue
                    text = message_text(chunk.content)
                    if text:
                        buffer += text
                        slot.markdown(buffer + " ▌")
                    continue

                for node, upd in data.items():
                    if not isinstance(upd, dict):
                        continue  # e.g. __interrupt__
                    if node == "chat_node":
                        last = upd["messages"][-1]
                        extra = last.additional_kwargs or {}
                        meta.update(
                            model=extra.get("routed_model", meta["model"]),
                            tier=extra.get("route_tier", meta["tier"]),
                            reason=extra.get("route_reason", meta["reason"]),
                        )
                        for tc in last.tool_calls or []:
                            calls[tc["id"]] = tc
                            label = TOOL_LABELS.get(tc["name"], tc["name"])
                            status.update(label=f"Running {label}…", state="running", expanded=True)
                            status.write(f"**{label}** `{tc['args']}`")
                    elif node == "tools":
                        for m in upd.get("messages", []):
                            if isinstance(m, ToolMessage):
                                _record_tool_result(meta, calls.get(m.tool_call_id, {}), m)
                                status.write(f"`{m.name}` returned")
                    elif node == "critic":
                        if upd.get("grounded") is True:
                            meta["grounded"] = True
                            status.write("Groundedness check passed")
                        elif upd.get("grounded") is False:
                            meta.update(grounded=None, self_corrected=True)
                            buffer = ""
                            slot.markdown("*Answer wasn't fully grounded; revising…*")
                            status.update(label="Self-correcting…", state="running", expanded=True)
                            status.write("Answer not supported by retrieved context; re-answering")
        except Exception as exc:  # noqa: BLE001
            meta["error"] = str(exc)
            status.update(label="Error", state="error", expanded=True)
            buffer = buffer or f"**Something went wrong.**\n\n`{type(exc).__name__}: {str(exc)[:300]}`"

        meta["latency"] = time.perf_counter() - t0
        after = app_cache.get_stats()
        if before.get("available") and after.get("available"):
            meta["cache_hit"] = meta["cache_hit"] or after["hits"] > before["hits"]
            meta["rate_limited"] = meta["rate_limited"] or after["rate_limited"] > before["rate_limited"]

        if meta["error"] is None:
            n = len(meta["tools"])
            status.update(
                label=f"Done in {meta['latency']:.1f}s" + (f" · {n} tool call(s)" if n else ""),
                state="complete", expanded=False,
            )
        if buffer:
            slot.markdown(buffer)
        else:
            slot.empty()
        if buffer:
            render_meta(meta, "live")
    meta["_calls"] = calls
    return buffer, meta


def finish_turn(text: str, meta: dict, cfg: dict) -> None:
    ss = st.session_state
    calls = meta.pop("_calls", {})
    pending = [] if meta.get("error") else get_pending_interrupts(cfg)
    if pending:
        ss["pending_interrupt"] = {"thread": thread_key, "payload": pending[0].value, "calls": calls}
    else:
        ss["pending_interrupt"] = None
        ss["message_history"].append({"role": "assistant", "content": text, "meta": meta})
    ss["turn_log"].append(
        {
            "time": time.strftime("%H:%M:%S"),
            "model": meta.get("model") or "—",
            "tier": meta.get("tier") or "—",
            "route reason": meta.get("reason") or "—",
            "tools": ", ".join(dict.fromkeys(t["name"] for t in meta["tools"])) or "—",
            "latency_s": round(meta.get("latency") or 0, 2),
            "cache hit": bool(meta.get("cache_hit")),
            "self-corrected": bool(meta.get("self_corrected")),
        }
    )


def view_chat() -> None:
    ss = st.session_state
    pending = ss.get("pending_interrupt")
    waiting = bool(pending and pending.get("thread") == thread_key)

    queued = ss.pop("queued_prompt", None)
    hero_slot = st.empty()
    if not ss["message_history"] and not queued and not waiting:
        meta = thread_document_metadata(thread_key)
        docs_pill = (
            f"{len(meta['files'])} document(s) indexed" if meta else "Upload PDFs from the sidebar to get started"
        )
        with hero_slot.container():
            h(
                f"""
                <div class="hero"><h1>How can I help you today?</h1>
                <p>Ask questions about your uploaded PDFs with cited answers, search the web,
                look up stock prices, or run calculations. {docs_pill}.</p></div>
                """
            )
            cols = st.columns(2)
            for i, (_, title, prompt) in enumerate(SUGGESTIONS):
                if cols[i % 2].button(title, key=f"sg-{title}", use_container_width=True):
                    ss["queued_prompt"] = prompt
                    st.rerun()

    render_history()

    prompt = st.chat_input(
        "Ask about your documents, or use tools…" if not waiting else "Waiting for your approval…",
        disabled=waiting,
    ) or queued

    if prompt and not waiting:
        hero_slot.empty()
        cfg = run_config()
        ss["message_history"].append({"role": "user", "content": prompt})
        ss["thread_titles"].pop(thread_key, None)
        with st.chat_message("user"):
            role_label("user")
            st.markdown(prompt)
        # correction_attempts resets per user turn so the self-correction budget
        # doesn't accumulate across turns in the same thread.
        text, meta = run_turn(
            {
                "messages": [HumanMessage(content=prompt)],
                "correction_attempts": 0,
                "grounded": None,
            },
            cfg,
        )
        finish_turn(text, meta, cfg)
        if ss.get("pending_interrupt"):
            st.rerun()
        return

    if waiting:
        payload = pending["payload"]
        h(
            f"""<div class="approval"><h4>Approval required</h4>
            <p>{esc(payload.get('message', 'The assistant wants to call an external API.'))}</p></div>"""
        )
        c1, c2, _ = st.columns([1, 1, 3])
        decision = None
        if c1.button("Approve", key="approve", use_container_width=True):
            decision = True
        if c2.button("Deny", key="deny", use_container_width=True):
            decision = False
        if decision is not None:
            cfg = run_config()
            # Command(resume=...) continues the paused turn; correction_attempts
            # is intentionally not reset since this is still the same turn.
            text, meta = run_turn(Command(resume={"approved": decision}), cfg, pending.get("calls"))
            finish_turn(text, meta, cfg)
            st.rerun()


# ============================================================================
# Observability view
# ============================================================================
def view_observability() -> None:
    top = st.columns([6, 1, 1])
    top[0].markdown('<div class="section-sub">Live health of every layer in the stack.</div>',
                    unsafe_allow_html=True)
    if top[1].button("Refresh", key="obs_refresh"):
        cached_status.clear()
    if top[2].button("Retry Redis", key="obs_redis"):
        app_cache.retry_connection()
        cached_status.clear()
    try:
        s = cached_status()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't read system status: {exc}")
        return

    r, ls = s["redis"], s["langsmith"]
    rate = f"{r['hit_rate']*100:.0f}%" if r.get("hit_rate") is not None else "—"
    cols = st.columns(4)
    cards = [
        stat_card("PostgreSQL", "Online" if s["postgres"]["ok"] else "Down",
                  f"LangGraph memory · {len(st.session_state['chat_threads'])} sessions"
                  if s["postgres"]["ok"] else s["postgres"]["detail"],
                  "ok" if s["postgres"]["ok"] else "bad"),
        stat_card("Redis", "Online" if r.get("available") else "Offline",
                  f"{r.get('hits', 0)} hits · {r.get('misses', 0)} misses · {rate} hit rate"
                  if r.get("available") else "Cache & rate-limit run fail-open",
                  "ok" if r.get("available") else "off"),
        stat_card("LangSmith", "Tracing" if ls["enabled"] else "Off", f"Project: {ls['project']}",
                  "ok" if ls["enabled"] else "off"),
        stat_card("MCP", f"{s['mcp_count']} tools", f"{len(s['tools'])} tools total", "acc"),
    ]
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)
    if ls["enabled"]:
        st.link_button("Open LangSmith traces ↗", ls["url"])

    st.markdown('<div class="section-title">Agent graph</div>', unsafe_allow_html=True)
    st.graphviz_chart(
        f"""digraph G {{
          bgcolor="transparent"; rankdir=LR; nodesep=0.4;
          node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
                color="#6366f1", fillcolor="#151c30", fontcolor="#e6e9f2", penwidth=1.3];
          edge [color="#8b5cf6", fontname="Helvetica", fontsize=9, fontcolor="#8b93ad"];
          START [shape=circle, fillcolor="#8b5cf6", fontcolor="white", label="start"];
          END [shape=circle, fillcolor="#22d3ee", fontcolor="#0a0e1a", label="end"];
          router [label="Model router\\n{s['models']['fast']} / {s['models']['strong']}", color="#22d3ee"];
          chat_node [label="chat_node\\n(LLM + tools)"];
          tools [label="ToolNode\\nrag · web · stock · calc · MCP"];
          hitl [label="Human approval\\n(stock price)", color="#fbbf24"];
          critic [label="critic\\ngroundedness check", color="#34d399"];
          START -> router -> chat_node;
          chat_node -> tools [label="tool_calls"];
          tools -> chat_node;
          tools -> hitl [style=dashed, label="interrupt"];
          hitl -> tools [style=dashed];
          chat_node -> critic [label="final answer"];
          critic -> chat_node [label="not grounded\\n(revise)", color="#f472b6"];
          critic -> END [label="grounded"];
        }}"""
    )

    st.markdown('<div class="section-title">Retrieval pipeline</div>', unsafe_allow_html=True)
    h(
        """<div class="flow">
        <div class="st"><b>PDFs</b><span>PyPDF · 1000/200 chunks</span></div><div class="ar">→</div>
        <div class="st"><b>BM25</b><span>sparse · weight 0.4</span></div><div class="ar">+</div>
        <div class="st"><b>FAISS</b><span>MiniLM dense · weight 0.6</span></div><div class="ar">→</div>
        <div class="st"><b>Ensemble</b><span>top-8 merged</span></div><div class="ar">→</div>
        <div class="st"><b>Cross-encoder</b><span>ms-marco rerank → top-4</span></div><div class="ar">→</div>
        <div class="st"><b>LLM + critic</b><span>cited, grounded answer</span></div></div>"""
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section-title">Tool registry</div>', unsafe_allow_html=True)
        df = pd.DataFrame(s["tools"]).rename(columns=str.title)
        st.dataframe(df, hide_index=True, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Guardrails</div>', unsafe_allow_html=True)
        lim = s["limits"]
        h(
            f"""<div class="card">
            <div class="sysrow"><span>Rate limit</span><span>{lim['rate_calls']} calls / {lim['rate_window']}s</span></div>
            <div class="sysrow"><span>Semantic cache threshold</span><span>cos ≥ {lim['cache_threshold']}</span></div>
            <div class="sysrow"><span>Cache TTL</span><span>{lim['cache_ttl']//3600}h</span></div>
            <div class="sysrow"><span>Max self-corrections</span><span>{lim['max_corrections']} / turn</span></div>
            <div class="sysrow"><span>Escalate to strong model</span><span>&gt; {s['routing']['heavy_chars']} chars</span></div>
            <div class="sysrow"><span>Redis cache entries</span><span>{', '.join(f'{k}: {v}' for k, v in r.get('entries', {}).items()) or '—'}</span></div>
            </div>"""
        )

    st.markdown('<div class="section-title">This session · turn telemetry</div>', unsafe_allow_html=True)
    log = st.session_state["turn_log"]
    if not log:
        st.caption("Send a message in Chat and per-turn routing, tools and latency will appear here.")
        return
    df = pd.DataFrame(log)
    c1, c2 = st.columns([2, 1])
    c1.bar_chart(df["latency_s"], height=180)
    c2.bar_chart(df["tier"].value_counts(), height=180)
    st.dataframe(df, hide_index=True, use_container_width=True)


# ============================================================================
# DeepEval view
# ============================================================================
def view_eval() -> None:
    from app.evaluation import METRIC_NAMES, generate_qa_pairs, run_deepeval

    ss = st.session_state
    docs = thread_documents(thread_key)
    st.markdown(
        '<div class="section-sub">Score the hybrid retrieval + answer pipeline on the documents '
        "indexed in this chat with <b>DeepEval</b> (LLM-as-judge).</div>",
        unsafe_allow_html=True,
    )
    if not docs:
        st.info("Upload at least one PDF in the sidebar to evaluate retrieval on it.")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("Auto-generate questions", use_container_width=True):
        with st.spinner("Drafting Q&A pairs from your documents…"):
            pairs = generate_qa_pairs(thread_key, n=4)
        if pairs:
            ss["eval_pairs"] = pairs
        else:
            st.warning("Couldn't generate questions; add some manually below.")
    threshold = c3.slider("Pass threshold", 0.1, 1.0, 0.5, 0.05)

    st.caption("Edit or add rows. Ground truth is needed for Contextual Precision / Recall.")
    edited = st.data_editor(
        pd.DataFrame(ss["eval_pairs"]),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "question": st.column_config.TextColumn("Question", width="large"),
            "ground_truth": st.column_config.TextColumn("Expected answer (ground truth)", width="large"),
        },
        key="eval_editor",
    )
    ss["eval_pairs"] = edited.fillna("").to_dict("records")

    if c2.button("▶ Run DeepEval", type="primary", use_container_width=True):
        pairs = [p for p in ss["eval_pairs"] if str(p.get("question", "")).strip()]
        if not pairs:
            st.warning("Add at least one question.")
        else:
            bar = st.progress(0.0, text="Starting…")
            try:
                ss["eval_rows"] = run_deepeval(
                    thread_key, pairs, threshold, lambda p, msg: bar.progress(min(p, 1.0), text=msg)
                )
                ss["eval_threshold"] = threshold
            except Exception as exc:  # noqa: BLE001
                st.error(f"Evaluation failed: {exc}")
            bar.empty()

    rows = ss.get("eval_rows")
    if not rows:
        return

    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
    th = ss.get("eval_threshold", 0.5)
    cols = st.columns(len(METRIC_NAMES))
    for col, name in zip(cols, METRIC_NAMES):
        vals = [r["scores"][name] for r in rows if r["scores"].get(name) is not None]
        avg = sum(vals) / len(vals) if vals else None
        pct = f"{avg*100:.0f}%" if avg is not None else "—"
        col.markdown(
            f"""<div class="card"><div class="k">{name}</div>
            <div class="score {score_class(avg)}">{pct}</div>
            <div class="bar"><i style="width:{(avg or 0)*100:.0f}%"></i></div>
            <div class="s" style="margin-top:.4rem">{len(vals)} scored · pass ≥ {th}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Per question</div>', unsafe_allow_html=True)
    for r in rows:
        ok = all(v is None or v >= th for v in r["scores"].values())
        with st.expander(("Pass · " if ok else "Review · ") + r["question"][:110]):
            st.markdown(f"**Answer:** {r['answer']}")
            if r["ground_truth"]:
                st.markdown(f"**Expected:** {r['ground_truth']}")
            mcols = st.columns(len(METRIC_NAMES))
            for col, name in zip(mcols, METRIC_NAMES):
                v = r["scores"].get(name)
                col.metric(name, "—" if v is None else f"{v:.2f}")
            for name in METRIC_NAMES:
                if r["reasons"].get(name):
                    st.caption(f"**{name}:** {r['reasons'][name]}")


# ============================================================================
# Retrieval Lab view
# ============================================================================
def view_lab() -> None:
    st.markdown(
        '<div class="section-sub">See what each retrieval stage returns for the same query — '
        "and what the cross-encoder keeps.</div>",
        unsafe_allow_html=True,
    )
    if not thread_documents(thread_key):
        st.info("Upload at least one PDF in the sidebar to use the Retrieval Lab.")
        return
    q = st.text_input("Query", placeholder="e.g. What were the main findings?", key="lab_q")
    k = st.slider("Results per stage", 3, 8, 5)
    if not q:
        return
    with st.spinner("Retrieving…"):
        res = debug_retrieve(thread_key, q, k)

    stages = [
        ("bm25", "BM25 · keyword", None),
        ("faiss", "FAISS · semantic", "L2 dist"),
        ("hybrid", "Hybrid ensemble", None),
        ("reranked", "Reranked · final", "score"),
    ]
    cols = st.columns(4)
    for col, (key, title, score_label) in zip(cols, stages):
        with col:
            st.markdown(f"**{title}**")
            for i, hit in enumerate(res.get(key, []), 1):
                page = f" · p.{hit['page']}" if hit["page"] else ""
                score = f"{score_label} {hit['score']:.2f}" if hit["score"] is not None and score_label else ""
                h(
                    f"""<div class="hit"><div class="h"><span>#{i} {esc(hit['file'])}{esc(page)}</span>
                    <span>{score}</span></div>
                    <div class="t">{esc(hit['text'][:260])}…</div></div>"""
                )


# ============================================================================
# Router
# ============================================================================
VIEWS = {
    NAV[0]: view_chat,
    NAV[2]: view_eval
}
if SHOW_EXTRA_VIEWS:
    VIEWS.update({NAV[1]: view_observability, NAV[2]: view_eval, NAV[3]: view_lab})
    st.radio("View", list(VIEWS), key="view", horizontal=True, label_visibility="collapsed")
else:
    st.session_state["view"] = NAV[0]

VIEWS[st.session_state["view"]]()
