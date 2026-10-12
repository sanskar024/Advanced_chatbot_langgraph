"""Visual design for the Streamlit frontend: CSS + small HTML helpers."""
import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg:#0e1116; --panel:#151a22; --panel2:#1a212b; --line:#262d39;
  --text:#e7eaf0; --muted:#8a94a6; --accent:#4c8bf5;
  --green:#3fb68b; --amber:#d9a441; --red:#e5675f;
}
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,sans-serif; }
.stApp{ background:var(--bg); }
#MainMenu, footer, [data-testid="stDecoration"]{ visibility:hidden; height:0; }
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:1.6rem; padding-bottom:6rem; max-width:860px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{ background:#0b0e13; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container{ padding-top:1.2rem; }
.brand-name{ font-weight:600; font-size:1rem; letter-spacing:-.01em; }
.brand-sub{ color:var(--muted); font-size:.74rem; margin:.1rem 0 1rem; }
.side-label{
  text-transform:uppercase; letter-spacing:.08em; font-size:.66rem; font-weight:600;
  color:var(--muted); margin:1.2rem 0 .45rem;
}
section[data-testid="stSidebar"] .stButton>button{
  width:100%; justify-content:flex-start; text-align:left; border-radius:8px;
  border:1px solid transparent; background:transparent; color:var(--text);
  padding:.4rem .65rem; font-size:.85rem; font-weight:400;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
section[data-testid="stSidebar"] .stButton>button:hover{ background:var(--panel); border-color:var(--line); }
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{ background:var(--panel2); border-color:var(--line); font-weight:500; }
.newchat .stButton>button{
  background:var(--accent)!important; border:none!important; color:#fff!important;
  font-weight:500!important; justify-content:center!important;
}
.newchat .stButton>button:hover{ filter:brightness(1.08); }
.doc-chip{
  padding:.5rem .65rem; margin:.3rem 0; background:var(--panel);
  border:1px solid var(--line); border-radius:8px; font-size:.8rem;
}
.doc-chip .n{ font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.doc-chip .m{ color:var(--muted); font-size:.72rem; }
.dot{ width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:.45rem; }
.dot.ok{ background:var(--green); } .dot.off{ background:#4a5263; } .dot.bad{ background:var(--red); }
.sysrow{ display:flex; justify-content:space-between; font-size:.76rem; color:var(--muted); padding:.2rem 0; }

/* ---------- top nav (only shown when extra views are enabled) ---------- */
div[data-testid="stRadio"] > div[role="radiogroup"]{
  gap:.25rem; background:var(--panel); padding:.25rem; border-radius:10px;
  border:1px solid var(--line); width:fit-content; flex-wrap:wrap;
}
div[data-testid="stRadio"] label{ padding:.3rem .85rem; border-radius:7px; cursor:pointer; margin:0!important; }
div[data-testid="stRadio"] label > div:first-child{ display:none; }
div[data-testid="stRadio"] label p{ font-size:.84rem; color:var(--muted); }
div[data-testid="stRadio"] label:has(input:checked){ background:var(--panel2); }
div[data-testid="stRadio"] label:has(input:checked) p{ color:var(--text); font-weight:500; }

/* ---------- empty state ---------- */
.hero{ padding:3.2rem 0 1.2rem; }
.hero h1{ font-size:1.9rem; font-weight:600; letter-spacing:-.02em; margin:0 0 .4rem; color:var(--text); }
.hero p{ color:var(--muted); font-size:.95rem; margin:0; max-width:520px; line-height:1.55; }
div[data-testid="stMain"] .stButton>button{
  border-radius:10px; border:1px solid var(--line); background:var(--panel);
  color:var(--text); text-align:left; padding:.8rem .95rem; height:100%;
  font-size:.86rem; font-weight:400; transition:border-color .12s ease, background .12s ease;
}
div[data-testid="stMain"] .stButton>button:hover{ border-color:var(--accent); background:var(--panel2); }

/* ---------- chat ---------- */
/* avatars are replaced by a small text label above each message */
[data-testid^="stChatMessageAvatar"], [data-testid^="chatAvatarIcon"]{ display:none!important; }
div[data-testid="stChatMessage"]{ background:transparent; border:none; padding:.7rem 0; gap:0; }
div[data-testid="stChatMessage"]:has(.role-user){
  background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:.75rem 1rem; margin:.4rem 0;
}
.role{ font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; font-weight:600; color:var(--muted); margin-bottom:.25rem; }
.role-assistant{ color:var(--accent); }
div[data-testid="stChatMessage"] p{ line-height:1.65; font-size:.94rem; }
div[data-testid="stChatMessage"] code{ font-family:'JetBrains Mono',monospace; font-size:.82rem; }
div[data-testid="stChatInput"]{ border-radius:12px; border:1px solid var(--line); background:var(--panel); }
div[data-testid="stChatInput"]:focus-within{ border-color:var(--accent); }
div[data-testid="stBottom"] > div{ background:var(--bg); }
.thinking{ color:var(--muted); font-size:.9rem; }

.chips{ display:flex; flex-wrap:wrap; gap:.35rem; margin:.6rem 0 .25rem; }
.chip{
  display:inline-block; font-size:.7rem; font-weight:500; padding:.15rem .55rem; border-radius:6px;
  border:1px solid var(--line); background:var(--panel); color:var(--muted); white-space:nowrap;
}
.chip.fast{ color:#8ab4f8; } .chip.strong{ color:#b9a6ff; }
.chip.tool{ color:var(--text); }
.chip.good{ color:var(--green); border-color:rgba(63,182,139,.35); }
.chip.warn{ color:var(--amber); border-color:rgba(217,164,65,.35); }
.chip.bad{ color:var(--red); border-color:rgba(229,103,95,.35); }

.cite{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:.6rem .8rem; margin:.4rem 0; }
.cite .h{ display:flex; gap:.5rem; align-items:center; font-size:.76rem; font-weight:500; }
.cite .num{ background:var(--panel2); border:1px solid var(--line); border-radius:4px; padding:0 .4rem; font-size:.7rem; color:var(--accent); }
.cite .t{ color:var(--muted); font-size:.8rem; margin-top:.35rem; line-height:1.5; }
.trace{ font-family:'JetBrains Mono',monospace; font-size:.76rem; color:var(--muted); }

.approval{ border:1px solid rgba(217,164,65,.45); background:rgba(217,164,65,.06); border-radius:10px; padding:.9rem 1.1rem; margin:.6rem 0; }
.approval h4{ margin:0 0 .2rem; font-size:.9rem; color:var(--amber); font-weight:600; }
.approval p{ margin:0; font-size:.9rem; }

/* ---------- cards / extra views (used only when enabled) ---------- */
.card{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:.9rem 1rem; height:100%; }
.card .k{ font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); font-weight:600; }
.card .v{ font-size:1.5rem; font-weight:600; letter-spacing:-.02em; margin:.15rem 0; }
.card .s{ font-size:.76rem; color:var(--muted); }
.card.ok{ border-top:2px solid var(--green); } .card.off{ border-top:2px solid #4a5263; }
.card.bad{ border-top:2px solid var(--red); } .card.acc{ border-top:2px solid var(--accent); }
.section-title{ font-size:1rem; font-weight:600; margin:1.6rem 0 .7rem; }
.section-sub{ color:var(--muted); font-size:.82rem; margin:-.3rem 0 .8rem; }
.flow{ display:flex; align-items:stretch; gap:.4rem; flex-wrap:wrap; margin:.4rem 0 1rem; }
.flow .st{ flex:1; min-width:120px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:.55rem .7rem; font-size:.78rem; }
.flow .st b{ display:block; font-size:.82rem; margin-bottom:.15rem; } .flow .st span{ color:var(--muted); font-size:.72rem; }
.flow .ar{ align-self:center; color:var(--muted); }
.score{ font-size:1.9rem; font-weight:600; }
.score.g{ color:var(--green);} .score.a{ color:var(--amber);} .score.r{ color:var(--red);} .score.n{ color:var(--muted);}
.bar{ height:4px; border-radius:4px; background:var(--line); overflow:hidden; margin-top:.5rem; }
.bar>i{ display:block; height:100%; background:var(--accent); }
.hit{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:.6rem .8rem; margin:.4rem 0; }
.hit .h{ display:flex; justify-content:space-between; font-size:.74rem; color:var(--muted); font-weight:500; }
.hit .t{ font-size:.82rem; margin-top:.3rem; line-height:1.5; color:#c9cfdb; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def h(markup: str) -> None:
    """Render an HTML snippet. Lines are stripped and joined so Markdown never
    mistakes indented HTML for a code block."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{esc(text)}</span>'


def stat_card(title: str, value: str, sub: str = "", tone: str = "acc") -> str:
    return (
        f'<div class="card {tone}"><div class="k">{esc(title)}</div>'
        f'<div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>'
    )


def score_class(score) -> str:
    if score is None:
        return "n"
    return "g" if score >= 0.8 else "a" if score >= 0.5 else "r"
