# Advanced AI Chatbot with DeepEval & MCP

An agentic RAG chatbot built with **LangGraph**, **FAISS**, LLMs, and **MCP**,
supporting document Q&A, live web search, stock-price lookups, calculations,
and autonomous multi-tool orchestration.

## Stack

- **LangGraph** — agent graph with conditional tool routing, a self-correction
  loop, and persistent memory
- **Hybrid RAG retrieval** — BM25 (sparse) + FAISS (dense) via an
  `EnsembleRetriever`, with a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranking the merged candidates
  before they reach the LLM
- **Self-correction / reflection loop** — after a RAG-grounded answer, a
  `critic` node checks whether the answer is actually supported by the
  retrieved context and asks the model to revise once if it isn't
- **DeepEval** — offline evaluation of the retrieval + generation pipeline
  (`evaluate_rag.py`), judged by the app's own Gemini model
- **PostgreSQL** — backs LangGraph's checkpointer, so conversation memory
  survives restarts and supports multiple concurrent sessions/threads
- **Redis** — semantic cache (embedding-similarity, not just exact-match) and
  a fixed-window rate limiter shared by the web search and stock-price tools
- **LangSmith** — optional full request tracing (per-node latency, token
  usage, tool inputs/outputs), enabled by setting `LANGSMITH_API_KEY`
- **MCP** — additional tools loaded dynamically from an MCP server via
  `langchain-mcp-adapters`
- **Human-in-the-loop** — the `get_stock_price` tool pauses the graph
  (`interrupt()`) and waits for explicit approval in the UI before making a
  fresh external API call (cache hits skip this, since no live call is made)
- **Docker** — containerized app + Postgres + Redis via `docker-compose`

## Project layout

```
app/
  config.py              # all env-driven settings, read once
  tracing.py             # LangSmith bootstrap (no-op without an API key)
  llm.py                 # shared LLM, embeddings, cross-encoder instances
  cache.py               # Redis semantic cache + rate limiter (fails open)
  retrieval.py           # hybrid BM25 + FAISS + rerank retriever, ingest_pdf
  state.py               # LangGraph state schema
  checkpointer.py        # PostgreSQL-backed checkpointer
  graph.py               # the agent graph, incl. the self-correction loop
  tools/
    calculator.py
    rag.py               # wraps retrieval.py for the agent
    web_search.py        # DuckDuckGo + semantic cache + rate limit
    stock.py             # Alpha Vantage + cache + rate limit + HITL
    mcp.py                # MCP tool discovery
frontend_streamlit.py    # Streamlit chat UI, incl. the HITL approval flow
evaluate_rag.py           # standalone DeepEval evaluation script
requirements.txt
Dockerfile
docker-compose.yml
.env.example
```

## Running with Docker (recommended)

1. Copy `.env.example` to `.env` and fill in `API_KEY` (Google Gemini) and
   `API_KEY_STOCKS` (Alpha Vantage). `LANGSMITH_API_KEY` is optional.
2. Start everything (Postgres + Redis + app):

   ```bash
   docker compose up --build
   ```

3. Open http://localhost:8501.

## Running locally without Docker

1. Start PostgreSQL and (optionally) Redis, and note their connection
   strings. Redis is optional locally — caching/rate limiting silently no-op
   without it.
2. Copy `.env.example` to `.env` and set `API_KEY`, `API_KEY_STOCKS`,
   `POSTGRES_URI`, and (if running Redis) `REDIS_URL`.
3. Install dependencies and run:

   ```bash
   pip install -r requirements.txt
   streamlit run frontend_streamlit.py
   ```

## Human-in-the-loop stock lookups

When the agent decides to call `get_stock_price`, the graph pauses (unless
the answer is already served from the semantic cache) and the UI shows an
**Approve / Deny** prompt before any external API call is made. This uses
LangGraph's `interrupt()` / `Command(resume=...)` mechanism, so the decision
— and the rest of the conversation — is persisted in Postgres.

## Self-correction / reflection loop

After the model produces a final answer that used `rag_tool`, the `critic`
node in `app/graph.py` asks the LLM a focused yes/no question: is this answer
actually supported by the retrieved context? If not, it injects a revision
request and lets the model try again once (`MAX_SELF_CORRECTIONS`, default
1) before the answer reaches the user. This runs at inference time and mirrors
the faithfulness metric `evaluate_rag.py` measures offline.

## Semantic caching & rate limiting

`app/cache.py` provides two Redis-backed utilities used by the web search and
stock-price tools:

- **Semantic cache** — embeds the query with the same sentence-transformer
  used for FAISS, and serves a cached result when a sufficiently similar
  query (cosine similarity ≥ `SEMANTIC_CACHE_THRESHOLD`, default 0.92) was
  answered recently — not just on exact string matches.
- **Rate limiter** — a fixed-window counter (`RATE_LIMIT_MAX_CALLS` per
  `RATE_LIMIT_WINDOW_SECONDS`) per tool, to bound external API usage.

Both fail open if Redis is unreachable: the app still works locally without
Redis, just without caching or rate limiting.

## LangSmith tracing

Set `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`) in `.env` to get
full tracing of every graph node and tool call — inputs, outputs, latency,
and token usage — in the LangSmith dashboard. Leave it blank to disable
tracing entirely; nothing else about the app changes.

## Evaluating retrieval quality with DeepEval

```bash
python evaluate_rag.py path/to/sample.pdf
```

This indexes the PDF through the same hybrid BM25 + FAISS + reranking
pipeline the chatbot uses, generates answers with the same LLM, and scores
them with DeepEval's `FaithfulnessMetric`, `AnswerRelevancyMetric`,
`ContextualPrecisionMetric`, and `ContextualRecallMetric`. The LLM-as-judge
calls are routed through a small `DeepEvalBaseLLM` wrapper
(`GeminiDeepEvalModel`) around the app's own Gemini model, so no separate
OpenAI key is needed. Edit `DEFAULT_EVAL_SET` in `evaluate_rag.py` with real
question/ground-truth pairs for your own document before trusting the
numbers, and consider wiring this into CI (fail the build on a score
regression against a fixed golden dataset) for a real evaluation gate.


  `EnsembleRetriever` feeds a `CrossEncoderReranker` for higher-precision
  context.
- The project was two flat scripts; it's now a proper `app/` package with
  single-responsibility modules (config, tracing, llm, cache, retrieval,
  state, checkpointer, graph, tools/) instead of one large file.
