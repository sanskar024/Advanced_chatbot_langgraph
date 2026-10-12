"""
Central configuration for the app, loaded once from the environment.
Every other module reads settings from here instead of calling os.getenv
directly, so all tunables live in one place.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Core API keys ---
API_KEY = os.getenv("API_KEY")  # Google Gemini
API_KEY_STOCKS = os.getenv("API_KEY_STOCKS")  # Alpha Vantage

# --- Persistence ---
POSTGRES_URI = os.getenv(
    "POSTGRES_URI", "postgresql://chatbot:chatbot@localhost:5432/chatbot"
)

# --- Redis (semantic cache + rate limiting) ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
SEMANTIC_CACHE_MAX_ENTRIES = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "50"))
SEMANTIC_CACHE_TTL_SECONDS = int(
    os.getenv("SEMANTIC_CACHE_TTL_SECONDS", str(6 * 60 * 60))
)
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_CALLS = int(os.getenv("RATE_LIMIT_MAX_CALLS", "10"))

# --- LangSmith tracing ---
# Accept both the legacy LANGCHAIN_* names and the current LANGSMITH_* names.
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = (
    os.getenv("LANGSMITH_PROJECT")
    or os.getenv("LANGCHAIN_PROJECT")
    or "advanced-rag-chatbot"
)
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")

# --- Self-correction / reflection loop ---
MAX_SELF_CORRECTIONS = int(os.getenv("MAX_SELF_CORRECTIONS", "1"))

# --- Dynamic model routing ---
# Simple prompts go to the fast model; long / analytical prompts, multi-tool
# runs and self-correction passes are escalated to the strong model.
FAST_MODEL_NAME = os.getenv("FAST_MODEL_NAME", "gemini-2.5-flash")
STRONG_MODEL_NAME = os.getenv("STRONG_MODEL_NAME", "gemini-2.5-pro")
HEAVY_TASK_CHAR_THRESHOLD = int(os.getenv("HEAVY_TASK_CHAR_THRESHOLD", "600"))
