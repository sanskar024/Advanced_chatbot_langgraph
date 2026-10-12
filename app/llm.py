"""
Shared model instances used across the app: the chat LLM, the embedding
model (used for both FAISS indexing and semantic cache similarity), and the
cross-encoder used to rerank hybrid retrieval results.
"""
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from . import config

fast_llm = ChatGoogleGenerativeAI(
    model=config.FAST_MODEL_NAME,
    google_api_key=config.API_KEY,
    temperature=0.2,
)

# Used for complex prompts, multi-tool runs and self-correction passes
# (see app/routing.py).
strong_llm = ChatGoogleGenerativeAI(
    model=config.STRONG_MODEL_NAME,
    google_api_key=config.API_KEY,
    temperature=0.2,
)

# Default model for utility calls (grounding check, eval judge, etc.).
llm = fast_llm

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
