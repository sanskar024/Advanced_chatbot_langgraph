"""
Shared model instances used across the app: the chat LLM, the embedding
model (used for both FAISS indexing and semantic cache similarity), and the
cross-encoder used to rerank hybrid retrieval results.
"""
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from . import config

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=config.API_KEY,
    temperature=0.2,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
