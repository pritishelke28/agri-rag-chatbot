"""Central configuration for the Agricultural RAG Chatbot."""
import os
from pathlib import Path
from dotenv import load_dotenv

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")  # explicit path so this works regardless of cwd
DATA_DIR = ROOT_DIR / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
RAW_PDFS_WIKI_DIR = DATA_DIR / "raw_pdfs_wikipedia"
DATASETS_DIR = DATA_DIR / "datasets"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
MODELS_DIR = ROOT_DIR / "artifacts"  # trained .pkl / .pt files land here

for d in [RAW_PDFS_DIR, RAW_PDFS_WIKI_DIR, DATASETS_DIR, CHROMA_PERSIST_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Embeddings / chunking ---------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Retrieval -----------------------------------------------------------
SEARCH_TYPE = "mmr"
RETRIEVER_K = 4

# --- LLM (Groq) ----------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# NOTE: llama-3.3-70b-versatile is deprecated on Groq (shutdown 08/16/2026).
# openai/gpt-oss-120b is Groq's current recommended production replacement.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = 0.2

# --- Chroma collection name ----------------------------------------------
COLLECTION_NAME = "agri_advisory_kb"
