"""
config.py — Genius AI Central Configuration

Single source of truth for:
  - LLM provider selection (ollama / azure_custom)
  - Storage paths for ChromaDB and DuckDB
  - Per-provider connection settings

To switch providers, change ACTIVE_LLM_PROVIDER here
or set the env var: ACTIVE_LLM_PROVIDER=ollama
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Application Mode ──────────────────────────────────────────────
# Options: "ollama", "azure_custom"
ACTIVE_LLM_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "ollama")

# ── Application Profile ───────────────────────────────────────────
# Profile name maps to profiles/{name}.yaml
# Examples: "default", "insurance_claims", "customer_support"
APP_PROFILE = os.getenv("APP_PROFILE", "default")

# ── Storage Paths ─────────────────────────────────────────────────
BASE_STORAGE_DIR = Path("./storage/rag")
CHROMA_DB_DIR    = BASE_STORAGE_DIR / "chroma_db"
DUCKDB_DIR       = BASE_STORAGE_DIR / "duckdb"

# Ensure base directories exist on import
BASE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
DUCKDB_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedding Provider ────────────────────────────────────────────
# Options: "huggingface", "azure_custom"
ACTIVE_EMBEDDING_PROVIDER = os.getenv("ACTIVE_EMBEDDING_PROVIDER", "huggingface")

# HuggingFace embedding model (used when ACTIVE_EMBEDDING_PROVIDER=huggingface)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# Azure Custom Embedding (used when ACTIVE_EMBEDDING_PROVIDER=azure_custom)
AZURE_EMBEDDING_CONFIG = {
    "base_url":  os.getenv("AZURE_EMBEDDING_BASE_URL", os.getenv("AZURE_BASE_URL", "https://genailab.tcs.in")),
    "model":     os.getenv("AZURE_EMBEDDING_MODEL", "azure/genailab-maas-text-embedding-3-large"),
    "api_key":   os.getenv("AZURE_EMBEDDING_API_KEY", os.getenv("AZURE_HACKATHON_API_KEY", "")),
}

# ── Ollama Configuration ──────────────────────────────────────────
OLLAMA_CONFIG = {
    "base_url":    os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "model":       os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
    "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
    "request_timeout": 120.0,
}

# ── Azure Custom Configuration ────────────────────────────────────
AZURE_CUSTOM_CONFIG = {
    "base_url": os.getenv("AZURE_BASE_URL", "https://genailab.tcs.in"),
    "model":    os.getenv("AZURE_MODEL", "azure_ai/genailab-maas-DeepSeek-V3-0324"),
    "api_key":  os.getenv("AZURE_HACKATHON_API_KEY", ""),
    "temperature": 0.1,
}

# ── RAG Tuning ────────────────────────────────────────────────────
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 128
SIMILARITY_TOP_K  = 5
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"

# ── Safety Limits ─────────────────────────────────────────────────
MAX_FILE_SIZE_MB       = 50
MAX_FILES_PER_SESSION  = 20
MAX_DATAFRAME_ROWS     = 500_000
SQL_QUERY_TIMEOUT_SECS = 30

# ── Supported File Extensions ─────────────────────────────────────
UNSTRUCTURED_EXTENSIONS = {".pdf", ".txt", ".md"}
STRUCTURED_EXTENSIONS   = {".xlsx", ".xls", ".csv", ".json"}

# ── Authentication ────────────────────────────────────────────────
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin")
DEFAULT_USER_PASSWORD  = os.getenv("DEFAULT_USER_PASSWORD", "GenAi@2025")

# List of users to seed into the database on startup.
# Format: (identifier, password, role)
INITIAL_USERS = [
    ("admin@genius.ai", DEFAULT_ADMIN_PASSWORD, "ADMIN"),
    ("harnish@genius.ai", DEFAULT_USER_PASSWORD, "USER"),
    ("hrishikesh@genius.ai", DEFAULT_USER_PASSWORD, "USER"),
    ("sarvesh@genius.ai", DEFAULT_USER_PASSWORD, "USER"),
    ("aniket@genius.ai", DEFAULT_USER_PASSWORD, "USER"),
    ("avnish@genius.ai", DEFAULT_USER_PASSWORD, "USER"),
]

