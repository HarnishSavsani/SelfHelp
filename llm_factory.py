"""
llm_factory.py — Genius AI LLM & Embedding Factory

Centralizes all model instantiation so no other file needs to
know which provider is active. Call setup_global_llm() once at
chat-start; LlamaIndex Settings picks it up everywhere else.

Supported LLM providers (ACTIVE_LLM_PROVIDER in config.py / .env):
  - "ollama"        → local Ollama server (default: qwen2.5:3b)
  - "azure_custom"  → Azure-hosted custom endpoint via LangChain wrapper

Supported Embedding providers (ACTIVE_EMBEDDING_PROVIDER):
  - "none"         → disable embeddings entirely (BM25-only document search)
  - "huggingface"   → local SentenceTransformers model (default: BAAI/bge-small-en-v1.5)
  - "azure_custom"  → Azure-hosted embedding via LangChain wrapper
"""

import logging

from llama_index.core import Settings

from config import (
    ACTIVE_LLM_PROVIDER,
    ACTIVE_EMBEDDING_PROVIDER,
    AZURE_CUSTOM_CONFIG,
    AZURE_EMBEDDING_CONFIG,
    OLLAMA_CONFIG,
    EMBED_MODEL_NAME,
    HF_HOME_DIR,
)

logger = logging.getLogger(__name__)

# ── Embedding (cached singleton) ─────────────────────────────────
_UNSET = object()
_embed_model = _UNSET


def get_embed_model():
    """Return (and cache) the embedding model based on ACTIVE_EMBEDDING_PROVIDER."""
    global _embed_model
    if _embed_model is not _UNSET:
        return _embed_model

    if ACTIVE_EMBEDDING_PROVIDER == "none":
        _embed_model = None
        Settings.embed_model = None
        logger.info("Embeddings disabled (ACTIVE_EMBEDDING_PROVIDER='none').")
        return _embed_model

    if ACTIVE_EMBEDDING_PROVIDER == "azure_custom":
        _embed_model = _build_azure_embedding()
    else:
        _embed_model = _build_huggingface_embedding()

    Settings.embed_model = _embed_model
    return _embed_model


def _build_huggingface_embedding():
    """Build a local SentenceTransformers embedding model.

    Uses llama_index's HuggingFaceEmbedding which wraps SentenceTransformers.
    The model (~80MB) auto-downloads on first use and is cached locally.
    No heavy HuggingFace Hub or Docling dependencies needed.
    """
    import os
    from pathlib import Path

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    logger.info(f"Loading SentenceTransformers embedding: {EMBED_MODEL_NAME}")

    def _repo_id_to_cache_dir(repo_id: str, hf_home: Path) -> Path:
        # HuggingFace hub cache layout: <HF_HOME>/models--ORG--NAME/...
        # Example: BAAI/bge-small-en-v1.5 -> models--BAAI--bge-small-en-v1.5
        return hf_home / ("models--" + repo_id.replace("/", "--"))

    def _has_local_snapshot(repo_id: str, hf_home: Path) -> bool:
        model_dir = _repo_id_to_cache_dir(repo_id, hf_home)
        snapshots = model_dir / "snapshots"
        if not snapshots.exists():
            return False
        # Any snapshot dir indicates a previously downloaded model.
        try:
            return any(p.is_dir() for p in snapshots.iterdir())
        except Exception:
            return False

    # If the model already exists locally, load strictly from disk.
    local_only = _has_local_snapshot(EMBED_MODEL_NAME, HF_HOME_DIR)
    model_kwargs = {"local_files_only": True} if local_only else {}

    try:
        def _load(local_files_only: bool):
            return HuggingFaceEmbedding(
                model_name=EMBED_MODEL_NAME,
                cache_folder=str(HF_HOME_DIR),
                model_kwargs={"local_files_only": True} if local_files_only else {},
            )

        # 1) If we have a cached snapshot, prefer strict offline load.
        if local_only:
            try:
                return _load(local_files_only=True)
            except Exception as e:
                logger.warning(
                    "Local embedding cache exists but failed to load. "
                    "Re-downloading into ./models/hub to repair the cache. "
                    f"Error: {type(e).__name__}: {e}"
                )
                # Fall through to a fresh download.

        # 2) Otherwise, allow a download (first-time), but ensure it lands in ./models/hub.
        embed = _load(local_files_only=False)
        # After a first-time download, ensure the *next* load is offline-only.
        if not local_only and _has_local_snapshot(EMBED_MODEL_NAME, HF_HOME_DIR):
            logger.info(
                "Embedding model cached under ./models/hub; subsequent runs can be offline."
            )
        return embed
    except Exception as e:
        logger.error(
            f"Failed to load embedding model '{EMBED_MODEL_NAME}'. "
            "Ensure 'sentence-transformers' is installed and the model "
            "is available (it auto-downloads on first use into ./models/hub)."
        )
        raise e


def _build_azure_embedding():
    """
    Build an Azure-hosted custom embedding model wrapped for LlamaIndex.
    Uses LangChain's OpenAIEmbeddings + LlamaIndex's LangchainEmbedding adapter.
    SSL verification is disabled for the TCS internal endpoint.
    """
    import httpx
    from langchain_openai import OpenAIEmbeddings
    from llama_index.embeddings.langchain import LangchainEmbedding

    logger.info(
        f"Loading Azure embedding model: {AZURE_EMBEDDING_CONFIG['model']} "
        f"at {AZURE_EMBEDDING_CONFIG['base_url']}"
    )

    client = httpx.Client(verify=False)
    lc_embed = OpenAIEmbeddings(
        base_url=AZURE_EMBEDDING_CONFIG["base_url"],
        model=AZURE_EMBEDDING_CONFIG["model"],
        api_key=AZURE_EMBEDDING_CONFIG["api_key"],
        http_client=client,
    )
    return LangchainEmbedding(lc_embed)


# ── LLM factory ───────────────────────────────────────────────────

def setup_global_llm():
    """
    Instantiate the LLM based on ACTIVE_LLM_PROVIDER and register it
    globally via LlamaIndex Settings so every component picks it up.

    Returns:
        The configured LlamaIndex-compatible LLM instance.
    """
    logger.info(f"Setting up LLM provider: {ACTIVE_LLM_PROVIDER!r}")

    if ACTIVE_LLM_PROVIDER == "ollama":
        llm = _build_ollama_llm()

    elif ACTIVE_LLM_PROVIDER == "azure_custom":
        llm = _build_azure_llm()

    else:
        raise ValueError(
            f"Unsupported LLM provider: {ACTIVE_LLM_PROVIDER!r}. "
            "Set ACTIVE_LLM_PROVIDER to 'ollama' or 'azure_custom'."
        )

    # Register globally — LlamaIndex reads Settings.llm everywhere
    Settings.llm = llm

    # Register embedding model (or disable embeddings) for the session
    get_embed_model()

    model_label = (
        OLLAMA_CONFIG["model"]
        if ACTIVE_LLM_PROVIDER == "ollama"
        else AZURE_CUSTOM_CONFIG["model"]
    )
    logger.info(f"LLM ready: provider={ACTIVE_LLM_PROVIDER!r}, model={model_label!r}")
    return llm


# ── Private LLM builders ─────────────────────────────────────────

def _build_ollama_llm():
    """Build a local Ollama LLM (no API key required)."""
    from llama_index.llms.ollama import Ollama  # type: ignore[import]

    return Ollama(
        base_url=OLLAMA_CONFIG["base_url"],
        model=OLLAMA_CONFIG["model"],
        temperature=OLLAMA_CONFIG["temperature"],
        request_timeout=OLLAMA_CONFIG["request_timeout"],
    )


def _build_azure_llm():
    """
    Build an Azure-hosted custom LLM wrapped for LlamaIndex.
    Uses LangChain's ChatOpenAI + LlamaIndex's LangChainLLM adapter.
    SSL verification is disabled for the TCS internal endpoint.
    """
    import httpx
    from langchain_openai import ChatOpenAI  # type: ignore[import]
    from llama_index.llms.langchain import LangChainLLM  # type: ignore[import]

    client = httpx.Client(verify=False)
    lc_llm = ChatOpenAI(
        base_url=AZURE_CUSTOM_CONFIG["base_url"],
        model=AZURE_CUSTOM_CONFIG["model"],
        api_key=AZURE_CUSTOM_CONFIG["api_key"],
        http_client=client,
        temperature=AZURE_CUSTOM_CONFIG["temperature"],
    )
    return LangChainLLM(llm=lc_llm)
