"""
llm_factory.py — Genius AI LLM & Embedding Factory

Centralizes all model instantiation so no other file needs to
know which provider is active. Call setup_global_llm() once at
chat-start; LlamaIndex Settings picks it up everywhere else.

Supported providers (ACTIVE_LLM_PROVIDER in config.py / .env):
  - "ollama"        → local Ollama server (default: qwen2.5:3b)
  - "azure_custom"  → Azure-hosted custom endpoint via LangChain wrapper
"""

import logging

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import (
    ACTIVE_LLM_PROVIDER,
    AZURE_CUSTOM_CONFIG,
    OLLAMA_CONFIG,
    EMBED_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# ── Embedding (shared across providers) ──────────────────────────
_embed_model: HuggingFaceEmbedding | None = None


def get_embed_model() -> HuggingFaceEmbedding:
    """Return (and cache) the HuggingFace embedding model."""
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        Settings.embed_model = _embed_model
    return _embed_model


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

    # Always ensure the embedding model is also registered
    get_embed_model()

    model_label = (
        OLLAMA_CONFIG["model"]
        if ACTIVE_LLM_PROVIDER == "ollama"
        else AZURE_CUSTOM_CONFIG["model"]
    )
    logger.info(f"LLM ready: provider={ACTIVE_LLM_PROVIDER!r}, model={model_label!r}")
    return llm


# ── Private builders ──────────────────────────────────────────────

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
