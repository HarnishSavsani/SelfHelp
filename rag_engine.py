"""
rag_engine.py — Genius AI RAG Engine (Lightweight)

Single-file RAG abstraction managing two pipelines:
  1. Unstructured (PDF/MD/TXT) → ChromaDB vector search (+ BM25 hybrid)
  2. Structured (Excel/CSV/JSON) → DuckDB SQL engine

Design Principles:
  - One RAGEngine instance per chat session
  - Thread-scoped persistence under rag_storage/chroma_db/{thread_id}
    and rag_storage/duckdb/{thread_id}/
  - LLM is resolved from LlamaIndex Settings (set by llm_factory)
  - Zero dependency on Chainlit internals (clean separation)
  - Incremental ingestion — files can be added mid-conversation

Lightweight Architecture:
  - PDF parsing via PyMuPDF4LLM (zero model files, instant startup)
  - Optional embeddings (SentenceTransformers) for vector search
  - BM25 keyword retrieval (can run with embeddings disabled)
  - No Docling, no heavy HuggingFace pipeline, no runtime downloads
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    PromptTemplate,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document, TextNode

from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from sqlalchemy import create_engine, text

from config import (
    CHROMA_DB_DIR,
    DUCKDB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SIMILARITY_TOP_K,
    MAX_FILE_SIZE_MB,
    MAX_FILES_PER_SESSION,
    MAX_DATAFRAME_ROWS,
    UNSTRUCTURED_EXTENSIONS,
    STRUCTURED_EXTENSIONS,
    ENABLE_HYBRID_SEARCH,
    DOCUMENT_RETRIEVAL_MODE,
    SHOW_SOURCES,
)
from app_profile import profile

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# DOCUMENT RESPONSE SYNTHESIS PROMPT
# ══════════════════════════════════════════════════════════════════

DOCUMENT_QA_PROMPT = PromptTemplate(
    "You are a helpful assistant answering questions about the user's uploaded document.\n\n"
    "Rules:\n"
    "1. Use ONLY the information in the context.\n"
    "2. If the answer is not in the context, say so plainly and suggest what to ask/look for.\n"
    "3. Be natural and concise. Prefer a short paragraph + bullets when listing items.\n"
    "4. If the user asks \"who\", \"which\", \"where\", or \"what are\", extract exact names/labels.\n"
    "5. Only include IDs/codes if the user asked for them or they are needed to disambiguate.\n"
    "6. Do NOT mention \"context\", \"chunks\", \"retriever\", or other internal details.\n"
    "7. Do NOT add a separate \"Sources\" section.\n\n"
    "Context:\n"
    "{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer: "
)


class RAGEngine:
    """Per-session RAG engine managing ChromaDB and DuckDB pipelines.

    The LLM is NOT passed as a constructor argument. It is resolved
    from LlamaIndex's global Settings.llm (set once by llm_factory).
    """

    def __init__(self, thread_id: str, embed_model=None):
        """
        Initialize with thread ID for persistence scoping.

        Args:
            thread_id:   Unique chat thread identifier.
            embed_model: Optional custom embedding model override.
                         Defaults to whatever is in Settings.embed_model.
        """
        self.thread_id = thread_id

        # ── Resolve LLM from global Settings ─────────────────────
        self.llm = Settings.llm

        # ── Storage paths (from config) ───────────────────────────
        self.chroma_dir    = CHROMA_DB_DIR / thread_id
        self.duckdb_path   = DUCKDB_DIR / thread_id / "structured.duckdb"
        self.metadata_path = DUCKDB_DIR / thread_id / "metadata.json"

        # Ensure directories exist
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Embedding model ───────────────────────────────────────
        self.embed_model = embed_model or Settings.embed_model

        # ── Document retrieval mode ───────────────────────────────
        self.document_retrieval_mode = DOCUMENT_RETRIEVAL_MODE
        self._embeddings_enabled = (
            self.document_retrieval_mode != "bm25" and self.embed_model is not None
        )

        # ── BM25 persistence (works with or without embeddings) ───
        self.bm25_nodes_path = self.duckdb_path.parent / "bm25_nodes.jsonl"

        # ── ChromaDB (Vector Store) ───────────────────────────────
        # Only initialize vector search when embeddings are enabled.
        self.chroma_client = None
        self.chroma_collection = None
        self.vector_store = None
        self.vector_index: Optional[VectorStoreIndex] = None

        if self._embeddings_enabled:
            # Lazy import to avoid pulling vector-store dependencies when running BM25-only.
            import chromadb  # type: ignore[import]
            from llama_index.vector_stores.chroma import ChromaVectorStore  # type: ignore[import]

            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self.chroma_collection = self.chroma_client.get_or_create_collection("documents")
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

            if self.chroma_collection.count() > 0:
                self.vector_index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    embed_model=self.embed_model,
                )
                logger.info(
                    f"[{thread_id}] Restored ChromaDB vector index "
                    f"({self.chroma_collection.count()} vectors)"
                )
        else:
            if self.document_retrieval_mode != "bm25":
                logger.warning(
                    f"[{thread_id}] Embeddings disabled; falling back to BM25-only "
                    f"(DOCUMENT_RETRIEVAL_MODE={self.document_retrieval_mode!r})."
                )

        # ── DuckDB (Structured SQL) ───────────────────────────────
        self.sql_engine   = None
        self.sql_database: Optional[SQLDatabase] = None
        self.sql_tables:   list[str] = []
        self.table_source_files: dict[str, str] = {}  # table_name → original filename

        if self.duckdb_path.exists():
            self._connect_duckdb()
            logger.info(
                f"[{thread_id}] Reconnected to DuckDB "
                f"(tables: {self.sql_tables})"
            )

        # ── Query Routing ─────────────────────────────────────────
        self.vector_query_engine  = None
        self.sql_query_engine     = None
        self.router_query_engine: Optional[RouterQueryEngine] = None

        # ── BM25 for hybrid retrieval ─────────────────────────────
        self._all_nodes: list[TextNode] = []  # Stored for BM25 re-ranking

        # ── File tracking ─────────────────────────────────────────
        self.metadata = self._load_metadata()

        # Restore BM25 nodes (so BM25-only mode still works after resume)
        self._load_bm25_nodes()
        # If the session was previously indexed before BM25 persistence existed,
        # try to rebuild BM25 nodes from Chroma (no embeddings required).
        if not self._all_nodes:
            self._bootstrap_bm25_nodes_from_chroma()

        if self.vector_index or self.sql_tables or self._all_nodes:
            self._rebuild_router()

        logger.info(f"[{thread_id}] RAGEngine initialized")

    # ══════════════════════════════════════════════════════════════
    # FILE CLASSIFICATION & VALIDATION
    # ══════════════════════════════════════════════════════════════

    def _classify_file(self, file_name: str) -> str:
        ext = Path(file_name).suffix.lower()
        if ext in UNSTRUCTURED_EXTENSIONS:
            return "unstructured"
        elif ext in STRUCTURED_EXTENSIONS:
            return "structured"
        return "unsupported"

    def _validate_file(self, file_path: str, file_name: str) -> Optional[str]:
        path = Path(file_path)
        if not path.exists():
            return f"❌ File not found: `{file_name}`"

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return (
                f"❌ File `{file_name}` is too large ({size_mb:.1f} MB). "
                f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
            )
        if path.stat().st_size == 0:
            return f"❌ File `{file_name}` is empty."

        total_files = len(self.metadata.get("files", []))
        if total_files >= MAX_FILES_PER_SESSION:
            return (
                f"❌ Maximum files per session reached ({MAX_FILES_PER_SESSION}). "
                "Please start a new chat to upload more files."
            )

        if self._classify_file(file_name) == "unsupported":
            supported = ", ".join(sorted(UNSTRUCTURED_EXTENSIONS | STRUCTURED_EXTENSIONS))
            return (
                f"❌ Unsupported file type for `{file_name}`. "
                f"Supported formats: {supported}"
            )
        return None

    # ══════════════════════════════════════════════════════════════
    # FILE PERSISTENCE & METADATA
    # ══════════════════════════════════════════════════════════════

    def _copy_source_file(self, file_path: str, file_name: str) -> Path:
        """Copy an uploaded file alongside the DuckDB directory."""
        source_dir = self.duckdb_path.parent / "source_files"
        source_dir.mkdir(parents=True, exist_ok=True)
        dest = source_dir / file_name

        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            counter = 2
            while dest.exists():
                dest = source_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.copy2(file_path, dest)
        logger.info(f"[{self.thread_id}] Copied source file to {dest}")
        return dest

    def _load_metadata(self) -> dict:
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {
            "thread_id": self.thread_id,
            "files": [],
            "vector_index_exists": False,
            "sql_tables": [],
        }

    def _save_metadata(self):
        self.metadata["vector_index_exists"] = self.vector_index is not None
        self.metadata["sql_tables"] = list(self.sql_tables)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"[{self.thread_id}] Metadata saved")

    # ══════════════════════════════════════════════════════════════
    # BM25 NODE PERSISTENCE (FOR BM25-ONLY MODE + RESUME)
    # ══════════════════════════════════════════════════════════════

    def _persist_nodes_for_bm25(self, nodes: list[TextNode]):
        """Append nodes to the BM25 store on disk for resume support."""
        if not nodes:
            return

        # Keep an in-memory copy for fast BM25 retrieval
        self._all_nodes.extend(nodes)

        try:
            self.bm25_nodes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.bm25_nodes_path, "a", encoding="utf-8") as f:
                for node in nodes:
                    rec = {
                        "id": getattr(node, "node_id", None),
                        "text": node.get_content(),
                        "metadata": getattr(node, "metadata", {}) or {},
                    }
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        except Exception as e:
            logger.warning(f"[{self.thread_id}] Failed to persist BM25 nodes: {e}")

    def _load_bm25_nodes(self):
        """Load persisted nodes for BM25 retrieval (best-effort)."""
        if not self.bm25_nodes_path.exists():
            return

        loaded = 0
        try:
            with open(self.bm25_nodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    text = rec.get("text", "")
                    if not text:
                        continue
                    meta = rec.get("metadata") or {}
                    node_id = rec.get("id")
                    try:
                        node = TextNode(text=text, metadata=meta, id_=node_id)
                    except Exception:
                        node = TextNode(text=text, metadata=meta)
                    self._all_nodes.append(node)
                    loaded += 1
        except Exception as e:
            logger.warning(f"[{self.thread_id}] Failed to load BM25 nodes: {e}")

        if loaded:
            logger.info(f"[{self.thread_id}] Restored {loaded} BM25 nodes from disk")

    def _bootstrap_bm25_nodes_from_chroma(self):
        """Best-effort: rebuild BM25 nodes from an existing Chroma collection.

        This helps when older sessions were indexed into Chroma before we started
        persisting `bm25_nodes.jsonl`.
        """
        if self._all_nodes:
            return
        if not self.chroma_dir.exists():
            return

        try:
            import chromadb  # type: ignore[import]

            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            collection = client.get_or_create_collection("documents")
            if collection.count() <= 0:
                return

            data = collection.get(include=["documents", "metadatas"])
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []

            rebuilt = []
            for text, meta in zip(docs, metas):
                if not text:
                    continue
                try:
                    rebuilt.append(TextNode(text=text, metadata=meta or {}))
                except Exception:
                    continue

            if rebuilt:
                # Persist so future resumes are fast and Chroma isn't required for BM25 mode.
                self._persist_nodes_for_bm25(rebuilt)
                logger.info(f"[{self.thread_id}] Bootstrapped {len(rebuilt)} BM25 nodes from Chroma")
        except Exception as e:
            logger.warning(f"[{self.thread_id}] Failed to bootstrap BM25 nodes from Chroma: {e}")

    def _add_file_metadata(
        self,
        file_name: str,
        file_type: str,
        *,
        chunks: int = 0,
        tables: list[str] | None = None,
        total_rows: int = 0,
        content_types: list[str] | None = None,
    ):
        entry: dict = {
            "name": file_name,
            "type": file_type,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        if file_type == "unstructured":
            entry["chunks"] = chunks
            if content_types:
                entry["content_types"] = content_types
        elif file_type == "structured":
            entry["tables"] = tables or []
            entry["total_rows"] = total_rows

        self.metadata["files"].append(entry)
        self._save_metadata()

    # ══════════════════════════════════════════════════════════════
    # DUCKDB CONNECTION MANAGEMENT
    # ══════════════════════════════════════════════════════════════

    def _connect_duckdb(self):
        if self.sql_engine is not None:
            self.sql_engine.dispose()

        self.sql_engine = create_engine(
            f"duckdb:///{self.duckdb_path}",
            pool_pre_ping=True,
        )

        with self.sql_engine.connect() as conn:
            tables_result = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                )
            ).fetchall()
        self.sql_tables = [row[0] for row in tables_result]

        if self.sql_tables:
            self.sql_database = SQLDatabase(
                self.sql_engine,
                include_tables=self.sql_tables,
            )
            # Rebuild source file mapping from metadata
            for f in self.metadata.get("files", []):
                if f.get("type") == "structured":
                    for tbl in f.get("tables", []):
                        self.table_source_files[tbl] = f["name"]

    # ══════════════════════════════════════════════════════════════
    # TABLE NAME & DATAFRAME CLEANING
    # ══════════════════════════════════════════════════════════════

    def _clean_table_name(self, name: str) -> str:
        name = Path(name).stem if "." in name else name
        name = re.sub(r"[^a-zA-Z0-9]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_").lower()
        if name and name[0].isdigit():
            name = f"t_{name}"
        return name or "unnamed_table"

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = []
        for col in df.columns:
            cleaned = re.sub(r"[^a-zA-Z0-9]", "_", str(col))
            cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
            if cleaned and cleaned[0].isdigit():
                cleaned = f"col_{cleaned}"
            if not cleaned:
                cleaned = f"col_{len(new_cols)}"
            new_cols.append(cleaned)

        seen: dict = {}
        final_cols = []
        for col in new_cols:
            if col in seen:
                seen[col] += 1
                final_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_cols.append(col)

        df.columns = final_cols
        df = df.dropna(axis=1, how="all")
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()
        return df

    # ══════════════════════════════════════════════════════════════
    # FILE INGESTION — PUBLIC API
    # ══════════════════════════════════════════════════════════════

    async def ingest_file(self, file_path: str, file_name: str) -> str:
        error = self._validate_file(file_path, file_name)
        if error:
            logger.warning(f"[{self.thread_id}] Validation failed for {file_name}: {error}")
            return error

        self._copy_source_file(file_path, file_name)
        file_type = self._classify_file(file_name)
        logger.info(f"[{self.thread_id}] Ingesting file: {file_name} ({file_type})")

        try:
            if file_type == "unstructured":
                result = await self._ingest_unstructured(file_path, file_name)
            else:
                result = await self._ingest_structured(file_path, file_name)
            self._rebuild_router()
            return result
        except Exception as e:
            error_msg = f"❌ Failed to process `{file_name}`: {type(e).__name__}: {e}"
            logger.error(f"[{self.thread_id}] {error_msg}", exc_info=True)
            return error_msg

    # ══════════════════════════════════════════════════════════════
    # UNSTRUCTURED INGESTION (PDF, MD, TXT → ChromaDB)
    # ══════════════════════════════════════════════════════════════

    async def _ingest_unstructured(self, file_path: str, file_name: str) -> str:
        ext = Path(file_name).suffix.lower()

        if ext == ".pdf":
            return await self._ingest_pdf(file_path, file_name)
        else:
            return await self._ingest_text_file(file_path, file_name)

    async def _ingest_text_file(self, file_path: str, file_name: str) -> str:
        """Ingest plain text / markdown files using SentenceSplitter."""
        reader    = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

        if not documents:
            return f"⚠️ No content could be extracted from `{file_name}`."

        for doc in documents:
            doc.metadata.update({
                "source_filename": file_name,
                "upload_time":     datetime.now(timezone.utc).isoformat(),
                "thread_id":       self.thread_id,
            })

        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes    = splitter.get_nodes_from_documents(documents)
        logger.info(f"[{self.thread_id}] Created {len(nodes)} chunks from {file_name}")

        self._index_nodes(nodes)
        self._add_file_metadata(file_name, "unstructured", chunks=len(nodes))

        mode_line = (
            "- 🔎 **Indexed for**: hybrid search (keyword + semantic)\n"
            if self._embeddings_enabled
            else "- 🔎 **Indexed for**: keyword search (no embeddings)\n"
        )
        return (
            f"📄 **{file_name}** processed successfully!\n\n"
            f"- 📦 **{len(nodes)}** sections added\n"
            + mode_line +
            f"- You can now ask questions about this document's content."
        )

    async def _ingest_pdf(self, file_path: str, file_name: str) -> str:
        """
        Ingest PDF files using PyMuPDF4LLM for high-quality markdown extraction.

        PyMuPDF4LLM advantages over Docling:
          - Zero model files needed (no .safetensors, no runtime downloads)
          - Instant startup (no layout model initialization)
          - Preserves tables as markdown tables
          - Preserves headers, lists, and formatting
          - Page-level metadata for accurate source citations

        Chunking uses MarkdownNodeParser which splits by markdown headers
        (##, ###) keeping tables and lists intact within their sections.
        """
        try:
            import pymupdf4llm

            logger.info(f"[{self.thread_id}] Using PyMuPDF4LLM for PDF parsing: {file_name}")

            # Extract PDF as structured markdown (preserves tables, headers, lists)
            md_text = pymupdf4llm.to_markdown(file_path, page_chunks=True)

            if not md_text:
                logger.warning(f"[{self.thread_id}] PyMuPDF4LLM returned no content, falling back to simple reader")
                return await self._ingest_text_file(file_path, file_name)

            # Build LlamaIndex Document objects with per-page metadata
            documents = []
            if isinstance(md_text, list):
                # page_chunks=True returns a list of dicts with 'text' and 'metadata'
                for page_data in md_text:
                    page_text = page_data.get("text", "") if isinstance(page_data, dict) else str(page_data)
                    page_meta = page_data.get("metadata", {}) if isinstance(page_data, dict) else {}

                    if not page_text.strip():
                        continue

                    doc = Document(
                        text=page_text,
                        metadata={
                            "source_filename": file_name,
                            "page_label":      str(page_meta.get("page", "")),
                            "upload_time":     datetime.now(timezone.utc).isoformat(),
                            "thread_id":       self.thread_id,
                            "parser":          "pymupdf4llm",
                        },
                    )
                    documents.append(doc)
            else:
                # Single string fallback
                documents.append(Document(
                    text=str(md_text),
                    metadata={
                        "source_filename": file_name,
                        "upload_time":     datetime.now(timezone.utc).isoformat(),
                        "thread_id":       self.thread_id,
                        "parser":          "pymupdf4llm",
                    },
                ))

            if not documents:
                return f"⚠️ No content could be extracted from `{file_name}`."

            total_chars = sum(len(doc.get_content()) for doc in documents)
            logger.info(
                f"[{self.thread_id}] PyMuPDF4LLM extracted {total_chars} chars "
                f"from {len(documents)} pages of {file_name}"
            )

            # Structure-aware chunking via MarkdownNodeParser
            # Splits by markdown headers (##, ###), keeping tables and lists intact
            try:
                from llama_index.core.node_parser import MarkdownNodeParser

                md_parser = MarkdownNodeParser()
                nodes = md_parser.get_nodes_from_documents(documents)

                if not nodes:
                    raise ValueError("MarkdownNodeParser returned 0 nodes")

                # For very large sections (>2000 chars), further split with
                # SentenceSplitter using a larger chunk size to preserve context
                final_nodes = []
                oversized_splitter = SentenceSplitter(
                    chunk_size=1500, chunk_overlap=200
                )
                for node in nodes:
                    content_len = len(node.get_content())
                    if content_len > 2000:
                        sub_doc = Document(
                            text=node.get_content(),
                            metadata=node.metadata.copy(),
                        )
                        sub_nodes = oversized_splitter.get_nodes_from_documents([sub_doc])
                        final_nodes.extend(sub_nodes)
                        logger.debug(
                            f"[{self.thread_id}] Split oversized section "
                            f"({content_len} chars) into {len(sub_nodes)} sub-chunks"
                        )
                    else:
                        final_nodes.append(node)

                nodes = final_nodes

                logger.info(
                    f"[{self.thread_id}] MarkdownNodeParser: {file_name} → "
                    f"{len(nodes)} structure-aware chunks"
                )
            except Exception as parser_err:
                logger.warning(
                    f"[{self.thread_id}] MarkdownNodeParser failed ({parser_err}), "
                    f"using SentenceSplitter fallback"
                )
                splitter = SentenceSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
                nodes = splitter.get_nodes_from_documents(documents)

            # Classify content types found in chunks
            content_types = set()
            for node in nodes:
                text_lower = node.get_content().lower()
                if "|" in text_lower and "---" in text_lower:
                    content_types.add("tables")
                if "```" in text_lower:
                    content_types.add("code")
                content_types.add("text")

            # Index nodes into ChromaDB + BM25
            self._index_nodes(nodes)
            self._add_file_metadata(
                file_name, "unstructured",
                chunks=len(nodes),
                content_types=list(content_types),
            )

            type_summary = ", ".join(sorted(content_types))
            mode_line = (
                "- 🔎 **Indexed for**: hybrid search (keyword + semantic)\n"
                if self._embeddings_enabled
                else "- 🔎 **Indexed for**: keyword search (no embeddings)\n"
            )
            return (
                f"📄 **{file_name}** processed successfully!\n\n"
                f"- 🔬 **Parser**: PyMuPDF4LLM (lightweight)\n"
                f"- 📦 **{len(nodes)}** structure-aware sections added\n"
                + mode_line +
                f"- 📋 **Content detected**: {type_summary}\n"
                f"- You can now ask questions about tables, text, and data in this document."
            )

        except ImportError:
            logger.warning(
                f"[{self.thread_id}] pymupdf4llm not installed, "
                f"falling back to simple reader"
            )
            return await self._ingest_text_file(file_path, file_name)

        except Exception as e:
            logger.warning(
                f"[{self.thread_id}] PyMuPDF4LLM parsing failed for {file_name}: {e}. "
                f"Falling back to simple reader."
            )
            return await self._ingest_text_file(file_path, file_name)

    def _index_nodes(self, nodes: list):
        """Index nodes for retrieval.

        - Always persists nodes for BM25 keyword retrieval (and resume support).
        - Optionally indexes into ChromaDB for vector retrieval when enabled.
        """
        # Persist for BM25 (and keep in-memory copy)
        self._persist_nodes_for_bm25(nodes)

        # Vector indexing is optional
        if not self._embeddings_enabled:
            return

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        if self.vector_index is None:
            self.vector_index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
        else:
            self.vector_index.insert_nodes(nodes)

    # ══════════════════════════════════════════════════════════════
    # STRUCTURED INGESTION (Excel, CSV, JSON → DuckDB)
    # ══════════════════════════════════════════════════════════════

    async def _ingest_structured(self, file_path: str, file_name: str) -> str:
        ext            = Path(file_name).suffix.lower()
        tables_created = []
        total_rows     = 0

        try:
            if ext in (".xlsx", ".xls"):
                sheets = pd.read_excel(file_path, sheet_name=None)
                for sheet_name, df in sheets.items():
                    if df.empty:
                        continue
                    if len(df) > MAX_DATAFRAME_ROWS:
                        df = df.head(MAX_DATAFRAME_ROWS)
                    table_name = self._clean_table_name(sheet_name)
                    df = self._clean_dataframe(df)
                    self._load_df_to_duckdb(df, table_name)
                    self.table_source_files[table_name] = file_name
                    tables_created.append((table_name, len(df), list(df.columns)))
                    total_rows += len(df)

            elif ext == ".csv":
                df = None
                for encoding in ("utf-8", "latin-1", "cp1252"):
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    return f"❌ Could not decode `{file_name}`. Unsupported encoding."
                if len(df) > MAX_DATAFRAME_ROWS:
                    df = df.head(MAX_DATAFRAME_ROWS)
                table_name = self._clean_table_name(file_name)
                df = self._clean_dataframe(df)
                self._load_df_to_duckdb(df, table_name)
                self.table_source_files[table_name] = file_name
                tables_created.append((table_name, len(df), list(df.columns)))
                total_rows = len(df)

            elif ext == ".json":
                try:
                    with open(file_path, "r") as f:
                        raw = json.load(f)
                    if isinstance(raw, list):
                        df = pd.json_normalize(raw)
                    elif isinstance(raw, dict):
                        for key, val in raw.items():
                            if isinstance(val, list):
                                df = pd.json_normalize(val)
                                break
                        else:
                            df = pd.json_normalize([raw])
                    else:
                        return f"❌ Unsupported JSON structure in `{file_name}`."
                except json.JSONDecodeError as e:
                    return f"❌ Invalid JSON in `{file_name}`: {e}"

                if len(df) > MAX_DATAFRAME_ROWS:
                    df = df.head(MAX_DATAFRAME_ROWS)
                table_name = self._clean_table_name(file_name)
                df = self._clean_dataframe(df)
                self._load_df_to_duckdb(df, table_name)
                self.table_source_files[table_name] = file_name
                tables_created.append((table_name, len(df), list(df.columns)))
                total_rows = len(df)

        except Exception as e:
            return f"❌ Error reading `{file_name}`: {type(e).__name__}: {e}"

        if not tables_created:
            return f"⚠️ No data could be extracted from `{file_name}`."

        table_names = [t[0] for t in tables_created]
        self._add_file_metadata(
            file_name, "structured",
            tables=table_names,
            total_rows=total_rows,
        )

        lines = [f"📊 **{file_name}** loaded successfully!\n"]
        lines.append("| Table | Rows | Columns |")
        lines.append("|-------|------|---------|")
        for tname, nrows, cols in tables_created:
            col_preview = ", ".join(cols[:5])
            if len(cols) > 5:
                col_preview += f", ... (+{len(cols) - 5} more)"
            lines.append(f"| `{tname}` | {nrows:,} | {len(cols)} ({col_preview}) |")
        lines.append("\n💡 You can now ask analytical questions about this data!")
        return "\n".join(lines)

    def _load_df_to_duckdb(self, df: pd.DataFrame, table_name: str):
        if self.sql_engine is not None:
            self.sql_engine.dispose()
            self.sql_engine = None

        conn = duckdb.connect(str(self.duckdb_path))
        try:
            conn.register("_temp_df", df)
            conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _temp_df')
            conn.unregister("_temp_df")
        finally:
            conn.close()

        if table_name not in self.sql_tables:
            self.sql_tables.append(table_name)

        self._connect_duckdb()
        logger.info(
            f"[{self.thread_id}] Loaded table '{table_name}' "
            f"with {len(df)} rows, {len(df.columns)} cols"
        )

    # ══════════════════════════════════════════════════════════════
    # TABLE CONTEXT (SAMPLE DATA FOR SMARTER SQL)
    # ══════════════════════════════════════════════════════════════

    def _get_sample_data_context(self) -> str:
        """Fetch sample rows from each table so the LLM understands the actual data."""
        if self.sql_engine is None or not self.sql_tables:
            return ""

        context_parts = []
        try:
            with self.sql_engine.connect() as conn:
                for table in self.sql_tables:
                    source_file = self.table_source_files.get(table, "uploaded file")
                    try:
                        result = conn.execute(text(f"SELECT * FROM {table} LIMIT 3"))
                        cols = list(result.keys())
                        rows = result.fetchall()
                        if rows:
                            header = " | ".join(cols)
                            sample_lines = []
                            for row in rows:
                                sample_lines.append(" | ".join(str(v) for v in row))
                            context_parts.append(
                                f"Table '{table}' (from {source_file}):\n"
                                f"  Columns: {header}\n"
                                f"  Sample rows:\n"
                                + "\n".join(f"    {line}" for line in sample_lines)
                            )
                    except Exception as e:
                        logger.warning(f"[{self.thread_id}] Failed to get sample for {table}: {e}")
        except Exception as e:
            logger.warning(f"[{self.thread_id}] Failed to get sample data context: {e}")

        return "\n\n".join(context_parts)

    def _get_source_files_description(self) -> str:
        """Build a user-friendly description of which files the data came from."""
        files = self.metadata.get("files", [])
        if not files:
            return "your uploaded data"
        file_names = [f["name"] for f in files if f.get("type") == "structured"]
        if len(file_names) == 1:
            return f"your uploaded file ({file_names[0]})"
        elif file_names:
            return f"your uploaded files ({', '.join(file_names)})"
        return "your uploaded data"

    # ══════════════════════════════════════════════════════════════
    # HYBRID RETRIEVAL (BM25 + VECTOR)
    # ══════════════════════════════════════════════════════════════

    def _bm25_retrieve(self, query: str, top_k: int = 5) -> list[TextNode]:
        """Keyword-based retrieval using BM25 — complementary to vector search."""
        if not self._all_nodes:
            return []

        try:
            from rank_bm25 import BM25Okapi

            def _tokenize(text: str) -> list[str]:
                # Basic, fast tokenizer that is robust to punctuation.
                return re.findall(r"[a-z0-9]+", (text or "").lower())

            def _normalize_query(q: str) -> str:
                # Cheap typo-fixes for common user inputs (keeps BM25 usable without embeddings).
                # NOTE: Keep this list small and obvious to avoid surprising rewrites.
                fixes = {
                    "spco": "spoc",
                    "sopc": "spoc",
                    "sopoc": "spoc",
                    "fro": "for",
                }
                toks = _tokenize(q)
                toks = [fixes.get(t, t) for t in toks]
                # Expand a few helpful synonyms when the user is asking for a point-of-contact.
                if "spoc" in toks:
                    toks.extend(["spocs", "poc", "contact", "sme", "expert"])
                return " ".join(toks)

            # Tokenize stored node texts
            corpus = [_tokenize(node.get_content()) for node in self._all_nodes]
            bm25 = BM25Okapi(corpus)

            # Score the query
            normalized_query = _normalize_query(query)
            query_tokens = _tokenize(normalized_query)
            scores = bm25.get_scores(query_tokens)

            # Get top-k nodes by BM25 score
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:top_k]

            results = [self._all_nodes[i] for i in top_indices if scores[i] > 0]
            logger.info(
                f"[{self.thread_id}] BM25 retrieved {len(results)} nodes "
                f"(top score: {max(scores):.3f})"
            )
            return results

        except Exception as e:
            logger.warning(f"[{self.thread_id}] BM25 retrieval failed: {e}")
            return []

    # ══════════════════════════════════════════════════════════════
    # CONTEXTUAL QUERY REWRITING
    # ══════════════════════════════════════════════════════════════

    def _rewrite_query(self, question: str, chat_history: list[dict] | None) -> str:
        """
        Rewrite ambiguous queries using chat history for better retrieval.

        Example: "What about tables?" → "What tables are in the uploaded document?"
        """
        if not chat_history or len(chat_history) < 2:
            return question

        # Only rewrite if the question seems ambiguous (short or uses pronouns)
        ambiguous_patterns = [
            r"^(what|how|tell|show|explain)\s+(about|me)\s",
            r"\b(it|this|that|those|these|them)\b",
            r"^(and|also|more|again)\s",
        ]
        is_ambiguous = len(question.split()) < 6 or any(
            re.search(p, question, re.IGNORECASE) for p in ambiguous_patterns
        )

        if not is_ambiguous:
            return question

        # Build condensed chat context (last 3 exchanges)
        recent = chat_history[-6:]
        context_lines = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:200]
            context_lines.append(f"{role}: {content}")

        try:
            rewrite_prompt = (
                "Given the conversation history and a follow-up question, "
                "rewrite the question to be self-contained and specific. "
                "If the question is already clear, return it as-is.\n\n"
                "Conversation:\n" + "\n".join(context_lines) + "\n\n"
                f"Follow-up question: {question}\n\n"
                "Rewritten question: "
            )
            response = self.llm.complete(rewrite_prompt)
            rewritten = str(response).strip()

            # Sanity check — don't use if it's too different or too long
            if rewritten and len(rewritten) < len(question) * 4:
                logger.info(
                    f"[{self.thread_id}] Query rewritten: "
                    f"'{question}' → '{rewritten}'"
                )
                return rewritten
        except Exception as e:
            logger.warning(f"[{self.thread_id}] Query rewriting failed: {e}")

        return question

    # ══════════════════════════════════════════════════════════════
    # QUERY ROUTING
    # ══════════════════════════════════════════════════════════════

    def _rebuild_router(self):
        """Rebuild the RouterQueryEngine whenever a new data source is added."""
        tools = []

        # ── Document search tool ─────────────────────────────────
        # Prefer hybrid/vector when available, otherwise fall back to BM25-only.
        if self.vector_index is not None and self.document_retrieval_mode != "bm25":
            from llama_index.core.retrievers import BaseRetriever
            from llama_index.core.schema import NodeWithScore
            from llama_index.core.query_engine import RetrieverQueryEngine

            # Create a true Hybrid Retriever combining Vector and BM25 results
            class HybridRetriever(BaseRetriever):
                def __init__(self, vector_retriever, bm25_func, thread_id):
                    self.vector_retriever = vector_retriever
                    self.bm25_func = bm25_func
                    self.thread_id = thread_id
                    super().__init__()

                def _retrieve(self, query_bundle):
                    vector_nodes = self.vector_retriever.retrieve(query_bundle)
                    
                    bm25_nodes = []
                    if ENABLE_HYBRID_SEARCH:
                        raw_bm25 = self.bm25_func(query_bundle.query_str, top_k=3)
                        for n in raw_bm25:
                            bm25_nodes.append(NodeWithScore(node=n, score=1.0))

                    all_nodes = []
                    seen_ids = set()
                    for node in vector_nodes + bm25_nodes:
                        if node.node.node_id not in seen_ids:
                            all_nodes.append(node)
                            seen_ids.add(node.node.node_id)
                    return all_nodes

            vector_retriever = self.vector_index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
            hybrid_retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_func=self._bm25_retrieve,
                thread_id=self.thread_id
            )

            self.vector_query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever,
                llm=self.llm,
                text_qa_template=DOCUMENT_QA_PROMPT,
            )
            tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=self.vector_query_engine,
                    name="document_search",
                    description=(
                        "Useful for searching through uploaded documents "
                        "(PDFs, text files, markdown). Use this when the user "
                        "asks questions about document content, policies, "
                        "reports, tables in documents, code snippets, "
                        "figures, charts, or any unstructured text."
                    ),
                )
            )
        elif self._all_nodes:
            from llama_index.core.retrievers import BaseRetriever
            from llama_index.core.schema import NodeWithScore
            from llama_index.core.query_engine import RetrieverQueryEngine

            class BM25OnlyRetriever(BaseRetriever):
                def __init__(self, bm25_func):
                    self.bm25_func = bm25_func
                    super().__init__()

                def _retrieve(self, query_bundle):
                    # Pull a wider net in BM25-only mode (typos / acronyms benefit).
                    raw = self.bm25_func(query_bundle.query_str, top_k=max(SIMILARITY_TOP_K, 12))
                    return [NodeWithScore(node=n, score=1.0) for n in raw]

            bm25_retriever = BM25OnlyRetriever(self._bm25_retrieve)
            self.vector_query_engine = RetrieverQueryEngine.from_args(
                retriever=bm25_retriever,
                llm=self.llm,
                text_qa_template=DOCUMENT_QA_PROMPT,
            )
            tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=self.vector_query_engine,
                    name="document_search",
                    description=(
                        "Useful for searching through uploaded documents "
                        "(PDFs, text files, markdown). Use this when the user "
                        "asks questions about document content, policies, "
                        "reports, tables in documents, code snippets, "
                        "figures, charts, or any unstructured text."
                    ),
                )
            )

        if self.sql_database is not None and self.sql_tables:
            table_info = ", ".join(self.sql_tables)
            sample_context = self._get_sample_data_context()
            source_desc = self._get_source_files_description()

            # --- Strict DuckDB Dialect Prompt ---
            sample_section = ""
            if sample_context:
                sample_section = (
                    f"\nSAMPLE DATA (use this to understand column contents):\n"
                    f"{sample_context}\n"
                )

            duckdb_prompt = PromptTemplate(
                "Given an input question, create a syntactically correct DuckDB SQL "
                "query to run, then return ONLY the SQL query.\n\n"
                "CRITICAL RULES (violating any will crash the system):\n"
                "1. NEVER use backticks (`) around table or column names.\n"
                "2. NEVER use double quotes around table or column names.\n"
                "3. Write all identifiers as plain lowercase text.\n"
                "4. End with a semicolon.\n"
                "5. Output ONLY the SQL query — no explanation, no markdown.\n\n"
                "SMART QUERY RULES:\n"
                "6. When the user asks about items by name, ALWAYS select human-readable "
                "columns (e.g. product_name, brand, category) — NOT just IDs.\n"
                "7. When listing items, include descriptive columns (name, brand, category, price) "
                "alongside any counts or aggregations.\n"
                "8. When counting unique items, report the COUNT, not every single ID.\n"
                "9. For 'what are the products', select product_name (and optionally brand, category) — "
                "NEVER just product_id.\n\n"
                "CORRECT examples:\n"
                "  SELECT product_name, brand, category FROM product;\n"
                "  SELECT COUNT(DISTINCT product_id) AS total_unique_products FROM product;\n"
                "  SELECT brand, COUNT(*) AS product_count FROM product GROUP BY brand "
                "ORDER BY product_count DESC LIMIT 5;\n"
                "  SELECT product_name, mrp_inr FROM product ORDER BY mrp_inr DESC LIMIT 10;\n\n"
                "WRONG examples (will crash or give bad results):\n"
                "  SELECT * FROM `product`;                    -- backticks crash DuckDB\n"
                "  SELECT DISTINCT product_id FROM product;    -- IDs are meaningless to users\n\n"
                "Use only the tables and columns from the schema below.\n"
                "{schema}\n"
                + sample_section +
                "\nQuestion: {query_str}\n"
                "SQLQuery: "
            )

            # --- Response synthesis prompt ---
            data_label = profile.data_source_label
            response_synthesis_prompt = PromptTemplate(
                "You are a helpful data analyst assistant. The user uploaded a data file "
                "and asked a question. You analyzed their data and got a result.\n\n"
                "Now compose a clear, friendly, natural language response.\n\n"
                "ABSOLUTE RULES:\n"
                "1. Answer the question directly using the data result provided.\n"
                "2. Include specific numbers, names, and values — be precise.\n"
                "3. NEVER mention 'database', 'table', 'SQL', 'query', or any technical terms.\n"
                "4. NEVER say 'run this query', 'execute', or suggest the user do anything technical.\n"
                "5. NEVER show any SQL code in your response.\n"
                f"6. Refer to the data source as '{data_label}' — "
                "NEVER as 'database' or 'table'.\n"
                "7. Use friendly formatting: bullet points, bold numbers, emojis where appropriate.\n"
                "8. If listing items, show names/descriptions — NEVER raw IDs.\n\n"
                "Question: {query_str}\n"
                "SQL Query (internal, DO NOT show this): {sql_query}\n"
                "Data Result: {context_str}\n\n"
                "Answer: "
            )

            self.sql_query_engine = NLSQLTableQueryEngine(
                sql_database=self.sql_database,
                tables=self.sql_tables,
                llm=self.llm,
                text_to_sql_prompt=duckdb_prompt,
                response_synthesis_prompt=response_synthesis_prompt,
            )
            tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=self.sql_query_engine,
                    name="data_analysis",
                    description=(
                        f"Useful for analyzing structured data in tables: "
                        f"[{table_info}]. Use this when the user asks "
                        f"about numbers, statistics, totals, comparisons, "
                        f"rankings, or any data-related questions from "
                        f"uploaded Excel/CSV/JSON files."
                    ),
                )
            )

        if not tools:
            self.router_query_engine = None
            return

        if len(tools) == 1:
            self.router_query_engine = tools[0].query_engine
            logger.info(f"[{self.thread_id}] Single engine active: {tools[0].metadata.name}")
        else:
            self.router_query_engine = RouterQueryEngine(
                selector=LLMSingleSelector.from_defaults(llm=self.llm),
                query_engine_tools=tools,
            )
            logger.info(
                f"[{self.thread_id}] Router rebuilt with "
                f"{len(tools)} engines: "
                f"{[t.metadata.name for t in tools]}"
            )

    # ══════════════════════════════════════════════════════════════
    # DIRECT SQL EXECUTION (FALLBACK)
    # ══════════════════════════════════════════════════════════════

    def _fix_sql_syntax(self, sql: str) -> str:
        """Fix common LLM SQL mistakes for DuckDB compatibility."""
        # Remove backticks (MySQL habit)
        sql = sql.replace("`", "")
        # Remove markdown code fences
        sql = re.sub(r"```(?:sql)?\s*", "", sql)
        sql = sql.strip().rstrip(";")
        return sql + ";"

    def _extract_sql_from_text(self, text: str) -> str | None:
        """Try to extract a SQL query from LLM text that contains one."""
        # Try code block first
        code_match = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()

        # Try to find a SELECT statement
        select_match = re.search(
            r"(SELECT\s+.+?;)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if select_match:
            return select_match.group(1).strip()

        # Check if the whole text is basically a SQL query
        stripped = text.strip()
        if stripped.upper().startswith("SELECT"):
            return stripped

        return None

    def _execute_sql_directly(self, sql: str) -> str | None:
        """Execute a SQL query directly against DuckDB and format results."""
        if self.sql_engine is None:
            return None

        fixed_sql = self._fix_sql_syntax(sql)
        logger.info(f"[{self.thread_id}] Direct SQL execution: {fixed_sql[:200]}")

        try:
            with self.sql_engine.connect() as conn:
                result = conn.execute(text(fixed_sql))
                columns = list(result.keys())
                rows = result.fetchall()

                if not rows:
                    return "The query ran successfully but returned no results."

                # Single value result (e.g. COUNT, AVG, SUM)
                if len(columns) == 1 and len(rows) == 1:
                    value = rows[0][0]
                    return f"**Result:** {value}"

                # Build a markdown table for multi-row / multi-column results
                lines = []
                lines.append("| " + " | ".join(str(c) for c in columns) + " |")
                lines.append("| " + " | ".join("---" for _ in columns) + " |")
                for row in rows[:50]:  # Limit to 50 rows
                    lines.append("| " + " | ".join(str(v) for v in row) + " |")
                if len(rows) > 50:
                    lines.append(f"\n*...and {len(rows) - 50} more rows*")

                return "\n".join(lines)

        except Exception as e:
            logger.warning(f"[{self.thread_id}] Direct SQL failed: {e}")
            return None

    def _looks_like_sql_not_answer(self, text: str) -> bool:
        """Detect if the LLM returned SQL instead of a natural language answer."""
        stripped = text.strip().upper()

        # Starts with SQL keywords
        if stripped.startswith(("SELECT ", "WITH ")):
            return True

        # Contains a SQL code block
        if re.search(r"```(?:sql)?\s*SELECT", text, re.IGNORECASE):
            return True

        # Contains phrases suggesting the user should run SQL themselves
        suggestion_patterns = [
            r"(?:run|execute|try)\s+(?:this|the)\s+(?:query|sql)",
            r"here\s*(?:is|'s)\s+(?:the|a)\s+(?:query|sql)",
            r"you\s+can\s+(?:run|execute|use)",
            r"the\s+(?:correct|valid|proper)\s+(?:query|sql)",
        ]
        for pattern in suggestion_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _sanitize_answer(self, answer: str) -> str:
        """Post-process: replace technical jargon with user-friendly language."""
        source_desc = self._get_source_files_description()

        # Replace "database" references with profile-appropriate labels
        data_label = profile.data_source_label
        replacements = [
            (r'\b[Tt]he database\b', data_label),
            (r'\b[Ii]n the database\b', f'in {data_label}'),
            (r'\b[Ff]rom the database\b', f'from {data_label}'),
            (r'\b[Oo]ur database\b', data_label),
            (r'\b[Tt]he table\b', data_label),
            (r'\b[Ii]n the table\b', f'in {data_label}'),
            (r'\b[Ff]rom the table\b', f'from {data_label}'),
            (r'\baccording to the database\b', f'based on {data_label}'),
            (r'\bAccording to the database\b', f'Based on {data_label}'),
        ]
        for pattern, replacement in replacements:
            answer = re.sub(pattern, replacement, answer)

        # Light cleanup for internal RAG jargon (keeps answers feeling natural).
        tech_cleanup = [
            (r"\b[Rr]etrieval-augmented generation\b", "document search"),
            (r"\bRAG\b", "document search"),
            (r"\b[Bb]m25\b", "keyword search"),
            (r"\b[Vv]ector (?:search|index|store)\b", "search"),
            (r"\b[Ee]mbeddings?\b", "search signals"),
            (r"\b[Cc]hunks?\b", "sections"),
        ]
        for pattern, replacement in tech_cleanup:
            answer = re.sub(pattern, replacement, answer)

        return answer

    # ══════════════════════════════════════════════════════════════
    # SOURCE CITATION EXTRACTION
    # ══════════════════════════════════════════════════════════════

    def _extract_source_citations(self, response) -> str:
        """Extract source citations from query response metadata."""
        if not SHOW_SOURCES:
            return ""

        citations = []
        try:
            if hasattr(response, 'source_nodes'):
                seen = set()
                for node in response.source_nodes:
                    meta = node.node.metadata if hasattr(node, 'node') else {}
                    filename = meta.get('source_filename', meta.get('file_name', ''))
                    page = meta.get('page_label', meta.get('page_number', ''))

                    if filename:
                        cite_key = f"{filename}:{page}"
                        if cite_key not in seen:
                            seen.add(cite_key)
                            if page:
                                citations.append(f"📄 {filename}, page {page}")
                            else:
                                citations.append(f"📄 {filename}")
        except Exception as e:
            logger.debug(f"[{self.thread_id}] Citation extraction issue: {e}")

        if citations:
            # Keep the answer readable: short, scannable sources list.
            lines = ["", "Sources:"]
            for c in citations[:5]:
                lines.append(f"- {c}")
            return "\n" + "\n".join(lines)
        return ""

    # ══════════════════════════════════════════════════════════════
    # QUERYING — PUBLIC API
    # ══════════════════════════════════════════════════════════════

    async def query(self, question: str, chat_history: list[dict] | None = None) -> str:
        if self.router_query_engine is None:
            return (
                "⚠️ No data has been loaded yet. "
                "Please upload a file first to start querying."
            )

        # ── Step 1: Cheap typo normalization (helps BM25-only mode) ────────
        # Keep this intentionally small to avoid surprising rewrites.
        enhanced_question = question
        enhanced_question = re.sub(r"\bspco\b", "spoc", enhanced_question, flags=re.IGNORECASE)
        enhanced_question = re.sub(r"\bfro\b", "for", enhanced_question, flags=re.IGNORECASE)

        # ── Step 2: Contextual query rewriting (LLM) ───────────────────────
        enhanced_question = self._rewrite_query(enhanced_question, chat_history)
        logger.info(f"[{self.thread_id}] Query: {enhanced_question[:100]}...")

        # (Hybrid search is now handled natively via the HybridRetriever in the router)
        try:
            response = await self.router_query_engine.aquery(enhanced_question)
            answer   = str(response).strip()

            # --- Check metadata for SQL results first ---
            if response.metadata and "result" in response.metadata:
                sql_query   = str(response.metadata.get("sql_query", "")).strip()
                result_data = response.metadata.get("result", [])

                # If the answer looks like SQL rather than a natural language answer
                if self._looks_like_sql_not_answer(answer):
                    logger.warning(
                        f"[{self.thread_id}] LLM returned SQL instead of answer, "
                        f"using metadata result directly"
                    )
                    if result_data:
                        # Try to format nicely
                        if isinstance(result_data, list) and len(result_data) == 1:
                            # Single row — extract the value
                            row = result_data[0]
                            if isinstance(row, (tuple, list)) and len(row) == 1:
                                answer = f"Based on your data, the answer is: **{row[0]}**"
                            else:
                                answer = f"**Result:** {row}"
                        else:
                            answer = f"**Results from your data:**\n```\n{result_data}\n```"
                    else:
                        answer = "The query ran successfully but returned no matching data."

            # --- Broader fallback: answer still looks like SQL ---
            if self._looks_like_sql_not_answer(answer):
                logger.warning(
                    f"[{self.thread_id}] Answer still looks like SQL, "
                    f"attempting direct execution"
                )
                extracted_sql = self._extract_sql_from_text(answer)
                if extracted_sql and self.sql_tables:
                    direct_result = self._execute_sql_directly(extracted_sql)
                    if direct_result:
                        answer = f"📊 {direct_result}"

            # --- Post-process: clean technical jargon from the answer ---
            answer = self._sanitize_answer(answer)

            logger.info(f"[{self.thread_id}] Query answered ({len(answer)} chars)")
            return answer
        except Exception as e:
            # --- Last resort: try to generate and execute SQL directly ---
            error_str = str(e)
            logger.error(f"[{self.thread_id}] Query engine failed: {error_str}", exc_info=True)

            # If DuckDB syntax error, try fixing and re-executing
            if self.sql_tables and ("syntax error" in error_str.lower() or "Parser Error" in error_str):
                extracted = self._extract_sql_from_text(error_str)
                if extracted:
                    direct_result = self._execute_sql_directly(extracted)
                    if direct_result:
                        return f"📊 {direct_result}"

            return (
                "❌ I encountered an error while querying your data. "
                "Please try rephrasing your question."
            )

    # ══════════════════════════════════════════════════════════════
    # STATE INSPECTION — PUBLIC API
    # ══════════════════════════════════════════════════════════════

    def has_data(self) -> bool:
        return bool(self._all_nodes) or self.vector_index is not None or len(self.sql_tables) > 0

    def get_loaded_files_summary(self) -> str:
        files = self.metadata.get("files", [])
        if not files:
            return "No files loaded."

        lines = ["📁 **Loaded Files:**\n"]
        for f in files:
            if f["type"] == "unstructured":
                content_types = f.get("content_types", [])
                type_str = f" (content: {', '.join(content_types)})" if content_types else ""
                lines.append(
                    f"- 📄 **{f['name']}** — {f.get('chunks', '?')} chunks "
                    f"(indexed for search){type_str}"
                )
            elif f["type"] == "structured":
                tables = ", ".join(f.get("tables", []))
                lines.append(
                    f"- 📊 **{f['name']}** — Tables: [{tables}], "
                    f"{f.get('total_rows', '?')} total rows"
                )
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════
    # PERSISTENCE & RESUME — CLASS METHOD
    # ══════════════════════════════════════════════════════════════

    @classmethod
    def load_from_storage(cls, thread_id: str, embed_model=None) -> Optional["RAGEngine"]:
        """
        Attempt to restore a RAGEngine from persisted storage.

        No llm argument needed — LLM is resolved from Settings.llm.
        """
        chroma_dir  = CHROMA_DB_DIR / thread_id
        duckdb_path = DUCKDB_DIR / thread_id / "structured.duckdb"

        has_vectors = chroma_dir.exists()
        has_duckdb  = duckdb_path.exists()
        has_bm25    = (DUCKDB_DIR / thread_id / "bm25_nodes.jsonl").exists()

        if not has_vectors and not has_duckdb and not has_bm25:
            logger.info(f"[{thread_id}] No RAG storage found for resume")
            return None

        engine = cls(thread_id=thread_id, embed_model=embed_model)

        if engine.has_data():
            logger.info(
                f"[{thread_id}] RAGEngine restored from storage "
                f"(vectors={has_vectors}, tables={engine.sql_tables})"
            )
            return engine

        logger.info(f"[{thread_id}] Storage exists but no queryable data")
        return None

    # ══════════════════════════════════════════════════════════════
    # CLEANUP
    # ══════════════════════════════════════════════════════════════

    def close(self):
        """Clean up all connections."""
        if self.sql_engine is not None:
            try:
                self.sql_engine.dispose()
            except Exception:
                pass
            self.sql_engine = None
        self.sql_database        = None
        self.router_query_engine = None
        self._all_nodes          = []
        logger.info(f"[{self.thread_id}] RAGEngine closed")
