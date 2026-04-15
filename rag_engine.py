"""
rag_engine.py — Genius AI RAG Engine

Single-file RAG abstraction managing two pipelines:
  1. Unstructured (PDF/MD/TXT) → ChromaDB vector search
  2. Structured (Excel/CSV/JSON) → DuckDB SQL engine

Design Principles:
  - One RAGEngine instance per chat session
  - Thread-scoped persistence under ./rag_storage/{thread_id}/
  - Zero dependency on Chainlit internals (clean separation)
  - Incremental ingestion — files can be added mid-conversation
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

import chromadb
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# ── Configuration Constants ───────────────────────────────────────

RAG_STORAGE_DIR = "./rag_storage"

# Embedding
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval
SIMILARITY_TOP_K = 5

# Limits
MAX_FILE_SIZE_MB = 50
MAX_FILES_PER_SESSION = 20
MAX_DATAFRAME_ROWS = 500_000  # Safety limit for structured files

# SQL Safety
SQL_QUERY_TIMEOUT_SECONDS = 30

# Supported Extensions
UNSTRUCTURED_EXTENSIONS = {".pdf", ".txt", ".md"}
STRUCTURED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".json"}


class RAGEngine:
    """Per-session RAG engine managing ChromaDB and DuckDB pipelines."""

    def __init__(self, thread_id: str, llm, embed_model=None):
        """
        Initialize with thread ID for persistence scoping.

        Args:
            thread_id: Unique chat thread identifier (used for storage paths).
            llm: LlamaIndex LLM instance (e.g., Groq).
            embed_model: Optional custom embedding model. Defaults to BAAI/bge-small-en-v1.5.
        """
        self.thread_id = thread_id
        self.llm = llm

        # ── Storage paths ─────────────────────────────────────────
        self.storage_dir = Path(RAG_STORAGE_DIR) / thread_id
        self.chroma_dir = self.storage_dir / "chroma_db"
        self.duckdb_path = self.storage_dir / "structured.duckdb"
        self.source_files_dir = self.storage_dir / "source_files"
        self.metadata_path = self.storage_dir / "metadata.json"

        # Ensure directories exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.source_files_dir.mkdir(parents=True, exist_ok=True)

        # ── Embedding model ───────────────────────────────────────
        self.embed_model = embed_model or HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME,
        )

        # ── Set global LlamaIndex defaults (prevents OpenAI fallback) ─
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # ── ChromaDB (Vector Store) ───────────────────────────────
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
        )
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            "documents",
        )
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection,
        )
        self.vector_index: Optional[VectorStoreIndex] = None

        # Build index from existing collection if it has data
        if self.chroma_collection.count() > 0:
            self.vector_index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embed_model,
            )
            logger.info(
                f"[{thread_id}] Restored ChromaDB vector index "
                f"({self.chroma_collection.count()} vectors)"
            )

        # ── DuckDB (Structured SQL) ───────────────────────────────
        self.sql_engine = None
        self.sql_database: Optional[SQLDatabase] = None
        self.sql_tables: list[str] = []

        # Reconnect if DB file exists (uses SQLAlchemy only)
        if self.duckdb_path.exists():
            self._connect_duckdb()
            logger.info(
                f"[{thread_id}] Reconnected to DuckDB "
                f"(tables: {self.sql_tables})"
            )

        # ── Query Routing ─────────────────────────────────────────
        self.vector_query_engine = None
        self.sql_query_engine = None
        self.router_query_engine: Optional[RouterQueryEngine] = None

        # ── File tracking ─────────────────────────────────────────
        self.metadata = self._load_metadata()

        # Build router if we already have data
        if self.vector_index or self.sql_tables:
            self._rebuild_router()

        logger.info(f"[{thread_id}] RAGEngine initialized")

    # ══════════════════════════════════════════════════════════════
    # FILE CLASSIFICATION & VALIDATION
    # ══════════════════════════════════════════════════════════════

    def _classify_file(self, file_name: str) -> str:
        """
        Classify a file as 'unstructured', 'structured', or 'unsupported'.

        Args:
            file_name: Name of the file (with extension).

        Returns:
            One of: 'unstructured', 'structured', 'unsupported'
        """
        ext = Path(file_name).suffix.lower()
        if ext in UNSTRUCTURED_EXTENSIONS:
            return "unstructured"
        elif ext in STRUCTURED_EXTENSIONS:
            return "structured"
        else:
            return "unsupported"

    def _validate_file(self, file_path: str, file_name: str) -> Optional[str]:
        """
        Validate a file before ingestion.

        Returns:
            None if valid, or an error message string if invalid.
        """
        path = Path(file_path)

        # Check existence
        if not path.exists():
            return f"❌ File not found: `{file_name}`"

        # Check size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return (
                f"❌ File `{file_name}` is too large ({size_mb:.1f} MB). "
                f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
            )

        # Check empty
        if path.stat().st_size == 0:
            return f"❌ File `{file_name}` is empty."

        # Check total file count in session
        total_files = len(self.metadata.get("files", []))
        if total_files >= MAX_FILES_PER_SESSION:
            return (
                f"❌ Maximum files per session reached ({MAX_FILES_PER_SESSION}). "
                f"Please start a new chat to upload more files."
            )

        # Check extension
        classification = self._classify_file(file_name)
        if classification == "unsupported":
            supported = ", ".join(
                sorted(UNSTRUCTURED_EXTENSIONS | STRUCTURED_EXTENSIONS)
            )
            return (
                f"❌ Unsupported file type for `{file_name}`. "
                f"Supported formats: {supported}"
            )

        return None  # Valid

    # ══════════════════════════════════════════════════════════════
    # FILE PERSISTENCE & METADATA
    # ══════════════════════════════════════════════════════════════

    def _copy_source_file(self, file_path: str, file_name: str) -> Path:
        """
        Copy an uploaded file to the persistent source_files directory.

        Returns:
            Path to the copied file.
        """
        dest = self.source_files_dir / file_name

        # Handle duplicate filenames by appending a number
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 2
            while dest.exists():
                dest = self.source_files_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.copy2(file_path, dest)
        logger.info(f"[{self.thread_id}] Copied source file to {dest}")
        return dest

    def _load_metadata(self) -> dict:
        """Load or initialize the metadata.json file."""
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
        """Persist metadata.json to disk."""
        self.metadata["vector_index_exists"] = self.vector_index is not None
        self.metadata["sql_tables"] = list(self.sql_tables)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"[{self.thread_id}] Metadata saved")

    def _add_file_metadata(
        self,
        file_name: str,
        file_type: str,
        *,
        chunks: int = 0,
        tables: list[str] | None = None,
        total_rows: int = 0,
    ):
        """Add a file entry to the metadata manifest."""
        entry = {
            "name": file_name,
            "type": file_type,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        if file_type == "unstructured":
            entry["chunks"] = chunks
        elif file_type == "structured":
            entry["tables"] = tables or []
            entry["total_rows"] = total_rows

        self.metadata["files"].append(entry)
        self._save_metadata()

    # ══════════════════════════════════════════════════════════════
    # DUCKDB CONNECTION MANAGEMENT
    # ══════════════════════════════════════════════════════════════

    def _connect_duckdb(self):
        """
        Establish connection to the DuckDB database file.

        Uses SQLAlchemy as the sole persistent connection to avoid
        DuckDB's dual-connection config conflict. Native duckdb is
        used only transiently for DataFrame loading via _load_df_to_duckdb.
        """
        # Dispose any existing engine to avoid connection conflicts
        if self.sql_engine is not None:
            self.sql_engine.dispose()

        # Create SQLAlchemy engine (sole persistent connection)
        self.sql_engine = create_engine(
            f"duckdb:///{self.duckdb_path}",
            pool_pre_ping=True,
        )

        # Discover existing tables via SQLAlchemy
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

    # ══════════════════════════════════════════════════════════════
    # TABLE NAME & DATAFRAME CLEANING
    # ══════════════════════════════════════════════════════════════

    def _clean_table_name(self, name: str) -> str:
        """
        Sanitize a sheet/file name into a valid SQL table name.

        Examples:
            "Sales Data (Q4)" → "sales_data_q4"
            "Sheet1" → "sheet1"
            "my-file.csv" → "my_file"
        """
        # Remove file extension if present
        name = Path(name).stem if "." in name else name
        # Replace non-alphanumeric chars with underscores
        name = re.sub(r"[^a-zA-Z0-9]", "_", name)
        # Collapse multiple underscores
        name = re.sub(r"_+", "_", name)
        # Strip leading/trailing underscores and lowercase
        name = name.strip("_").lower()
        # Ensure it doesn't start with a number
        if name and name[0].isdigit():
            name = f"t_{name}"
        # Fallback for empty names
        if not name:
            name = "unnamed_table"
        return name

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a DataFrame for SQL ingestion:
          - Lowercase column names
          - Replace special chars with underscores
          - Drop fully-null columns
          - Strip whitespace from string columns
        """
        # Clean column names
        new_cols = []
        for col in df.columns:
            cleaned = re.sub(r"[^a-zA-Z0-9]", "_", str(col))
            cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
            if cleaned and cleaned[0].isdigit():
                cleaned = f"col_{cleaned}"
            if not cleaned:
                cleaned = f"col_{len(new_cols)}"
            new_cols.append(cleaned)

        # Handle duplicate column names
        seen = {}
        final_cols = []
        for col in new_cols:
            if col in seen:
                seen[col] += 1
                final_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_cols.append(col)

        df.columns = final_cols

        # Drop fully-null columns
        df = df.dropna(axis=1, how="all")

        # Strip whitespace from string columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        return df

    # ══════════════════════════════════════════════════════════════
    # FILE INGESTION — PUBLIC API
    # ══════════════════════════════════════════════════════════════

    async def ingest_file(self, file_path: str, file_name: str) -> str:
        """
        Route a file to the correct ingestion pipeline.

        Args:
            file_path: Absolute or relative path to the uploaded file.
            file_name: Original file name (used for classification & display).

        Returns:
            A user-facing status message string.
        """
        # Validate
        error = self._validate_file(file_path, file_name)
        if error:
            logger.warning(f"[{self.thread_id}] Validation failed for {file_name}: {error}")
            return error

        # Copy to persistent storage
        self._copy_source_file(file_path, file_name)

        # Classify and route
        file_type = self._classify_file(file_name)
        logger.info(f"[{self.thread_id}] Ingesting file: {file_name} ({file_type})")

        try:
            if file_type == "unstructured":
                result = await self._ingest_unstructured(file_path, file_name)
            else:
                result = await self._ingest_structured(file_path, file_name)

            # Rebuild query router with new data sources
            self._rebuild_router()
            return result

        except Exception as e:
            error_msg = (
                f"❌ Failed to process `{file_name}`: {type(e).__name__}: {str(e)}"
            )
            logger.error(f"[{self.thread_id}] {error_msg}", exc_info=True)
            return error_msg

    # ══════════════════════════════════════════════════════════════
    # UNSTRUCTURED INGESTION (PDF, MD, TXT → ChromaDB)
    # ══════════════════════════════════════════════════════════════

    async def _ingest_unstructured(self, file_path: str, file_name: str) -> str:
        """
        Ingest an unstructured document into the ChromaDB vector store.

        Steps:
          1. Parse document using SimpleDirectoryReader
          2. Chunk into nodes with SentenceSplitter
          3. Tag with metadata (source, time, thread)
          4. Embed and insert into ChromaDB via VectorStoreIndex
        """
        # Parse the document
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

        if not documents:
            return f"⚠️ No content could be extracted from `{file_name}`."

        # Add metadata to each document
        for doc in documents:
            doc.metadata.update({
                "source_filename": file_name,
                "upload_time": datetime.now(timezone.utc).isoformat(),
                "thread_id": self.thread_id,
            })

        # Chunk documents into nodes
        splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        nodes = splitter.get_nodes_from_documents(documents)

        logger.info(
            f"[{self.thread_id}] Created {len(nodes)} chunks from {file_name}"
        )

        # Build or update the vector index
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
        )

        if self.vector_index is None:
            # First file — create new index
            self.vector_index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
        else:
            # Subsequent files — insert into existing index
            self.vector_index.insert_nodes(nodes)

        # Update metadata
        self._add_file_metadata(file_name, "unstructured", chunks=len(nodes))

        return (
            f"📄 **{file_name}** processed successfully!\n\n"
            f"- **{len(nodes)}** document chunks indexed for semantic search\n"
            f"- You can now ask questions about this document's content."
        )

    # ══════════════════════════════════════════════════════════════
    # STRUCTURED INGESTION (Excel, CSV, JSON → DuckDB)
    # ══════════════════════════════════════════════════════════════

    async def _ingest_structured(self, file_path: str, file_name: str) -> str:
        """
        Ingest structured data into DuckDB.

        Steps:
          1. Load file with Pandas (auto-detect format)
          2. Clean DataFrame (column names, nulls, types)
          3. Create SQL table in DuckDB from DataFrame
          4. Rebuild NLSQLTableQueryEngine
        """
        ext = Path(file_name).suffix.lower()
        tables_created = []
        total_rows = 0

        try:
            if ext in (".xlsx", ".xls"):
                # Excel: each sheet becomes a table
                sheets = pd.read_excel(file_path, sheet_name=None)
                for sheet_name, df in sheets.items():
                    if df.empty:
                        logger.warning(
                            f"[{self.thread_id}] Empty sheet skipped: {sheet_name}"
                        )
                        continue
                    if len(df) > MAX_DATAFRAME_ROWS:
                        logger.warning(
                            f"[{self.thread_id}] Sheet {sheet_name} truncated "
                            f"to {MAX_DATAFRAME_ROWS} rows"
                        )
                        df = df.head(MAX_DATAFRAME_ROWS)

                    table_name = self._clean_table_name(sheet_name)
                    df = self._clean_dataframe(df)
                    self._load_df_to_duckdb(df, table_name)
                    tables_created.append((table_name, len(df), list(df.columns)))
                    total_rows += len(df)

            elif ext == ".csv":
                # CSV: file name becomes table name
                # Try multiple encodings
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
                tables_created.append((table_name, len(df), list(df.columns)))
                total_rows = len(df)

            elif ext == ".json":
                # JSON: attempt to normalize nested structures
                try:
                    with open(file_path, "r") as f:
                        raw = json.load(f)
                    if isinstance(raw, list):
                        df = pd.json_normalize(raw)
                    elif isinstance(raw, dict):
                        # Try to find the first list value
                        for key, val in raw.items():
                            if isinstance(val, list):
                                df = pd.json_normalize(val)
                                break
                        else:
                            df = pd.json_normalize([raw])
                    else:
                        return f"❌ Unsupported JSON structure in `{file_name}`."
                except json.JSONDecodeError as e:
                    return f"❌ Invalid JSON in `{file_name}`: {str(e)}"

                if len(df) > MAX_DATAFRAME_ROWS:
                    df = df.head(MAX_DATAFRAME_ROWS)

                table_name = self._clean_table_name(file_name)
                df = self._clean_dataframe(df)
                self._load_df_to_duckdb(df, table_name)
                tables_created.append((table_name, len(df), list(df.columns)))
                total_rows = len(df)

        except Exception as e:
            return (
                f"❌ Error reading `{file_name}`: {type(e).__name__}: {str(e)}"
            )

        if not tables_created:
            return f"⚠️ No data could be extracted from `{file_name}`."

        # Update metadata
        table_names = [t[0] for t in tables_created]
        self._add_file_metadata(
            file_name, "structured",
            tables=table_names,
            total_rows=total_rows,
        )

        # Build the response message
        lines = [f"📊 **{file_name}** loaded successfully!\n"]
        lines.append("| Table | Rows | Columns |")
        lines.append("|-------|------|---------|")
        for tname, nrows, cols in tables_created:
            col_preview = ", ".join(cols[:5])
            if len(cols) > 5:
                col_preview += f", ... (+{len(cols) - 5} more)"
            lines.append(f"| `{tname}` | {nrows:,} | {len(cols)} ({col_preview}) |")
        lines.append(
            "\n💡 You can now ask analytical questions about this data!"
        )
        return "\n".join(lines)

    def _load_df_to_duckdb(self, df: pd.DataFrame, table_name: str):
        """
        Load a cleaned DataFrame into DuckDB as a SQL table.

        Uses a transient native duckdb connection for high-perf DataFrame
        loading, then closes it so the SQLAlchemy engine can reconnect.
        """
        # Dispose SQLAlchemy engine temporarily to free the file lock
        if self.sql_engine is not None:
            self.sql_engine.dispose()
            self.sql_engine = None

        # Use transient native duckdb for DataFrame loading (fast, zero-copy)
        conn = duckdb.connect(str(self.duckdb_path))
        try:
            conn.register("_temp_df", df)
            conn.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _temp_df'
            )
            conn.unregister("_temp_df")
        finally:
            conn.close()

        # Track the table
        if table_name not in self.sql_tables:
            self.sql_tables.append(table_name)

        # Re-establish SQLAlchemy engine and refresh SQLDatabase for LlamaIndex
        self._connect_duckdb()

        logger.info(
            f"[{self.thread_id}] Loaded table '{table_name}' "
            f"with {len(df)} rows, {len(df.columns)} cols"
        )

    # ══════════════════════════════════════════════════════════════
    # QUERY ROUTING
    # ══════════════════════════════════════════════════════════════

    def _rebuild_router(self):
        """
        Rebuild the RouterQueryEngine whenever a new data source is added.

        Creates QueryEngineTools for each available engine and combines
        them under an LLM-based router.
        """
        tools = []

        # Vector search tool
        if self.vector_index is not None:
            self.vector_query_engine = self.vector_index.as_query_engine(
                similarity_top_k=SIMILARITY_TOP_K,
                llm=self.llm,
            )
            tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=self.vector_query_engine,
                    name="document_search",
                    description=(
                        "Useful for searching through uploaded documents "
                        "(PDFs, text files, markdown). Use this when the user "
                        "asks questions about document content, policies, "
                        "reports, or any unstructured text."
                    ),
                )
            )

        # SQL tool
        if self.sql_database is not None and self.sql_tables:
            table_info = ", ".join(self.sql_tables)
            self.sql_query_engine = NLSQLTableQueryEngine(
                sql_database=self.sql_database,
                tables=self.sql_tables,
                llm=self.llm,
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

        if len(tools) == 0:
            self.router_query_engine = None
            return

        if len(tools) == 1:
            # Only one engine available — use it directly
            self.router_query_engine = tools[0].query_engine
            logger.info(
                f"[{self.thread_id}] Single engine active: {tools[0].metadata.name}"
            )
        else:
            # Multiple engines — use LLM router
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
    # QUERYING — PUBLIC API
    # ══════════════════════════════════════════════════════════════

    async def query(self, question: str, chat_history: list[dict] | None = None) -> str:
        """
        Query the RAG system.

        Routes the question through the RouterQueryEngine which selects
        the appropriate backend (vector search or SQL).

        Args:
            question: The user's natural language question.
            chat_history: Optional conversation history for context.

        Returns:
            The synthesized answer string.
        """
        if self.router_query_engine is None:
            return (
                "⚠️ No data has been loaded yet. "
                "Please upload a file first to start querying."
            )

        logger.info(f"[{self.thread_id}] Query: {question[:100]}...")

        try:
            response = await self.router_query_engine.aquery(question)
            answer = str(response)

            logger.info(
                f"[{self.thread_id}] Query answered "
                f"({len(answer)} chars)"
            )
            return answer

        except Exception as e:
            error_msg = f"❌ Query failed: {type(e).__name__}: {str(e)}"
            logger.error(f"[{self.thread_id}] {error_msg}", exc_info=True)
            return error_msg

    # ══════════════════════════════════════════════════════════════
    # STATE INSPECTION — PUBLIC API
    # ══════════════════════════════════════════════════════════════

    def has_data(self) -> bool:
        """Check if any files have been ingested in this session."""
        has_vectors = self.vector_index is not None
        has_tables = len(self.sql_tables) > 0
        return has_vectors or has_tables

    def get_loaded_files_summary(self) -> str:
        """Return a human-readable summary of all loaded files and tables."""
        files = self.metadata.get("files", [])
        if not files:
            return "No files loaded."

        lines = ["📁 **Loaded Files:**\n"]
        for f in files:
            if f["type"] == "unstructured":
                lines.append(
                    f"- 📄 **{f['name']}** — {f.get('chunks', '?')} chunks "
                    f"(indexed for search)"
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
    def load_from_storage(cls, thread_id: str, llm, embed_model=None) -> Optional["RAGEngine"]:
        """
        Attempt to restore a RAGEngine from persisted storage.

        Used during chat resume to reconnect to previously uploaded data.

        Args:
            thread_id: The chat thread ID to restore.
            llm: LlamaIndex LLM instance.
            embed_model: Optional embedding model override.

        Returns:
            A restored RAGEngine instance, or None if no storage exists.
        """
        storage_dir = Path(RAG_STORAGE_DIR) / thread_id

        if not storage_dir.exists():
            logger.info(f"[{thread_id}] No RAG storage found for resume")
            return None

        # Check if there's actually any data
        chroma_dir = storage_dir / "chroma_db"
        duckdb_path = storage_dir / "structured.duckdb"

        has_vectors = chroma_dir.exists()
        has_duckdb = duckdb_path.exists()

        if not has_vectors and not has_duckdb:
            logger.info(f"[{thread_id}] Storage dir exists but no data found")
            return None

        # The constructor handles reconnection automatically
        engine = cls(thread_id=thread_id, llm=llm, embed_model=embed_model)

        if engine.has_data():
            logger.info(
                f"[{thread_id}] RAGEngine restored from storage "
                f"(vectors={has_vectors}, tables={engine.sql_tables})"
            )
            return engine
        else:
            logger.info(f"[{thread_id}] Storage exists but no queryable data")
            return None

    # ══════════════════════════════════════════════════════════════
    # CLEANUP
    # ══════════════════════════════════════════════════════════════

    def close(self):
        """Clean up all connections (SQLAlchemy engine, DuckDB, ChromaDB)."""
        if self.sql_engine is not None:
            try:
                self.sql_engine.dispose()
            except Exception:
                pass
            self.sql_engine = None
        self.sql_database = None
        self.router_query_engine = None
        logger.info(f"[{self.thread_id}] RAGEngine closed")
