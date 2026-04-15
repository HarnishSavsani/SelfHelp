# Genius AI - Structured & Unstructured RAG

A production-ready RAG (Retrieval-Augmented Generation) assistant built with Chainlit and LlamaIndex. It handles both structured data (Excel, CSV, JSON) using DuckDB and unstructured documents (PDF, MD, TXT) using ChromaDB.

## 🚀 Quick Start with `uv`

We recommend using [uv](https://github.com/astral-sh/uv) for extremely fast environment management.

### 1. Install `uv` (if not present)

If you don't have `uv` installed, run one of the following:

**macOS/Linux:**
```bash
curl -LsSf https://astral-sh/uv/install.sh | sh
```

**Homebrew:**
```bash
brew install uv
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral-sh/uv/install.ps1 | iex"
```

### 2. Setup Environment

Once `uv` is installed, run the following to create a virtual environment and install dependencies:

```bash
# Recreate venv and install dependencies from pyproject.toml
uv sync

# OR using requirements.txt
uv pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your details:

```bash
cp .env.example .env
```

Ensure `OLLAMA_MODEL` is set to `qwen2.5:7b` (or your preferred model).

### 4. Run the Application

```bash
source .venv/bin/activate
chainlit run app.py
```

## 🛠 Manual Setup (without `uv`)

If you prefer using standard `pip`:

1.  **Create venv**: `python -m venv .venv`
2.  **Activate**: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
3.  **Install**: `pip install -r requirements.txt`

## 📦 Features

-   **Structured RAG**: Powered by DuckDB. Automatically detects schemas and performs NL-to-SQL.
-   **Unstructured RAG**: Powered by ChromaDB. Semantic search across PDFs and text files.
-   **Smart Fallbacks**: Robust error handling for SQL generation and execution.
-   **Production Synthesis**: Natural language responses that avoid technical jargon (e.g., "database", "SQL").

## 📂 Project Structure

-   `app.py`: Main Chainlit application and UI logic.
-   `rag_engine.py`: Core RAG logic, routing, and SQL execution.
-   `data_layer.py`: Persistence and user session management.
-   `llm_factory.py`: Configuration for Ollama and Embedding models.
-   `storage/`: Persistent data for ChromaDB and DuckDB.
