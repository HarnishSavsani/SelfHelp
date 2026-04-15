# Genius AI — Config-Driven RAG Framework

A production-ready, open-source RAG (Retrieval-Augmented Generation) framework built with **Chainlit** and **LlamaIndex**. Switch between domain-specific applications (Insurance, Medical, Customer Support, etc.) by changing a single config line.

## ✨ Key Features

- **Dual RAG Pipeline** — Structured data (Excel/CSV/JSON → DuckDB SQL) + Unstructured docs (PDF/MD/TXT → ChromaDB vectors)
- **Config-Driven Profiles** — One YAML file per application. Change `APP_PROFILE=insurance_claims` in `.env` and restart.
- **Built-in Guardrails** — Domain-specific topic filtering (no LLM call, pure keyword matching for speed)
- **Smart SQL Synthesis** — Natural language answers from data queries, never raw SQL or technical jargon
- **Persistent Sessions** — Upload files once, query across chat sessions

---

## 🚀 Quick Start with `uv`

We recommend [uv](https://github.com/astral-sh/uv) for fast environment management.

### 1. Install `uv` (if not present)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Homebrew:**
```bash
brew install uv
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Setup Environment

```bash
# Clone the repo
git clone <your-repo-url>
cd chainlit

# Create venv and install all dependencies
uv sync

# OR using requirements.txt
uv venv && uv pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` to set your model and profile:
```env
OLLAMA_MODEL=qwen2.5:7b
APP_PROFILE=default          # or: insurance_claims, customer_support
```

### 4. Run

```bash
source .venv/bin/activate
chainlit run app.py
```

---

## 🛠 Manual Setup (without `uv`)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Edit with your settings
chainlit run app.py
```

---

## 🎭 Application Profiles

Profiles let you turn this generic RAG into a **domain-specific assistant** by changing one line in `.env`.

### How It Works

```
profiles/
├── default.yaml              # Generic RAG — no guardrails
├── insurance_claims.yaml     # Insurance claims analyst
└── customer_support.yaml     # Support ticket analyzer
```

Each profile controls:
| Feature | Example |
|---|---|
| **App Name** | "ClaimAssist AI", "SupportIQ" |
| **System Prompt** | Domain-specific personality and rules |
| **Guardrails** | Blocked/allowed topics, strictness level |
| **Welcome Message** | Custom onboarding for the domain |
| **Data Labels** | "your claim data" vs "your support tickets" |
| **File Restrictions** | Which file types are accepted |

### Switching Profiles

```bash
# In .env — just change this line:
APP_PROFILE=insurance_claims
```

Then restart:
```bash
chainlit run app.py
```

### Creating Your Own Profile

1. Copy `profiles/default.yaml` to `profiles/my_domain.yaml`
2. Edit the YAML to define your domain's personality, topics, and guardrails
3. Set `APP_PROFILE=my_domain` in `.env`
4. Restart the app

### Guardrail Strictness Levels

| Level | Behavior |
|---|---|
| `relaxed` | No filtering — all queries pass (default profile) |
| `moderate` | Blocks explicitly forbidden topics, allows everything else |
| `strict` | Only allows queries matching defined allowed topics |

---

## 📂 Project Structure

```
├── app.py                  # Chainlit UI, auth, message handling
├── rag_engine.py           # Core RAG logic, SQL/vector routing
├── app_profile.py          # Profile loader and guardrail engine
├── llm_factory.py          # LLM/embedding model configuration
├── data_layer.py           # SQLite persistence and auth
├── config.py               # Central config (env vars, paths)
├── profiles/               # Domain-specific YAML profiles
│   ├── default.yaml
│   ├── insurance_claims.yaml
│   └── customer_support.yaml
├── storage/                # Persistent data (auto-created)
├── requirements.txt        # Pip dependencies
└── pyproject.toml          # UV/project metadata
```

---

## 📦 Requirements

- **Python** ≥ 3.12
- **Ollama** running locally (or Azure endpoint configured)
- Recommended model: `qwen2.5:7b` (pull via `ollama pull qwen2.5:7b`)
