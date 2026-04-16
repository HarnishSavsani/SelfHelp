# 🚀 Genius AI - Setup & Installation Guide

This guide provides step-by-step instructions to get the Genius AI RAG application running on your machine using **`uv`** for fast, reliable dependency management.

---

## 💻 Prerequisites

1.  **Ollama**: Ensure Ollama is installed and running (`qwen2.5:7b` model recommended).
2.  **UV**: The lightning-fast Python package installer and manager.

---

## 🪟 Windows Setup (PowerShell)

### 1. Set Execution Policy
Windows restricts script execution by default. To allow the virtual environment activation script to run, open PowerShell as **Administrator** and run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install `uv` (if not installed)
```powershell
powershell -c "ir | iex"  # Official standalone installer
# OR via pip if you have Python:
pip install uv
```

### 3. Create and Activate Virtual Environment
Navigate to the project directory:
```powershell
uv venv
.venv\Scripts\activate
```

### 4. Install Dependencies
```powershell
uv pip install -r requirements.txt
```

---

## 🍏 Mac / 🐧 Linux Setup (Terminal)

### 1. Install `uv` (if not installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# OR via pip:
pip install uv
```

### 2. Create and Activate Virtual Environment
```bash
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
```

---

## ⚙️ Configuration & Running

### 1. Environment Variables
Copy the example environment file and fill in your keys (if using Azure) or use the defaults for Ollama:
```bash
cp .env.example .env
```

### 2. Run the Application
Start the Chainlit interface:
```bash
chainlit run app.py
```

---

## 🛠️ Helper Tools

### Generate Folder Structure
To generate a clean `.txt` file of your project structure for documentation or sharing:
```bash
python helper_scripts/folder_structure.py
```
*Generated file: `folder_structure.txt`*

### Verification
Once running, navigate to `http://localhost:8000` in your browser. You can login using the default credentials found in your `.env` file.
