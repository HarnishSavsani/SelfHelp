import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure we're in the project root to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load config to get LOCAL_MODELS_DIR and set HF_HOME
import config

def download_bge():
    print(f"\n[1/2] Downloading BGE Embedding Model ({config.EMBED_MODEL_NAME})...")
    try:
        from huggingface_hub import snapshot_download
        model_name = config.EMBED_MODEL_NAME
        # Download the model files
        snapshot_download(repo_id=model_name)
        print("✅ BGE model downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download BGE model: {e}")

def download_docling_models():
    print("\n[2/2] Downloading Docling Pipeline Models (Layout, TableFormer, OCR)...")
    try:
        from docling.utils.model_downloader import download_models
        download_models()
        print("✅ Docling models downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download Docling models: {e}")

if __name__ == "__main__":
    print("="*70)
    print("🤖 Genius AI — Offline Model Pre-fetcher")
    print("="*70)
    print(f"Target Directory: {config.LOCAL_MODELS_DIR}")
    print("This will download all necessary AI models for offline/air-gapped use.")
    print("Note: This REQUIRES an active internet connection.")
    print("...")
    
    # Remove OFFLINE_MODE restrictions for this script specifically
    os.environ.pop("HF_HUB_OFFLINE", None)
    
    download_bge()
    download_docling_models()
    
    print("\n" + "="*70)
    print("🎉 ALL DONE!")
    print(f"Models are now cached in '{config.LOCAL_MODELS_DIR}'.")
    print("You can now safely copy this project to your offline environment.")
    print("Set OFFLINE_MODE=true in your .env file to enable air-gapped support.")
    print("="*70)
