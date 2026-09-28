from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
INBOX_DIR = DATA_DIR / "inbox"
ARCHIVE_DIR = DATA_DIR / "archive"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
REGISTRY_PATH = DATA_DIR / "document_registry.json"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "nomic-embed-text"
CHAT_PROVIDER = os.getenv("PDF_RAG_CHAT_PROVIDER", "ollama").strip().lower()
CHAT_MODEL = os.getenv(
    "PDF_RAG_CHAT_MODEL",
    {
        "gemini": "gemini-2.5-flash",
        "claude": "claude-sonnet-4-5",
    }.get(CHAT_PROVIDER, "llama3.2"),
).strip()
FALLBACK_PROVIDER = os.getenv("PDF_RAG_FALLBACK_PROVIDER", "ollama").strip().lower()
FALLBACK_MODEL = os.getenv("PDF_RAG_FALLBACK_MODEL", "llama3.2").strip()

INBOX_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)