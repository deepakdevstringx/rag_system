from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INBOX_DIR = DATA_DIR / "inbox"
ARCHIVE_DIR = DATA_DIR / "archive"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
REGISTRY_PATH = DATA_DIR / "document_registry.json"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

INBOX_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)