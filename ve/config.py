import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UNPROCESSED_PATH = os.path.join(BASE_DIR, "unprocessed_pdfs")
DOCS_PATH = os.path.join(BASE_DIR, "versioned_pdfs")
DB_PATH = os.path.join(BASE_DIR, "versioned_db")
REGISTRY_FILE = os.path.join(BASE_DIR, "versioned_document_registry.json")

# Chunking Configuration
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Model Configuration
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

# Ensure required directories exist
os.makedirs(UNPROCESSED_PATH, exist_ok=True)
os.makedirs(DOCS_PATH, exist_ok=True)