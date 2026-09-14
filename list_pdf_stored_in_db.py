#---- Global configuration ----#
from pypdf import PdfReader

import chromadb

Docs_Path = "./pdfs"
Db_Path = "./db"

#---- Vector Database setup ----#
chrome_client = chromadb.PersistentClient(path=Db_Path)

#Initialize or connect to collection using cosine similarity metric
collection = chrome_client.get_or_create_collection(
    name="smart_pdf_rag",
    metadata={"hnsw:space": "cosine"} # Metadata for the collection, specifying the similarity metric to use
)

def print_database_file_list():
    """Fetches all metadata entries directly from ChromaDB and lists unique source files."""
    # Retrieve metadata for all items stored in the collection
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data.get("metadatas", [])
    
    if not metadatas:
        print("❌ Database is empty!")
        return

    # Extract unique source filenames
    unique_files = sorted(set(meta["source"] for meta in metadatas if "source" in meta))
    
    print("\n==========================================")
    print(f"📊 Total Chunks in DB: {collection.count()}")
    print(f"📄 Total Unique PDF Files in DB: {len(unique_files)}")
    print("==========================================")
    for idx, filename in enumerate(unique_files, start=1):
        print(f"  {idx}. {filename}")
    print("==========================================\n")

# Run this check
print_database_file_list()