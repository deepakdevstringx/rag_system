from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db"

client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
collection = client.get_collection(name="versioned_pdf_rag")

def print_indexed_documents():
    """List unique source files stored in the active versioned collection."""
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data.get("metadatas", [])
    chunk_count = sum(1 for meta in metadatas if meta.get("type") != "system_index")
    
    if not metadatas:
        print("❌ Database is empty!")
        return

    # Extract unique source filenames
    unique_files = sorted(
        {meta["source"] for meta in metadatas if "source" in meta and meta.get("type") != "system_index"}
    )
    
    print("\n==========================================")
    print(f"📊 Total Chunks in DB: {chunk_count}")
    print(f"📄 Total Unique PDF Files in DB: {len(unique_files)}")
    print("==========================================")
    for idx, filename in enumerate(unique_files, start=1):
        print(f"  {idx}. {filename}")
    print("==========================================\n")

if __name__ == "__main__":
    print_indexed_documents()