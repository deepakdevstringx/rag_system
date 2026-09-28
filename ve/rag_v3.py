import os
import glob
import re
import shutil
import json
import difflib
from datetime import datetime
from functools import lru_cache

import chromadb
import ollama
from pypdf import PdfReader

# ---- Global Configuration (Isolated Paths for Version RAG) ---- #
UNPROCESSED_PATH = "./unprocessed_pdfs"
DOCS_PATH = "./versioned_pdfs"
DB_PATH = "./versioned_db"
REGISTRY_FILE = "./versioned_document_registry.json"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Ensure isolated directories exist
os.makedirs(UNPROCESSED_PATH, exist_ok=True)
os.makedirs(DOCS_PATH, exist_ok=True)

# ---- Isolated Vector Database Setup ---- #
chrome_client = chromadb.PersistentClient(path=DB_PATH)
collection = chrome_client.get_or_create_collection(
    name="versioned_pdf_rag",
    metadata={"hnsw:space": "cosine"}
)

# ---- Helper Functions ---- #
def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=4)

@lru_cache(maxsize=128)
def get_cached_embedding(text):
    response = ollama.embed(model="nomic-embed-text", input=text, keep_alive="24h")
    return tuple(response["embeddings"][0])

def recursive_sentence_splitter(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = re.split(r'(\n\n|\.\s+)', text)
    chunks, current_chunk = [], ""
    for item in paragraphs:
        if len(current_chunk) + len(item) <= chunk_size:
            current_chunk += item
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + item if len(current_chunk) >= overlap else item
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

# ---- System Index Generator ---- #
def update_system_index_in_vector_db(registry):
    """Embeds a human-readable document index directly into vector space."""
    catalog_lines = [
        "ENTERPRISE POLICY CATALOG INDEX:",
        "List of all stored files, policies, and active versions in the database:\n"
    ]
    for filename, info in registry.items():
        catalog_lines.append(f"- Policy File: '{filename}' (Active Version: v{info['latest_version']})")

    catalog_doc = "\n".join(catalog_lines)
    
    embed_res = ollama.embed(model="nomic-embed-text", input=catalog_doc, keep_alive="24h")
    
    collection.upsert(
        ids=["SYSTEM_DOCUMENT_CATALOG_INDEX"],
        embeddings=[embed_res["embeddings"][0]],
        documents=[catalog_doc],
        metadatas=[{"source": "SYSTEM_CATALOG", "is_active": True, "type": "system_index"}]
    )
    print("✅ Vector System Catalog Index refreshed in versioned DB.")

# ---- Core Processing & Ingestion Engine ---- #
def process_unprocessed_folder():
    """Scans ./unprocessed_pdfs, handles versioning, indexes vectors, and updates catalog."""
    files_to_process = glob.glob(os.path.join(UNPROCESSED_PATH, "*.pdf"))
    if not files_to_process:
        print(f"📁 No new files found in '{UNPROCESSED_PATH}'.")
        return

    registry = load_registry()
    
    for file_path in files_to_process:
        base_name = os.path.basename(file_path)
        reader = PdfReader(file_path)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

        # Version handling logic
        if base_name in registry:
            current_version = registry[base_name]["latest_version"] + 1
            old_version = registry[base_name]["latest_version"]
            
            # Soft-deactivate chunks belonging to the previous version
            old_chunks = collection.get(where={"source": base_name, "version": old_version})
            for cid in old_chunks["ids"]:
                collection.update(ids=[cid], metadatas=[{"is_active": False}])
                
            print(f"🔄 Updating '{base_name}' to Version {current_version} (Deactivated v{old_version})")
        else:
            current_version = 1
            print(f"🆕 Ingesting new document '{base_name}' as Version 1")

        # Chunking & Embedding
        chunks_batch, metadatas_batch, ids_batch = [], [], []
        chunk_counter = 0

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if not page_text:
                continue
            for chunk in recursive_sentence_splitter(page_text):
                chunks_batch.append(chunk)
                metadatas_batch.append({
                    "source": base_name,
                    "page": page_num,
                    "version": current_version,
                    "is_active": True,
                    "type": "document_chunk"
                })
                ids_batch.append(f"{base_name}_v{current_version}_c{chunk_counter}")
                chunk_counter += 1

        if chunks_batch:
            embed_res = ollama.embed(model="nomic-embed-text", input=chunks_batch, keep_alive="24h")
            collection.add(
                ids=ids_batch,
                embeddings=embed_res["embeddings"],
                documents=chunks_batch,
                metadatas=metadatas_batch
            )

        # Registry Update
        if base_name not in registry:
            registry[base_name] = {"latest_version": 1, "history": {}}
        
        registry[base_name]["latest_version"] = current_version
        registry[base_name]["history"][str(current_version)] = {
            "timestamp": datetime.now().isoformat(),
            "full_text": full_text
        }

        # Move processed PDF to dedicated storage
        target_path = os.path.join(DOCS_PATH, f"v{current_version}_{base_name}")
        shutil.move(file_path, target_path)

    save_registry(registry)
    update_system_index_in_vector_db(registry)

# ---- Diff & Retrieval Tools ---- #
def get_document_diff(filename, v1, v2):
    """Calculates diff between version v1 and v2 of a document."""
    registry = load_registry()
    if filename not in registry:
        return f"File '{filename}' not found."
    
    hist = registry[filename]["history"]
    if str(v1) not in hist or str(v2) not in hist:
        return f"Specified versions (v{v1}, v{v2}) do not exist for '{filename}'."

    t1 = hist[str(v1)]["full_text"].splitlines()
    t2 = hist[str(v2)]["full_text"].splitlines()

    diff = difflib.unified_diff(
        t1, t2, 
        fromfile=f"{filename} (v{v1})", 
        tofile=f"{filename} (v{v2})", 
        lineterm=""
    )
    return "\n".join(diff)

def retrieve_context(query, k=5):
    """Retrieves document context or routes system metadata queries directly."""
    query_embedding = list(get_cached_embedding(query))

    # Fast path for document inventory queries
    if any(w in query.lower() for w in ["list", "files", "policies", "all pdfs", "documents"]):
        results = collection.get(ids=["SYSTEM_DOCUMENT_CATALOG_INDEX"])
        if results["documents"]:
            return results["documents"][0], {"[SYSTEM_CATALOG]"}

    # Search only ACTIVE version chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"is_active": True}
    )
    
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]

    formatted_contexts = []
    citations = []
    for doc, meta in zip(docs, metadatas):
        source_tag = f"[{meta['source']} - v{meta.get('version', 1)} Page {meta.get('page', 'N/A')}]"
        formatted_contexts.append(f"SOURCE {source_tag}:\n{doc}")
        citations.append(source_tag)

    return "\n\n--\n\n".join(formatted_contexts), set(citations)

# ---- Execution Entrypoint ---- #
if __name__ == "__main__":
    # Check for pending uploads in unprocessed_pdfs
    process_unprocessed_folder()
    
    print(f"\n📂 Connected to Versioned ChromaDB at '{DB_PATH}'")
    print(f"Total Chunks Stored: {collection.count()}")