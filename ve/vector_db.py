import os
import glob
import shutil
from functools import lru_cache
import chromadb
import ollama
from pypdf import PdfReader

from config import DB_PATH, UNPROCESSED_PATH, DOCS_PATH, EMBED_MODEL
from splitter import recursive_sentence_splitter
from registry import load_registry, save_registry, update_document_version

# Initialize Persistent Vector Database
chrome_client = chromadb.PersistentClient(path=DB_PATH)
collection = chrome_client.get_or_create_collection(
    name="versioned_pdf_rag",
    metadata={"hnsw:space": "cosine"}
)

@lru_cache(maxsize=128)
def get_cached_embedding(text):
    response = ollama.embed(model=EMBED_MODEL, input=text, keep_alive="24h")
    return tuple(response["embeddings"][0])

def update_system_index_in_vector_db(registry):
    """Embeds a human-readable document catalog directly into vector space."""
    catalog_lines = [
        "ENTERPRISE POLICY CATALOG INDEX:",
        "List of all stored files, policies, and active versions in the database:\n"
    ]
    for filename, info in registry.items():
        catalog_lines.append(f"- Policy File: '{filename}' (Active Version: v{info['latest_version']})")

    catalog_doc = "\n".join(catalog_lines)
    embed_res = ollama.embed(model=EMBED_MODEL, input=catalog_doc, keep_alive="24h")
    
    collection.upsert(
        ids=["SYSTEM_DOCUMENT_CATALOG_INDEX"],
        embeddings=[embed_res["embeddings"][0]],
        documents=[catalog_doc],
        metadatas=[{"source": "SYSTEM_CATALOG", "is_active": True, "type": "system_index"}]
    )
    print("✅ Vector System Catalog Index refreshed.")

def process_unprocessed_folder():
    """Ingests unprocessed PDFs, handles versioning, updates embeddings & catalog."""
    files_to_process = glob.glob(os.path.join(UNPROCESSED_PATH, "*.pdf"))
    if not files_to_process:
        print(f"📁 No new files found in '{UNPROCESSED_PATH}'.")
        return

    registry = load_registry()
    
    for file_path in files_to_process:
        base_name = os.path.basename(file_path)
        reader = PdfReader(file_path)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

        # Deactivate old version chunks if updating
        if base_name in registry:
            old_version = registry[base_name]["latest_version"]
            old_chunks = collection.get(
                where={
                    "$and": [
                        {"source": {"$eq": base_name}},
                        {"version": {"$eq": old_version}},
                    ]
                }
            )
            for cid in old_chunks["ids"]:
                collection.update(ids=[cid], metadatas=[{"is_active": False}])
            print(f"🔄 Updating '{base_name}' (Deactivating v{old_version} chunks)")
        
        current_version = update_document_version(registry, base_name, full_text)
        print(f"🆕 Indexing '{base_name}' as Version {current_version}")

        # Extract and Batch Chunk
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
            embed_res = ollama.embed(model=EMBED_MODEL, input=chunks_batch, keep_alive="24h")
            collection.add(
                ids=ids_batch,
                embeddings=embed_res["embeddings"],
                documents=chunks_batch,
                metadatas=metadatas_batch
            )

        # Move file out of unprocessed folder
        target_path = os.path.join(DOCS_PATH, f"v{current_version}_{base_name}")
        shutil.move(file_path, target_path)

    save_registry(registry)
    update_system_index_in_vector_db(registry)

def retrieve_context(query, k=5):
    """Retrieves document context or routes system inventory queries."""
    query_embedding = list(get_cached_embedding(query))

    # Fast path for catalog/document list queries
    if any(w in query.lower() for w in ["list", "files", "policies", "all pdfs", "documents"]):
        results = collection.get(ids=["SYSTEM_DOCUMENT_CATALOG_INDEX"])
        if results["documents"]:
            return results["documents"][0], {"[SYSTEM_CATALOG]"}

    # Retrieve ACTIVE version chunks
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