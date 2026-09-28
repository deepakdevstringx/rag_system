from functools import lru_cache
import glob
import shutil
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

from .config import ARCHIVE_DIR, EMBEDDING_MODEL, INBOX_DIR, VECTOR_DB_DIR
from .document_registry import load_registry, register_document_version, save_registry
from .text_splitter import split_text_into_chunks

chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
collection = chroma_client.get_or_create_collection(
    name="versioned_pdf_rag",
    metadata={"hnsw:space": "cosine"}
)

@lru_cache(maxsize=128)
def get_cached_embedding(text):
    """Return a cached Ollama embedding for a repeated search string."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=text, keep_alive="24h")
    return tuple(response["embeddings"][0])

def refresh_document_catalog(registry):
    """Embed and upsert the human-readable document catalog for inventory queries."""
    catalog_lines = [
        "ENTERPRISE POLICY CATALOG INDEX:",
        "List of all stored files, policies, and active versions in the database:\n"
    ]
    for filename, info in registry.items():
        catalog_lines.append(f"- Policy File: '{filename}' (Active Version: v{info['latest_version']})")

    catalog_doc = "\n".join(catalog_lines)
    embed_res = ollama.embed(model=EMBEDDING_MODEL, input=catalog_doc, keep_alive="24h")
    
    collection.upsert(
        ids=["SYSTEM_DOCUMENT_CATALOG_INDEX"],
        embeddings=[embed_res["embeddings"][0]],
        documents=[catalog_doc],
        metadatas=[{"source": "SYSTEM_CATALOG", "is_active": True, "type": "system_index"}]
    )
    print("✅ Vector System Catalog Index refreshed.")

def ingest_pending_pdfs():
    """Extract, version, embed, archive, and catalog PDFs waiting in the inbox."""
    files_to_process = glob.glob(str(INBOX_DIR / "*.pdf"))
    if not files_to_process:
        print(f"📁 No new files found in '{INBOX_DIR}'.")
        return

    registry = load_registry()
    
    for file_path in files_to_process:
        source_name = Path(file_path).name
        reader = PdfReader(file_path)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

        # Deactivate old version chunks if updating
        if source_name in registry:
            old_version = registry[source_name]["latest_version"]
            old_chunks = collection.get(
                where={
                    "$and": [
                        {"source": {"$eq": source_name}},
                        {"version": {"$eq": old_version}},
                    ]
                }
            )
            for cid in old_chunks["ids"]:
                collection.update(ids=[cid], metadatas=[{"is_active": False}])
            print(f"🔄 Updating '{source_name}' (Deactivating v{old_version} chunks)")
        
        current_version = register_document_version(registry, source_name, full_text)
        print(f"🆕 Indexing '{source_name}' as Version {current_version}")

        # Extract and Batch Chunk
        chunks_batch, metadatas_batch, ids_batch = [], [], []
        chunk_counter = 0

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if not page_text:
                continue
            for chunk in split_text_into_chunks(page_text):
                chunks_batch.append(chunk)
                metadatas_batch.append({
                    "source": source_name,
                    "page": page_num,
                    "version": current_version,
                    "is_active": True,
                    "type": "document_chunk"
                })
                ids_batch.append(f"{source_name}_v{current_version}_c{chunk_counter}")
                chunk_counter += 1

        if chunks_batch:
            embed_res = ollama.embed(model=EMBEDDING_MODEL, input=chunks_batch, keep_alive="24h")
            collection.add(
                ids=ids_batch,
                embeddings=embed_res["embeddings"],
                documents=chunks_batch,
                metadatas=metadatas_batch
            )

        # Move file out of unprocessed folder
        target_path = ARCHIVE_DIR / f"v{current_version}_{source_name}"
        shutil.move(file_path, str(target_path))

    save_registry(registry)
    refresh_document_catalog(registry)

def retrieve_context(query, k=5, filter_source=None):
    """Retrieve active document chunks or the catalog, with citations for answers."""
    query_embedding = list(get_cached_embedding(query))

    # Fast path for catalog/document list queries
    if any(w in query.lower() for w in ["list", "files", "policies", "all pdfs", "documents"]):
        results = collection.get(ids=["SYSTEM_DOCUMENT_CATALOG_INDEX"])
        if results["documents"]:
            return results["documents"][0], {"[SYSTEM_CATALOG]"}

    # Retrieve ACTIVE version chunks
    where_filter = {"is_active": True}
    if filter_source:
        where_filter = {
            "$and": [
                {"is_active": {"$eq": True}},
                {"source": {"$eq": filter_source}},
            ]
        }

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter,
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