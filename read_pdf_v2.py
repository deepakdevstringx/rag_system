#---- Global configuration ----#
from functools import lru_cache
import glob
import os
import re

import chromadb
import ollama
from pypdf import PdfReader

Docs_Path = "./pdfs"
Db_Path = "./db"
Chunk_Size = 600
Chunk_Overlap = 100


#---- Vector Database setup ----#
chrome_client = chromadb.PersistentClient(path=Db_Path)

collection = chrome_client.get_or_create_collection(
    name="smart_pdf_rag",
    metadata={"hnsw:space": "cosine"}
)

def recursive_sentence_splitter(text, chunk_size=Chunk_Size, overlap=Chunk_Overlap):
    """Splits text cleanly on paragraphs and sentence boundaries."""
    paragraphs = re.split(r'(\n\n|\.\s+)', text)
    chunks = []
    current_chunk = ""

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

# --- OPTIMIZATION 1: Cached Query Embedding ---
@lru_cache(maxsize=128)
def get_cached_embedding(query_text):
    """Caches query embeddings to skip Ollama calls for repeated inputs."""
    response = ollama.embed(
        model="nomic-embed-text", 
        input=query_text,
        keep_alive="24h"  # Keep model loaded for a long session using a valid duration format
    )
    return tuple(response["embeddings"][0])


def extract_and_index_pdfs():
    """Extracts text from pages, generates batch vectors, and indexes them."""
    pdf_files = glob.glob(os.path.join(Docs_Path, "*.pdf"))
    if not pdf_files:
        print("No PDF files found in the specified directory.")
        return

    print("Indexing documents with page metadata...")
    doc_id_counter = 0

    for pdf_file in pdf_files:
        fileName = os.path.basename(pdf_file)
        reader = PdfReader(pdf_file)

        chunks_batch = []
        metadatas_batch = []
        ids_batch = []

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if not page_text:
                continue

            page_chunks = recursive_sentence_splitter(page_text)

            for chunk in page_chunks:
                chunks_batch.append(chunk)
                metadatas_batch.append({"source": fileName, "page": page_num})
                ids_batch.append(f"chunk_{doc_id_counter}")
                doc_id_counter += 1

        # --- OPTIMIZATION 2: Batch Vector Generation & Insertion ---
        # Generate embeddings for the entire PDF in 1 call instead of chunk-by-chunk
        if chunks_batch:
            response = ollama.embed(
                model="nomic-embed-text", 
                input=chunks_batch,
                keep_alive="24h"
            )
            embeddings = response["embeddings"]

            collection.add(
                ids=ids_batch,
                embeddings=embeddings,
                documents=chunks_batch,
                metadatas=metadatas_batch
            )

    print(f"Indexed {doc_id_counter} chunks from {len(pdf_files)} PDF files.")


def retrieve_context(query, k=3):
    """Fetches top k matching text chunks using cached embeddings."""
    query_embedding = list(get_cached_embedding(query))

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]

    formatted_contexts = []
    citations = []
    for doc, meta in zip(docs, metadatas):
        source_tag = f"[{meta['source']} - Page {meta['page']}]"
        formatted_contexts.append(f"SOURCE {source_tag}:\n{doc}")
        citations.append(source_tag)

    return "\n\n--\n\n".join(formatted_contexts), set(citations)


def start_chat_session():
    chat_history = []
    print("Smart RAG Assistant Ready! Type 'exit' to quit.\n")

    while True:
        user_query = input("\n Ask a question: ").strip()

        if user_query.lower() == 'exit':
            break
        if not user_query:
            continue

        optimized_query = optimize_query(user_query)

        print(f"\n🔎 Optimized query: {optimized_query}")

        context, citations = retrieve_context(optimized_query, k=5)
        # context, citations = retrieve_context(user_query, k=3)

        system_instruction = f"""You are a precise enterprise research assistant.
            Answer the user's question using ONLY the provided document context below. 

            INSTRUCTIONS:
            1. Base your answer strictly on the provided context. Do not invent information.
            2. If the context does not contain enough info, state clearly that you cannot find it.

            DOCUMENT CONTEXT:
            {context}
            """

        messages = [{"role": "system", "content": system_instruction}] + chat_history[-4:] + [{"role": "user", "content": user_query}]

        print("\n🤔 Searching and generating response...\n")
        
        # --- OPTIMIZATION 3: High-speed streaming + model keep alive ---
        response_stream = ollama.chat(
            model="llama3.2",
            messages=messages,
            stream=True,
            keep_alive="24h",  # Keeps LLM pre-loaded in VRAM with valid duration format
            options={
                "num_predict": 512,  # Limit max response tokens to avoid long lingering generations
                "temperature": 0.1   # Lower temperature speeds up token picking
            }
        )

        full_response = ""
        print("🤖 ANSWER:")
        for chunk in response_stream:
            content = chunk["message"]["content"]
            full_response += content
            print(content, end="", flush=True)

        print("\n\n📄 Sources consulted:")
        for citation in citations:
            print(f" . {citation}")
        print("-" * 50)
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": full_response})

        
def optimize_query(user_query):
    prompt = f"""
    Rewrite the user's question for semantic search against an enterprise
    PDF knowledge base.

    Rules:
    1. Fix spelling mistakes.
    2. Fix grammar.
    3. Preserve the user's original intent.
    4. Do not answer the question.
    5. Do not invent facts.
    7. Add relevant synonyms when useful.
    9. Keep it concise.
    10. Return ONLY the optimized search query.

    User question:
    {user_query}
    """

    #   Rules:
    #     1. Fix spelling mistakes.
    #     2. Fix grammar.
    #     3. Preserve the user's original intent.
    #     4. Do not answer the question.
    #     5. Do not invent facts.
    #     6. Expand important abbreviations only when reasonably obvious.
    #     7. Add relevant synonyms when useful.
    #     8. Make the query specific enough for vector search.
    #     9. Keep it concise.
    #     10. Return ONLY the optimized search query.

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": "You optimize user queries for semantic search."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.0,
            "num_predict": 100
        }
    )

    return response["message"]["content"].strip()


if __name__ == "__main__":
    if collection.count() == 0:
        extract_and_index_pdfs()
    else:
        print(f"📂 Connected to ChromaDB containing {collection.count()} chunks.\n")
        
    start_chat_session()