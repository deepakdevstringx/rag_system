

#---- Global configuration ----#
import glob
import os
import re

import ollama
from pypdf import PdfReader

import chromadb


Docs_Path = "./pdfs"
Db_Path = "./db"
Chunk_Size = 600
Chunk_Overlap = 100


#---- Vector Database setup ----#
chrome_client = chromadb.PersistentClient(path=Db_Path)

#Initialize or connect to collection using cosine similarity metric
collection = chrome_client.get_or_create_collection(
    name="smart_pdf_rag",
    metadata={"hnsw:space": "cosine"} # Metadata for the collection, specifying the similarity metric to use
)

def recursive_sentence_splitter(text, chunk_size=Chunk_Size, overlap = Chunk_Overlap):
    """Splits text cleanly on paragraphs and sentences boundaries."""
    # Split the text by double newlines (paragraphs) or sentence delimiters (.)
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

def extract_and_index_pdfs():
    """Extracts text from page, generate vectors, and indexes then with page metadata. """
    pdf_files = glob.glob(os.path.join(Docs_Path, "*.pdf"))
    if not pdf_files:
        print("No Pdf files found in the specified directory.")
        return

    print("indexing documents with page metadata...")
    doc_id_counter = 0

    for pdf_file in pdf_files:
        #Extract text from the pdf file
        fileName = os.path.basename(pdf_file)
        reader = PdfReader(pdf_file)

        # Track page numbers explicitly to enable source citations
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if not page_text:
                continue  # Skip pages with no text

            page_chunks = recursive_sentence_splitter(page_text)

            for chunk in page_chunks:
                #Generate a 758-dimensional vector via Ollama
                response = ollama.embed(model="nomic-embed-text", input=chunk)
                print(response)
                embedding = response["embeddings"][0]

                # Insert document, embedding vector, and citation metadata into the DB
                collection.add(
                    ids=[f"chunk_{doc_id_counter}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source": fileName,
                        "page": page_num
                    }]
                )
                doc_id_counter += 1
    print(f"Indexed {doc_id_counter} chunks from {len(pdf_files)} PDF files.")


def retrieve__context(query, k=3):
    """Embeds the query and fetches the top k matching text chunks with citation."""
    query_response = ollama.embed(model="nomic-embed-text", input=query)
    query_embedding = query_response["embeddings"][0]

    # Retrieve the top k most similar chunks from the collection
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
        formatted_contexts.append(f"SOURCE: {source_tag}:\n{doc}")
        citations.append(source_tag)

    return "\n\n--\n\n".join(formatted_contexts), set(citations)

def start_chat_session():
    chat_history = []
    print("Smart Rag Assistant Ready! Type 'exit' to quit.\n")

    while True:
        user_query = input("\n Ask a question: ").strip()
        if user_query.lower() == 'exit':
            break
        if not user_query:
            continue

        context, citations = retrieve__context(user_query, k=3)

        system_instruction = f"""You are a precise enterprise research assistant.
            Answer the user's question using ONLY the provided document context below. 

            INSTRUCTIONS:
            1. Base your answer strictly on the provided context. Do not invent information.
            2. If the context does not contain enough info, state clearly that you cannot find it.

            DOCUMENT CONTEXT:
            {context}
            """

        messages = [{"role":"system", "content": system_instruction}
                    ] + chat_history[-4:] + [{"role":"user", "content": user_query}]

        print("\n🤔 Searching and generating response...\n")
        
        response_stream = ollama.chat(
            model="llama3.2",
            messages=messages,
            stream=True
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
        chat_history.append({"role":"user", "content": user_query})
        chat_history.append({"role":"assistant", "content": full_response})

if __name__ == "__main__":
    if collection.count() == 0:
        extract_and_index_pdfs()
    else:
        print(f"📂 Connected to ChromaDB containing {collection.count()} chunks.\n")
        
    start_chat_session()






        