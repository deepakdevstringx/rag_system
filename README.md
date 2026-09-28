# Local PDF RAG

A local PDF question-answering application using ChromaDB for vector storage and Ollama for embeddings and responses.

## Layout

```text
project/
├── data/
│   ├── archive/                 # Processed, versioned PDF files
│   ├── inbox/                   # PDFs waiting for indexing
│   ├── legacy/                  # Preserved data from the previous application
│   ├── vector_db/               # Active ChromaDB store
│   └── document_registry.json   # Document version history
├── scripts/
│   ├── check_ollama.py          # Check local Ollama connectivity
│   └── list_documents.py        # List indexed source documents
├── src/pdf_rag/
│   ├── chat.py                  # Interactive chat and CLI entry point
│   ├── config.py                # Models and application paths
│   ├── document_registry.py     # Version metadata and text diffs
│   ├── text_splitter.py         # PDF text chunking
│   └── vector_store.py          # PDF ingestion and retrieval
├── tests/                       # Unit tests
└── pyproject.toml               # Package metadata and dependencies
```

Application code is in `src/pdf_rag`; PDFs, ChromaDB files, and registry state are in `data`. The project virtual environment is `.venv` and is not part of the application source tree.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ollama pull nomic-embed-text
ollama pull llama3.2
```

Ollama must be running locally. If it is not running as a service, start it in another terminal with `ollama serve`.

## Run

Put new PDFs in `data/inbox/`, then launch from the project root:

```bash
python -m pdf_rag
```

The app indexes pending PDFs, moves processed files into `data/archive/`, and stores embeddings in `data/vector_db/`. Enter `exit` to end the chat.

List indexed documents and their chunk counts:

```bash
python scripts/list_documents.py
```

Check the Ollama chat model:

```bash
python scripts/check_ollama.py
```

Run the unit tests with the standard library:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Data Safety

The previous unversioned PDFs and ChromaDB data are preserved under `data/legacy/`. The active PDFs, registry, and vector index are also retained as local data. Removing a vector database deletes its index and may require re-indexing the PDFs.
