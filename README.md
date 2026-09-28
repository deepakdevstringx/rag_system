# Local PDF RAG

A local PDF question-answering application using ChromaDB for vector storage, Ollama for embeddings, and selectable Ollama, Gemini, or Claude chat models.

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
python -m pip install -r requirements.txt
python -m pip install -e .
ollama pull nomic-embed-text
ollama pull llama3.2
```

Copy `.env.example` to `.env` to set the default chat provider and model. The default is Ollama. Keep API keys in `.env`; it is ignored by Git.

To use Gemini by default, set:

```dotenv
PDF_RAG_CHAT_PROVIDER=gemini
PDF_RAG_CHAT_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=your-key
ANTHROPIC_API_KEY=
```

To use Claude, set `PDF_RAG_CHAT_PROVIDER=claude`, choose a Claude model such as `claude-sonnet-4-5`, and set `ANTHROPIC_API_KEY` in `.env`. Gemini and Claude keys are independent; keep whichever provider keys you use in the ignored `.env` file.

Ollama must be running locally when selected. If it is not running as a service, start it in another terminal with `ollama serve`. Gemini requires internet access and a valid Google AI API key.

The chat model can be changed at runtime with `/provider ollama`, `/provider gemini`, `/provider claude`, or `/model MODEL_NAME`. Use `/provider` to show the active choice. Gemini requires `GOOGLE_API_KEY`; Claude requires `ANTHROPIC_API_KEY`. When Claude fails, requests fall back to Ollama `llama3.2` by default; configure this with `PDF_RAG_FALLBACK_PROVIDER` and `PDF_RAG_FALLBACK_MODEL`. The embedding model remains `nomic-embed-text` on Ollama so it matches the existing ChromaDB vectors; changing embedding models requires rebuilding the vector index.

## Run

Put new PDFs in `data/inbox/`, then launch from the project root:

```bash
python -m pdf_rag
```

The app indexes pending PDFs, moves processed files into `data/archive/`, and stores embeddings in `data/vector_db/`. Enter `exit` to end the chat.

To compare a document's revisions, ask for its differences or changes, for example `What changed between versions of Hackathon 2026?`. If no specific version pair is given, the assistant compares each adjacent revision (v1 to v2, then v2 to v3, and so on). You can request a specific pair with wording such as `Compare v1 and v3 of Hackathon 2026`. The comparison uses saved extracted text and summarizes only additions and removals; documents need to have been ingested in more than one version.

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
