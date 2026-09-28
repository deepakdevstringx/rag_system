# Local PDF RAG Chatbot

This project reads PDF files, creates vector embeddings locally, stores them in ChromaDB, and answers questions using Ollama.

## Models Used

- `nomic-embed-text`: creates embeddings for PDF chunks and user queries.
- `llama3.2`: improves the search query and generates the final answer.

## Project Files

- `read_pdf_v2.py`: main PDF indexing and chat application.
- `read_pdf.py`: earlier, slower version that embeds one chunk at a time.
- `list_pdf_stored_in_db.py`: lists PDFs and chunk counts stored in ChromaDB.
- `test_ollama.py`: basic Ollama connectivity test.
- `pdfs/`: place PDF files here.
- `db/`: persistent ChromaDB data. Do not delete it unless you want to rebuild the index.

## First-Time Setup

Run these commands from the project directory:

```bash
cd /home/deepak/Devstringx/learning
python3 -m venv .venv
source .venv/bin/activate
pip install pypdf chromadb ollama
```

For later sessions, only activation is needed:

```bash
cd /home/deepak/Devstringx/learning
source .venv/bin/activate
```

## Install and Prepare Ollama

Check that Ollama is installed:

```bash
ollama --version
```

Download the required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Check installed models:

```bash
ollama list
```

## Start the Ollama Server

Open a separate terminal and run:

```bash
ollama serve
```

Leave this terminal running. If Ollama is already running as a system service, do not start a second server.

The local API normally uses:

```text
http://localhost:11434
```

## Add PDFs

Copy one or more PDF files into:

```text
/home/deepak/Devstringx/learning/pdfs/
```

The current application reads PDF files directly from `./pdfs`.

## Start the Chatbot

With the virtual environment active and Ollama running:

```bash
cd /home/deepak/Devstringx/learning
source .venv/bin/activate
python read_pdf_v2.py
```

On the first run, the application will:

1. Extract text from each PDF page.
2. Split the text into overlapping chunks.
3. Create embeddings with `nomic-embed-text`.
4. Store the vectors and page metadata in ChromaDB.
5. Start the interactive chat.

Ask questions at the prompt. Type the following to stop:

```text
exit
```

## Run It Again

After the first indexing run, the application detects the existing ChromaDB collection and reuses it:

```bash
python read_pdf_v2.py
```

New PDFs are not automatically added when the collection already contains chunks. To rebuild the index after adding or replacing PDFs, stop the application and remove the local database first:

```bash
rm -rf db
python read_pdf_v2.py
```

Only run that command when you intentionally want to rebuild the index.

## Inspect Stored PDFs

To see the PDFs and chunk count currently stored in ChromaDB:

```bash
python list_pdf_stored_in_db.py
```

To see the number of stored vectors directly:

```bash
python -c "import chromadb; c=chromadb.PersistentClient(path='./db').get_or_create_collection(name='smart_pdf_rag'); print(c.count())"
```

## Test Ollama

Run the basic Ollama test:

```bash
python test_ollama.py
```

Test the embedding model manually:

```bash
ollama run nomic-embed-text
```

Test the chat model manually:

```bash
ollama run llama3.2
```

## Useful Ollama Commands

```bash
ollama ps                 # Show running models
ollama show llama3.2      # Show model details
ollama show nomic-embed-text
ollama rm <model-name>    # Remove a model
ollama help               # Show command help
```

## Troubleshooting

### `Connection refused` or Ollama connection error

Make sure Ollama is running:

```bash
ollama serve
```

### Model not found

Download both required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### No PDF files found

Confirm that files have a `.pdf` extension and are inside the `pdfs/` directory:

```bash
ls -lh pdfs
```

### Answers do not include a newly added PDF

The existing ChromaDB index is reused. Rebuild it after adding documents:

```bash
rm -rf db
python read_pdf_v2.py
```

### Check Python packages

```bash
pip list | grep -E 'chromadb|ollama|pypdf'
```

## Deactivate the Virtual Environment

```bash
deactivate
```
