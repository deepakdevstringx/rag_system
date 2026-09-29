# `_answer_request` Walkthrough

This document follows one browser chat request through the Python API: how the request is routed, what gets sent to a model, where document evidence comes from, how the answer is checked, and how the result is returned to React.

## 1. Entry Point and Inputs

The function is defined in [`api.py`](../src/pdf_rag/api.py#L54-L55):

```python
def _answer_request(message, history):
```

- `message` is the latest question typed by the user.
- `history` is recent conversation supplied by the web client. The API filters it to user/assistant text and keeps at most the latest six messages before calling this function ([`api.py`](../src/pdf_rag/api.py#L192-L210)).
- The function returns a pair: `(answer_text, citations)`. `citations` is a list of source labels, or an empty list for answers that did not use retrieved documents.

`_answer_request` runs in FastAPI's thread pool because model calls, Chroma queries, and embedding calls are synchronous operations ([`api.py`](../src/pdf_rag/api.py#L203-L210)).

## 2. First Branch: Greeting

The first decision checks whether the message is a simple greeting ([`api.py`](../src/pdf_rag/api.py#L56-L69)). The greeting classifier recognizes a small fixed set such as `hi`, `hello`, and `good morning` ([`chat.py`](../src/pdf_rag/chat.py#L14-L24)).

When it is a greeting:

1. No PDF search or embedding is performed.
2. The function sends Claude/the currently selected chat provider a system instruction, the last six history messages, and the greeting.
3. It allows a short response (`max_tokens=100`, temperature `0.3`).
4. It returns `(answer, [])`, meaning there are no document citations.

## 3. Second Branch: Version Comparison

If the input is not a greeting, the function checks whether it sounds like a version-change request ([`api.py`](../src/pdf_rag/api.py#L71-L82)). The intent helper looks for phrases including `difference`, `changed`, `compare`, `old version`, and `latest version` ([`chat.py`](../src/pdf_rag/chat.py#L27-L35)).

For a comparison request:

1. `build_version_comparison_context` identifies the document and obtains changed sections from registry history ([`chat.py`](../src/pdf_rag/chat.py#L82-L123)).
2. If no document is identified, the function returns a clarification message and no citations.
3. If only one version exists, it returns that explanation directly.
4. Otherwise, `summarize_document_version_changes` asks the selected model to explain the removed and added text, and the function returns the summary with citations naming the versions compared ([`chat.py`](../src/pdf_rag/chat.py#L126-L149)).

This route bypasses vector retrieval because the registry contains the extracted text for each saved version and can compare it directly.

## 4. Normal Question: Rewrite the Query

Questions that are neither greetings nor version comparisons enter the RAG path. First, `optimize_and_normalize_query` asks the selected model to correct spelling and resolve follow-ups from conversation history ([`api.py`](../src/pdf_rag/api.py#L84), [`chat.py`](../src/pdf_rag/chat.py#L152-L178)).

For example, a misspelled question like “who is the principle of usa” may be rewritten as “Who is the president of the United States of America?” The rewrite is for searching; the original `message` is still used as the question to answer.

## 5. Retrieval and Retry Loop

The function runs at most three attempts ([`api.py`](../src/pdf_rag/api.py#L85-L93)).

- Attempt 1 searches without restricting to one source document.
- Attempts 2 and 3 first ask the model to select a document filename from the catalog; that filename is passed as `filter_source`.
- `retrieve_context` embeds the search query with Ollama and asks ChromaDB for the nearest active document chunks ([`vector_store.py`](../src/pdf_rag/vector_store.py#L114-L150)). It returns formatted excerpts plus source/page citations.

A retry is intended to narrow the search if the first draft is not accepted. The catalog is the model-visible list of processed filenames; document names also appear in the source metadata attached to retrieved chunks.

## 6. Draft Answer: What Goes to the Chat Model

Once passages are retrieved, the function makes a draft-answer request ([`api.py`](../src/pdf_rag/api.py#L94-L110)). Its messages consist of:

1. A system message instructing the model to answer only from the document context and emit `INSUFFICIENT_CONTEXT` if the context is irrelevant or insufficient. The retrieved excerpts are interpolated into this system message.
2. Up to six recent conversation messages.
3. The user's original question.

The request uses temperature `0.0` and allows up to 512 output tokens. The result is stored in `draft_answer`; it is not returned to the browser yet.

## 7. Verification and Important Current Limitation

The API then calls `evaluate_response_sufficiency(message, context, draft_answer)` ([`api.py`](../src/pdf_rag/api.py#L111)). The helper immediately rejects a draft containing `INSUFFICIENT_CONTEXT`; otherwise, it sends a second request asking the model whether the draft is useful and relevant ([`chat.py`](../src/pdf_rag/chat.py#L181-L212)).

**Current implementation caveat:** although the helper receives `context` as an argument, the verification prompt currently includes only `user_query` and `draft_answer`; it does not insert the retrieved context into the verifier request ([`chat.py`](../src/pdf_rag/chat.py#L187-L203)). So this step checks question/draft relevance, but it cannot independently confirm every draft claim against the source passages. The draft-answer prompt does contain the passages, but the separate verifier does not currently see them.

If verification returns `sufficient=True`, `_answer_request` returns the exact checked draft and sorted citations ([`api.py`](../src/pdf_rag/api.py#L111-L113)). Otherwise it repeats retrieval and drafting. If all three attempts fail, it returns a fixed “couldn't find enough relevant information” response with no citations ([`api.py`](../src/pdf_rag/api.py#L115-L118)).

## 8. Returning the Answer to React

The `/api/chat` endpoint calls `_answer_request`, then sends the result as Server-Sent Events ([`api.py`](../src/pdf_rag/api.py#L192-L226)):

1. `status`: tells the browser that document search started.
2. `sources`: carries the citation list.
3. One or more `token` events: the completed answer string is divided into 48-character pieces. These are transport chunks, not separate model generations.
4. `done`: identifies the configured provider.

Thus, the model creates a complete draft inside `_answer_request`; FastAPI subsequently slices that completed text for delivery to the React client. The browser joins the token pieces for display.

## Flow at a Glance

```text
Browser message + recent history
              |
              v
      _answer_request
        /     |      \
 greeting  compare   normal question
    model     registry    rewrite query
    reply       diff          |
                            Chroma search
                                |
                           draft answer
                                |
                         sufficiency check
                         /             \
                    rejected          accepted
                       |                 |
                 retry (up to 3)   answer + citations
                                         |
                                    SSE to React
```

## Model Calls Made on a Normal Question

A normal document question may produce several separate model calls:

1. Query rewrite.
2. Draft answer for each retrieval attempt.
3. Sufficiency check for each draft.
4. Catalog filename selection on retry attempts 2 and 3.

Greeting and version-comparison branches use different calls and skip normal vector retrieval. The exact number depends on whether the first answer is accepted.
