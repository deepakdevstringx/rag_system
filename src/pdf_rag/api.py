"""HTTP API for the React RAG client."""

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .chat import (
    build_version_comparison_context,
    evaluate_response_sufficiency,
    find_target_document_from_catalog,
    is_simple_greeting,
    is_version_comparison_query,
    optimize_and_normalize_query,
    summarize_document_version_changes,
)
from .chat_model import collect_model_usage, get_chat_selection, invoke_chat
from .config import CHAT_PROVIDER, INBOX_DIR
from .document_registry import load_registry
from .vector_store import ingest_pending_pdfs, retrieve_context


app = FastAPI(title="Local PDF RAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_ingestion_lock = threading.Lock()
_MAX_PDF_BYTES = 30 * 1024 * 1024


class ChatRequest(BaseModel):
    """User message and recent conversation supplied by the browser."""

    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)


def _event(name, payload):
    """Serialize one server-sent event with JSON data."""
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _answer_request(message, history):
    """Retrieve evidence, draft an answer, and verify it before returning."""
    if is_simple_greeting(message):
        answer = invoke_chat(
            [
                {
                    "role": "system",
                    "content": "Respond briefly and warmly. Invite the user to ask about their uploaded company documents.",
                },
                *history[-6:],
                {"role": "user", "content": message},
            ],
            temperature=0.3,
            max_tokens=100,
            operation="greeting_response",
        )
        return answer, []

    if is_version_comparison_query(message):
        comparison_context, citations = build_version_comparison_context(message)
        if comparison_context is None:
            return (
                "I couldn't identify a versioned document. Include part of its filename, "
                "for example: 'compare Hackathon 2026 versions'.",
                [],
            )
        if comparison_context.startswith("Only one version is available"):
            return comparison_context, sorted(citations)
        answer = summarize_document_version_changes(message, comparison_context, history)
        return answer, sorted(citations)

    search_query = optimize_and_normalize_query(message, history)
    for attempt in range(3):
        target_document = (
            find_target_document_from_catalog(search_query) if attempt > 0 else None
        )
        context, citations = retrieve_context(
            search_query,
            k=5,
            filter_source=target_document,
        )
        draft_answer = invoke_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the document context. "
                        "Do not add unsupported facts. If the context is irrelevant or "
                        "insufficient, output exactly INSUFFICIENT_CONTEXT.\n\n"
                        f"DOCUMENT CONTEXT:\n{context}"
                    ),
                },
                *history[-6:],
                {"role": "user", "content": message},
            ],
            temperature=0.0,
            max_tokens=512,
            operation="answer_draft",
        )
        verified, _reason = evaluate_response_sufficiency(message, context, draft_answer)
        if verified:
            return draft_answer, sorted(citations)

    return (
        "I couldn't find enough relevant information in the processed documents to answer accurately.",
        [],
    )


def _answer_request_with_usage(message, history):
    """Run one answer flow and return the usage record for every model call."""
    with collect_model_usage() as usage_records:
        answer, citations = _answer_request(message, history)
    return answer, citations, usage_records


@app.get("/api/health")
def health_check():
    """Expose API and active-model status for the frontend connection indicator."""
    provider, model = get_chat_selection()
    return {"status": "ok", "provider": provider, "model": model}


@app.get("/api/documents")
def list_documents():
    """List each processed PDF with its latest version and archived revisions."""
    registry = load_registry()
    documents = []
    for source_name, document_info in sorted(registry.items()):
        versions = sorted(int(version) for version in document_info.get("history", {}))
        documents.append(
            {
                "name": source_name,
                "latest_version": document_info.get("latest_version", 1),
                "versions": versions,
                "version_count": len(versions),
                "latest_updated": document_info.get("history", {})
                .get(str(document_info.get("latest_version", 1)), {})
                .get("timestamp"),
            }
        )
    return {"documents": documents, "total": len(documents)}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Validate, stage, and ingest one PDF, returning its processed version."""
    safe_name = Path(file.filename or "").name
    if not safe_name or not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    content = await file.read(_MAX_PDF_BYTES + 1)
    if len(content) > _MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF must be 30 MB or smaller.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

    destination = INBOX_DIR / safe_name
    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail=f"'{safe_name}' is already waiting to be processed.",
        )

    destination.write_bytes(content)
    try:
        await run_in_threadpool(_ingest_with_lock)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {error}") from error

    registry = load_registry()
    document_info = registry.get(safe_name)
    if document_info is None:
        raise HTTPException(status_code=500, detail="The PDF was not added to the document registry.")

    return {
        "name": safe_name,
        "latest_version": document_info["latest_version"],
        "message": "PDF processed and added to search.",
    }


def _ingest_with_lock():
    """Serialize Chroma ingestion so concurrent uploads cannot race."""
    with _ingestion_lock:
        ingest_pending_pdfs()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Stream a verified RAG answer and its source citations as SSE events."""
    cleaned_history = [
        {"role": item["role"], "content": item["content"]}
        for item in request.history
        if item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ][-6:]

    async def event_stream():
        yield _event("status", {"message": "Searching processed documents"})
        try:
            answer, citations, model_usage = await run_in_threadpool(
                _answer_request_with_usage,
                request.message,
                cleaned_history,
            )
            yield _event("sources", {"items": citations})
            yield _event("usage", {"calls": model_usage})
            for offset in range(0, len(answer), 48):
                yield _event("token", {"text": answer[offset:offset + 48]})
                await asyncio.sleep(0)
            yield _event("done", {"provider": CHAT_PROVIDER})
        except Exception as error:
            yield _event("error", {"message": str(error)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )