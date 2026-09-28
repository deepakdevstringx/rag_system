import re

from .config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text near sentence and paragraph boundaries with shared overlap."""
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