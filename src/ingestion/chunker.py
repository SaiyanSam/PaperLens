"""Page-aware overlapping text chunking for PaperLens."""

from __future__ import annotations

import hashlib

from src.models.chunk import DocumentChunk
from src.models.document import ParsedDocument


DEFAULT_CHUNK_SIZE = 180
DEFAULT_CHUNK_OVERLAP = 30


def create_chunk_id(
    document_id: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    """Create a stable identifier for an extracted text chunk."""

    digest = hashlib.sha256()

    digest.update(document_id.encode("utf-8"))
    digest.update(str(page_number).encode("utf-8"))
    digest.update(str(chunk_index).encode("utf-8"))
    digest.update(text.encode("utf-8"))

    return digest.hexdigest()[:20]


def chunk_document(
    document: ParsedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """
    Split a parsed PDF into overlapping page-level chunks.

    Chunks do not cross page boundaries, which keeps page citations exact.

    Args:
        document: Validated parsed PDF.
        chunk_size: Maximum number of whitespace-delimited words.
        chunk_overlap: Number of words shared by consecutive chunks.

    Returns:
        Ordered searchable chunks.

    Raises:
        ValueError: For invalid chunking parameters.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    step_size = chunk_size - chunk_overlap
    chunks: list[DocumentChunk] = []
    global_chunk_index = 0

    for page in document.pages:
        words = page.text.split()

        if not words:
            continue

        for start_index in range(0, len(words), step_size):
            chunk_words = words[
                start_index : start_index + chunk_size
            ]

            if not chunk_words:
                continue

            chunk_text = " ".join(chunk_words)

            chunks.append(
                DocumentChunk(
                    chunk_id=create_chunk_id(
                        document_id=document.document_id,
                        page_number=page.page_number,
                        chunk_index=global_chunk_index,
                        text=chunk_text,
                    ),
                    document_id=document.document_id,
                    document_name=document.document_name,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    chunk_index=global_chunk_index,
                    section=None,
                    text=chunk_text,
                    character_count=len(chunk_text),
                    word_count=len(chunk_words),
                )
            )

            global_chunk_index += 1

            # The final chunk already contains all remaining words.
            if start_index + chunk_size >= len(words):
                break

    return chunks


def chunk_documents(
    documents: list[ParsedDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Chunk several documents into one ordered collection."""

    chunks: list[DocumentChunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks
