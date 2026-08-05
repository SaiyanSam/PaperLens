"""Tests for the PaperLens FAISS vector store."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.chunk import DocumentChunk
from src.retrieval.vector_store import (
    FAISSVectorStore,
    VectorStoreError,
)


def make_chunk(
    chunk_index: int,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{chunk_index}",
        document_id="document-1",
        document_name="paper.pdf",
        page_start=chunk_index + 1,
        page_end=chunk_index + 1,
        chunk_index=chunk_index,
        section=None,
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
    )


def test_vector_store_returns_nearest_chunk() -> None:
    chunks = [
        make_chunk(0, "robot exploration"),
        make_chunk(1, "image classification"),
        make_chunk(2, "language modelling"),
    ]

    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store = FAISSVectorStore(dimension=3)
    store.add(embeddings, chunks)

    query = np.asarray(
        [[0.9, 0.1, 0.0]],
        dtype=np.float32,
    )

    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk-0"
    assert results[0].rank == 1


def test_top_k_is_limited_by_store_size() -> None:
    chunk = make_chunk(0, "robot exploration")

    store = FAISSVectorStore(dimension=2)
    store.add(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        [chunk],
    )

    results = store.search(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        top_k=5,
    )

    assert len(results) == 1


def test_embedding_count_must_match_chunks() -> None:
    store = FAISSVectorStore(dimension=2)

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="match"):
        store.add(
            embeddings,
            [make_chunk(0, "only one chunk")],
        )


def test_wrong_embedding_dimension_is_rejected() -> None:
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(ValueError, match="dimension"):
        store.add(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            [make_chunk(0, "text")],
        )


def test_empty_store_cannot_be_searched() -> None:
    store = FAISSVectorStore(dimension=2)

    with pytest.raises(VectorStoreError, match="empty"):
        store.search(
            np.asarray([[1.0, 0.0]], dtype=np.float32)
        )


def test_vector_store_save_and_load(
    tmp_path,
) -> None:
    chunks = [
        make_chunk(0, "robot exploration"),
        make_chunk(1, "image classification"),
    ]

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    original = FAISSVectorStore(dimension=2)
    original.add(embeddings, chunks)
    original.save(tmp_path)

    loaded = FAISSVectorStore.load(tmp_path)

    assert loaded.dimension == 2
    assert loaded.size == 2
    assert loaded.chunks == chunks

    results = loaded.search(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        top_k=1,
    )

    assert results[0].chunk.chunk_id == "chunk-0"
