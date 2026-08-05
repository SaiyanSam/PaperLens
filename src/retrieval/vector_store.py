"""FAISS-backed vector storage for PaperLens."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.models.chunk import DocumentChunk, RetrievedChunk


class VectorStoreError(RuntimeError):
    """Raised when vector-store state or input is invalid."""


class FAISSVectorStore:
    """
    Store normalized chunk embeddings in a FAISS inner-product index.

    Because embeddings are L2-normalized, inner product is equivalent
    to cosine similarity.
    """

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[DocumentChunk] = []

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""

        return int(self.index.ntotal)

    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[DocumentChunk],
    ) -> None:
        """Add aligned embeddings and chunk metadata."""

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must be a two-dimensional array"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"received {embeddings.shape[1]}"
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks"
            )

        if not chunks:
            return

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Search for the most similar indexed chunks."""

        if self.size == 0:
            raise VectorStoreError(
                "Cannot search an empty vector store"
            )

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape != (1, self.dimension):
            raise ValueError(
                "query_embedding must have shape "
                f"(1, {self.dimension})"
            )

        result_count = min(top_k, self.size)

        scores, indices = self.index.search(
            query_embedding,
            result_count,
        )

        results: list[RetrievedChunk] = []

        for rank, (score, index_position) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            if index_position < 0:
                continue

            results.append(
                RetrievedChunk(
                    rank=rank,
                    chunk=self.chunks[int(index_position)],
                    similarity_score=float(score),
                )
            )

        return results

    def save(
        self,
        directory: str | Path,
    ) -> None:
        """Persist the FAISS index and chunk metadata."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss"),
        )

        metadata = {
            "dimension": self.dimension,
            "chunks": [
                chunk.model_dump()
                for chunk in self.chunks
            ],
        }

        (directory / "metadata.json").write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "FAISSVectorStore":
        """Load a persisted vector store."""

        directory = Path(directory)

        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}"
            )

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        store = cls(
            dimension=int(metadata["dimension"])
        )

        store.index = faiss.read_index(
            str(index_path)
        )

        store.chunks = [
            DocumentChunk.model_validate(record)
            for record in metadata["chunks"]
        ]

        if store.index.ntotal != len(store.chunks):
            raise VectorStoreError(
                "FAISS index size does not match chunk metadata"
            )

        return store
