"""Sentence Transformer embedding wrapper for PaperLens."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Encode document chunks and user queries into normalized vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    @property
    def dimension(self) -> int:
        """Return the model embedding dimension."""
    
        return int(
            self.model.get_embedding_dimension()
        )

    def encode_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode document texts as normalized float32 vectors."""

        if not texts:
            return np.empty(
                shape=(0, self.dimension),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one query as a normalized float32 vector."""

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embedding, dtype=np.float32)
