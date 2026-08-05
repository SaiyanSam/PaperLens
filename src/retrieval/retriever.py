"""High-level semantic retrieval for PaperLens."""

from __future__ import annotations

from src.models.chunk import DocumentChunk, RetrievedChunk
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import FAISSVectorStore


class SemanticRetriever:
    """Coordinate query embedding and FAISS search."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
    ) -> None:
        if embedding_model.dimension != vector_store.dimension:
            raise ValueError(
                "Embedding-model dimension does not match "
                "vector-store dimension"
            )

        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return the chunks most relevant to a query."""

        query_embedding = self.embedding_model.encode_query(query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )


def build_retriever(
    chunks: list[DocumentChunk],
    embedding_model: EmbeddingModel | None = None,
    batch_size: int = 32,
) -> SemanticRetriever:
    """Build an in-memory retriever from document chunks."""

    if not chunks:
        raise ValueError(
            "At least one document chunk is required"
        )

    model = embedding_model or EmbeddingModel()

    embeddings = model.encode_documents(
        [chunk.text for chunk in chunks],
        batch_size=batch_size,
    )

    store = FAISSVectorStore(
        dimension=model.dimension
    )

    store.add(
        embeddings=embeddings,
        chunks=chunks,
    )

    return SemanticRetriever(
        embedding_model=model,
        vector_store=store,
    )
