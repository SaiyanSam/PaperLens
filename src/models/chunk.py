"""Validated chunk and retrieval models for PaperLens."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class DocumentChunk(BaseModel):
    """A searchable text chunk derived from one PDF page."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_name: str = Field(min_length=1)

    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    chunk_index: int = Field(ge=0)
    section: str | None = None

    text: str = Field(min_length=1)
    character_count: int = Field(ge=1)
    word_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_chunk(self) -> "DocumentChunk":
        """Ensure chunk metadata agrees with its text."""

        if self.page_end < self.page_start:
            raise ValueError(
                "page_end cannot be smaller than page_start"
            )

        if self.character_count != len(self.text):
            raise ValueError(
                "character_count does not match chunk text"
            )

        if self.word_count != len(self.text.split()):
            raise ValueError(
                "word_count does not match chunk text"
            )

        return self


class RetrievedChunk(BaseModel):
    """A document chunk returned by semantic retrieval."""

    rank: int = Field(ge=1)
    chunk: DocumentChunk
    similarity_score: float = Field(ge=-1.0, le=1.0)
