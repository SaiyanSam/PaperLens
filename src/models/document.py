"""Validated document models for PaperLens."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class DocumentPage(BaseModel):
    """Text and metadata extracted from one PDF page."""

    page_number: int = Field(ge=1)
    text: str
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    is_empty: bool = False

    @model_validator(mode="after")
    def validate_statistics(self) -> "DocumentPage":
        expected_characters = len(self.text)
        expected_words = len(self.text.split())
        expected_empty = not bool(self.text.strip())

        if self.character_count != expected_characters:
            raise ValueError(
                "character_count does not match the page text"
            )

        if self.word_count != expected_words:
            raise ValueError(
                "word_count does not match the page text"
            )

        if self.is_empty != expected_empty:
            raise ValueError(
                "is_empty does not match the page text"
            )

        return self


class ParsedDocument(BaseModel):
    """Validated representation of an extracted PDF."""

    document_id: str = Field(min_length=1)
    document_name: str = Field(min_length=1)
    num_pages: int = Field(ge=1)
    total_characters: int = Field(ge=0)
    total_words: int = Field(ge=0)
    empty_pages: list[int] = Field(default_factory=list)
    pages: list[DocumentPage]

    @model_validator(mode="after")
    def validate_document(self) -> "ParsedDocument":
        if len(self.pages) != self.num_pages:
            raise ValueError(
                "num_pages does not match the number of page records"
            )

        expected_page_numbers = list(range(1, self.num_pages + 1))
        actual_page_numbers = [
            page.page_number for page in self.pages
        ]

        if actual_page_numbers != expected_page_numbers:
            raise ValueError(
                "pages must be ordered and numbered from 1"
            )

        expected_characters = sum(
            page.character_count for page in self.pages
        )
        expected_words = sum(
            page.word_count for page in self.pages
        )
        expected_empty_pages = [
            page.page_number
            for page in self.pages
            if page.is_empty
        ]

        if self.total_characters != expected_characters:
            raise ValueError(
                "total_characters does not match page statistics"
            )

        if self.total_words != expected_words:
            raise ValueError(
                "total_words does not match page statistics"
            )

        if self.empty_pages != expected_empty_pages:
            raise ValueError(
                "empty_pages does not match page records"
            )

        return self
