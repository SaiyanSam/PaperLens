"""Tests for the PaperLens PDF ingestion pipeline.

Use:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m 
"""

from __future__ import annotations

import fitz
import pytest

from src.ingestion.pdf_parser import (
    PDFExtractionError,
    parse_pdf_bytes,
)


def create_test_pdf(page_texts: list[str]) -> bytes:
    """Create an in-memory PDF with the supplied page text."""

    document = fitz.open()

    for text in page_texts:
        page = document.new_page()

        if text:
            page.insert_text(
                point=(72, 72),
                text=text,
            )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def test_parse_pdf_preserves_page_metadata() -> None:
    """The parser should preserve text and one-based page numbers."""

    pdf_bytes = create_test_pdf(
        [
            "First page of the research paper.",
            "Second page containing experimental results.",
        ]
    )

    document = parse_pdf_bytes(
        pdf_bytes=pdf_bytes,
        document_name="sample.pdf",
    )

    assert document.document_name == "sample.pdf"
    assert document.num_pages == 2
    assert len(document.pages) == 2

    assert document.pages[0].page_number == 1
    assert "First page" in document.pages[0].text

    assert document.pages[1].page_number == 2
    assert "experimental results" in document.pages[1].text


def test_parser_calculates_document_statistics() -> None:
    """Aggregate statistics should match the individual pages."""

    pdf_bytes = create_test_pdf(
        [
            "One two three.",
            "Four five.",
        ]
    )

    document = parse_pdf_bytes(
        pdf_bytes=pdf_bytes,
        document_name="statistics.pdf",
    )

    assert document.total_words == 5

    expected_characters = sum(
        len(page.text)
        for page in document.pages
    )

    assert document.total_characters == expected_characters


def test_parse_pdf_detects_empty_pages() -> None:
    """Pages without extractable text should be recorded."""

    pdf_bytes = create_test_pdf(
        [
            "Page containing text.",
            "",
        ]
    )

    document = parse_pdf_bytes(
        pdf_bytes=pdf_bytes,
        document_name="empty-page.pdf",
    )

    assert document.num_pages == 2
    assert document.pages[1].is_empty is True
    assert document.pages[1].word_count == 0
    assert document.empty_pages == [2]


def test_document_id_is_stable() -> None:
    """The same name and PDF contents should produce the same ID."""

    pdf_bytes = create_test_pdf(
        ["Stable document content."]
    )

    first = parse_pdf_bytes(
        pdf_bytes=pdf_bytes,
        document_name="paper.pdf",
    )

    second = parse_pdf_bytes(
        pdf_bytes=pdf_bytes,
        document_name="paper.pdf",
    )

    assert first.document_id == second.document_id


def test_different_documents_have_different_ids() -> None:
    """Different PDF contents should produce different IDs."""

    first_pdf = create_test_pdf(["First document."])
    second_pdf = create_test_pdf(["Second document."])

    first = parse_pdf_bytes(
        pdf_bytes=first_pdf,
        document_name="paper.pdf",
    )

    second = parse_pdf_bytes(
        pdf_bytes=second_pdf,
        document_name="paper.pdf",
    )

    assert first.document_id != second.document_id


def test_empty_upload_is_rejected() -> None:
    """An empty upload should fail before PDF parsing."""

    with pytest.raises(ValueError, match="empty"):
        parse_pdf_bytes(
            pdf_bytes=b"",
            document_name="empty.pdf",
        )


def test_non_pdf_filename_is_rejected() -> None:
    """The parser should reject unsupported file extensions."""

    with pytest.raises(ValueError, match="Only PDF"):
        parse_pdf_bytes(
            pdf_bytes=b"not a PDF",
            document_name="notes.txt",
        )


def test_corrupted_pdf_is_rejected() -> None:
    """Invalid PDF bytes should produce a controlled parser error."""

    with pytest.raises(PDFExtractionError):
        parse_pdf_bytes(
            pdf_bytes=b"This is not valid PDF data.",
            document_name="corrupted.pdf",
        )
