"""Tests for PaperLens document chunking."""

from __future__ import annotations

from src.ingestion.chunker import chunk_document
from src.models.document import DocumentPage, ParsedDocument


def make_document(page_texts: list[str]) -> ParsedDocument:
    """Build a validated document without creating a PDF."""

    pages = [
        DocumentPage(
            page_number=index,
            text=text,
            character_count=len(text),
            word_count=len(text.split()),
            is_empty=not bool(text.strip()),
        )
        for index, text in enumerate(page_texts, start=1)
    ]

    return ParsedDocument(
        document_id="test-document",
        document_name="test.pdf",
        num_pages=len(pages),
        total_characters=sum(
            page.character_count for page in pages
        ),
        total_words=sum(page.word_count for page in pages),
        empty_pages=[
            page.page_number for page in pages if page.is_empty
        ],
        pages=pages,
    )


def test_short_page_produces_one_chunk() -> None:
    document = make_document(["one two three four"])

    chunks = chunk_document(
        document,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "one two three four"
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1


def test_long_page_produces_overlapping_chunks() -> None:
    document = make_document(
        ["one two three four five six seven eight"]
    )

    chunks = chunk_document(
        document,
        chunk_size=5,
        chunk_overlap=2,
    )

    assert len(chunks) == 2
    assert chunks[0].text == "one two three four five"
    assert chunks[1].text == "four five six seven eight"


def test_chunks_do_not_cross_page_boundaries() -> None:
    document = make_document(
        [
            "page one contains several words",
            "page two contains other words",
        ]
    )

    chunks = chunk_document(
        document,
        chunk_size=3,
        chunk_overlap=1,
    )

    assert all(
        chunk.page_start == chunk.page_end
        for chunk in chunks
    )

    assert {chunk.page_start for chunk in chunks} == {1, 2}


def test_empty_pages_are_skipped() -> None:
    document = make_document(
        [
            "page with text",
            "",
        ]
    )

    chunks = chunk_document(
        document,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert len(chunks) == 1
    assert chunks[0].page_start == 1


def test_chunk_ids_are_stable() -> None:
    document = make_document(["one two three four five"])

    first = chunk_document(
        document,
        chunk_size=3,
        chunk_overlap=1,
    )
    second = chunk_document(
        document,
        chunk_size=3,
        chunk_overlap=1,
    )

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_invalid_overlap_is_rejected() -> None:
    document = make_document(["one two three"])

    try:
        chunk_document(
            document,
            chunk_size=3,
            chunk_overlap=3,
        )
    except ValueError as exc:
        assert "smaller than chunk_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
