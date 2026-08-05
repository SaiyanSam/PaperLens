"""Page-wise PDF extraction for PaperLens."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz

from src.ingestion.text_cleaner import clean_page_text
from src.models.document import DocumentPage, ParsedDocument


class PDFExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened or extracted."""


def create_document_id(
    document_name: str,
    pdf_bytes: bytes,
) -> str:
    """Create a stable identifier from the filename and file contents."""

    digest = hashlib.sha256()
    digest.update(document_name.encode("utf-8"))
    digest.update(pdf_bytes)

    return digest.hexdigest()[:16]


def parse_pdf_bytes(
    pdf_bytes: bytes,
    document_name: str,
) -> ParsedDocument:
    """
    Parse a PDF supplied as bytes.

    Returns a validated ParsedDocument containing page-wise text,
    page numbers, and extraction statistics.
    """

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    document_name = Path(document_name).name.strip()

    if not document_name:
        raise ValueError("The PDF must have a valid filename.")

    if not document_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    try:
        pdf_document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise PDFExtractionError(
            f"Could not open '{document_name}' as a PDF."
        ) from exc

    try:
        if pdf_document.page_count == 0:
            raise PDFExtractionError(
                f"'{document_name}' contains no pages."
            )

        pages: list[DocumentPage] = []

        for page_index in range(pdf_document.page_count):
            page = pdf_document.load_page(page_index)

            try:
                raw_text = page.get_text("text", sort=True)
            except Exception as exc:
                raise PDFExtractionError(
                    f"Text extraction failed on page "
                    f"{page_index + 1} of '{document_name}'."
                ) from exc

            cleaned_text = clean_page_text(raw_text)

            page_record = DocumentPage(
                page_number=page_index + 1,
                text=cleaned_text,
                character_count=len(cleaned_text),
                word_count=len(cleaned_text.split()),
                is_empty=not bool(cleaned_text.strip()),
            )

            pages.append(page_record)

        return ParsedDocument(
            document_id=create_document_id(
                document_name=document_name,
                pdf_bytes=pdf_bytes,
            ),
            document_name=document_name,
            num_pages=len(pages),
            total_characters=sum(
                page.character_count for page in pages
            ),
            total_words=sum(
                page.word_count for page in pages
            ),
            empty_pages=[
                page.page_number
                for page in pages
                if page.is_empty
            ],
            pages=pages,
        )

    finally:
        pdf_document.close()


def parse_pdf_file(
    pdf_path: str | Path,
) -> ParsedDocument:
    """Parse a PDF stored on disk."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")

    return parse_pdf_bytes(
        pdf_bytes=path.read_bytes(),
        document_name=path.name,
    )


def save_parsed_document(
    document: ParsedDocument,
    output_directory: str | Path = "data/outputs",
) -> Path:
    """Save a parsed document as formatted JSON."""

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_stem = Path(document.document_name).stem
    output_path = output_directory / (
        f"{safe_stem}_{document.document_id}.json"
    )

    output_path.write_text(
        json.dumps(
            document.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path
