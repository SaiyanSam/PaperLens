"""Lightweight text cleaning for extracted PDF pages."""

from __future__ import annotations

import re


def clean_page_text(text: str) -> str:
    """
    Clean text extracted from a PDF page.

    The cleaning is intentionally conservative so technical content,
    equations, symbols, and section structure are not aggressively altered.
    """

    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\x00", "")

    # Join words broken across lines, for example:
    # "reinforce-\nment" -> "reinforcement"
    cleaned = re.sub(r"(?<=\w)-\n(?=\w)", "", cleaned)

    # Normalize repeated spaces and tabs.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Remove spaces surrounding line breaks.
    cleaned = re.sub(r" *\n *", "\n", cleaned)

    # Avoid very large runs of blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
