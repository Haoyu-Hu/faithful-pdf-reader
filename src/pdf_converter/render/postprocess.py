"""Markdown post-processing: whitespace normalization, optional mdformat."""

from __future__ import annotations

import re

_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_LEADING_WS_BEFORE_NEWLINE = re.compile(r"\n[ \t]+\n")


def normalize_markdown(md: str) -> str:
    """Cheap pre-mdformat cleanup. Keeps content intact, just tidies whitespace."""
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = _TRAILING_WS.sub("\n", md)
    md = _LEADING_WS_BEFORE_NEWLINE.sub("\n\n", md)
    md = _MULTI_BLANK.sub("\n\n", md)
    return md.strip() + "\n"


def mdformat_text(md: str) -> str:
    """Run mdformat with the GFM extension. Returns input unchanged on failure."""
    try:
        import mdformat  # type: ignore

        return mdformat.text(md, extensions={"gfm"})
    except Exception:
        return md


__all__ = ["mdformat_text", "normalize_markdown"]
