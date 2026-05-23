"""Frequency-based header / footer detection.

Replaces the reference's hardcoded ``y < 50px`` heuristic. Strings that recur
on at least ``chrome_freq_threshold`` of pages within the header or footer
band are dropped from the IR.

Only TextBlock / HeadingBlock content is considered for chrome; tables,
figures, equations, code, lists are never treated as chrome.
"""

from __future__ import annotations

import re
from collections import Counter

from pdf_converter.config import Config
from pdf_converter.ir import Document, HeadingBlock, TextBlock
from pdf_converter.ir.blocks import BlockKind
from pdf_converter.utils.logging import get_logger

_logger = get_logger("extractors.heuristic.chrome")

_WS_RE = re.compile(r"\s+")
_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s+)?\d+(?:\s*(?:/|of)\s*\d+)?\s*$", re.IGNORECASE)


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.strip()).lower()


class ChromeDetector:
    name = "heuristic.chrome"

    def __call__(self, doc: Document, *, fitz_doc, config: Config) -> None:
        if doc.n_pages == 0:
            return

        top = config.heuristic.header_band_ratio
        bot = config.heuristic.footer_band_ratio
        freq = config.heuristic.chrome_freq_threshold
        min_pages = max(2, config.heuristic.chrome_min_pages)

        # Tally text appearing in header or footer band, per-page.
        # Page-number-shaped strings are treated as a single class regardless of
        # the actual number ("1", "2", ... -> "<pgnum>").
        per_page: list[set[str]] = []
        for page in doc.pages:
            band_top = page.height * top
            band_bot = page.height * (1.0 - bot)
            here: set[str] = set()
            for b in page.blocks:
                if b.kind not in (BlockKind.TEXT, BlockKind.HEADING):
                    continue
                text = _get_text(b)
                if not text.strip():
                    continue
                in_top = b.bbox.y1 <= band_top
                in_bot = b.bbox.y0 >= band_bot
                if not (in_top or in_bot):
                    continue
                key = "<pgnum>" if _PAGE_NUM_RE.match(text) else _normalize(text)
                here.add(key)
            per_page.append(here)

        counter: Counter[str] = Counter()
        for s in per_page:
            counter.update(s)

        n_pages = doc.n_pages
        chrome: set[str] = {
            k
            for k, c in counter.items()
            if c >= min_pages and (c / n_pages) >= freq
        }
        # Page numbers are always chrome regardless of recurrence count -- a
        # 2-page doc still has 2 page numbers and that's what they are.
        if counter.get("<pgnum>", 0) >= 1:
            chrome.add("<pgnum>")
        doc.chrome_texts = chrome
        _logger.debug("detected chrome strings: %s", chrome)

        # Second pass: drop chrome blocks.
        for page in doc.pages:
            band_top = page.height * top
            band_bot = page.height * (1.0 - bot)
            keep = []
            for b in page.blocks:
                if b.kind in (BlockKind.TEXT, BlockKind.HEADING):
                    text = _get_text(b)
                    key = "<pgnum>" if _PAGE_NUM_RE.match(text) else _normalize(text)
                    in_band = b.bbox.y1 <= band_top or b.bbox.y0 >= band_bot
                    if in_band and key in chrome:
                        continue
                keep.append(b)
            page.blocks = keep


def _get_text(b) -> str:
    if isinstance(b, (TextBlock, HeadingBlock)):
        return "".join(s.text for s in b.spans)
    return ""


__all__ = ["ChromeDetector"]
