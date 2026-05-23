"""Font analysis helpers used by heading and code-block detection."""

from __future__ import annotations

import re
import statistics
from collections import Counter

from pdf_converter.ir.span import Span

_MONOSPACE_HINTS = (
    "mono", "courier", "consolas", "menlo", "fira", "inconsolata",
    "source code", "ubuntu mono", "cascadia", "iosevka", "jetbrains", "ibmplexmono",
)


def is_monospace_font(font_name: str, flags_is_monospace: bool = False) -> bool:
    """Best-effort detection from font name and PDF flag bit 3."""
    if flags_is_monospace:
        return True
    n = font_name.lower()
    return any(h in n for h in _MONOSPACE_HINTS)


def font_stats(spans: list[Span]) -> dict[str, float]:
    """Return body-text stats: median size, p10, p90, std.

    Whitespace-only spans are skipped.
    """
    sizes = [s.font_size for s in spans if s.text.strip()]
    if not sizes:
        return {"median": 0.0, "p10": 0.0, "p90": 0.0, "std": 0.0}
    sizes_sorted = sorted(sizes)
    n = len(sizes_sorted)
    return {
        "median": statistics.median(sizes_sorted),
        "p10": sizes_sorted[max(0, n // 10)],
        "p90": sizes_sorted[min(n - 1, (n * 9) // 10)],
        "std": statistics.stdev(sizes_sorted) if n > 1 else 0.0,
    }


# U+FFFD (replacement char) plus C0 control range minus tab/newline.
_REPLACEMENT_CHAR_RE = re.compile("[�\x00-\x08\x0b\x0c\x0e-\x1f]")


def is_cid_broken(text: str, threshold: float = 0.20) -> bool:
    """Detect spans where >=`threshold` of characters are PDF-CID rubbish.

    PyMuPDF returns U+FFFD or control characters when a font lacks a
    ToUnicode CMap. We use this signal to route the page to OCR (when ML is
    enabled) or to log a clear warning.
    """
    if not text:
        return False
    bad = len(_REPLACEMENT_CHAR_RE.findall(text))
    return (bad / len(text)) >= threshold


# Math glyph range -- intentionally uses Unicode ranges that ruff RUF001
# warns about (some chars look ASCII-like). Suppress for the whole regex.
_MATH_GLYPH_RE = re.compile(
    "["
    "Ͱ-Ͽ"        # Greek
    "∀-⋿"        # Mathematical operators  # noqa: RUF001
    "⟀-⟯"        # Misc math A
    "⦀-⧿"        # Misc math B
    "⨀-⫿"        # Supplemental math operators
    "℀-⅏"        # Letterlike symbols
    "=+\\-*/<>"
    "≤≥±∓∞×÷≈≠≡∝"  # noqa: RUF001
    "∂∇∫∑∏∮"
    "]"
)


def math_density(text: str) -> float:
    """Fraction of characters in ``text`` that are math-likely glyphs."""
    if not text:
        return 0.0
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return 0.0
    matches = sum(1 for c in visible if _MATH_GLYPH_RE.match(c))
    return matches / len(visible)


def font_histogram(spans: list[Span]) -> Counter[tuple[str, float]]:
    """Return a counter of (font_name, rounded_size) for inspection / debug."""
    return Counter((s.font, round(s.font_size, 1)) for s in spans if s.text.strip())
