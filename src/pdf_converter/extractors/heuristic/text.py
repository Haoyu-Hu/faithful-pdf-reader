"""Heuristic text extractor: PyMuPDF blocks -> TextBlock / HeadingBlock / ListBlock.

Key improvements over the reference:

* Headings detected by **z-score over the body-text font size distribution**,
  not absolute thresholds. Each document calibrates itself.
* List items kept as ``ListBlock`` with nested ``ListItem`` structure, so the
  renderer can emit proper markdown lists.
* Spans preserved with bold/italic/monospace/super-sub flags for downstream
  formatting.
* Headers and footers are *not* dropped here — that is the job of
  ``ChromeDetector`` (frequency-based), which runs after all pages are loaded.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pdf_converter.config import Config
from pdf_converter.ir import (
    BBox,
    Document,
    HeadingBlock,
    ListBlock,
    ListItem,
    Page,
    Provenance,
    Span,
    TextBlock,
)
from pdf_converter.utils.fonts import font_stats, is_monospace_font
from pdf_converter.utils.logging import get_logger

if TYPE_CHECKING:
    pass  # type: ignore

_logger = get_logger("extractors.heuristic.text")

# Bullet characters considered list markers (unordered)
_BULLETS = "•◦▪▫●○⦁∙·-‣⁃"  # noqa: RUF001 - intentional Unicode bullets
# Numbered list patterns: "1." "1)" "(1)" "a." "A)" "i)" "iv."
_NUMBERED_RE = re.compile(
    r"^\s*(?:\((?P<paren>[0-9]+|[a-zA-Z]+|[ivxlcdm]+)\)|"
    r"(?P<plain>[0-9]+|[a-zA-Z]|[ivxlcdm]+)[.)])\s+",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(rf"^\s*([{re.escape(_BULLETS)}])\s+")
_HR_RE = re.compile(r"^[\s_\-=*]{3,}$")


class TextExtractor:
    """Populate ``doc`` with TextBlock/HeadingBlock/ListBlock per page."""

    name = "heuristic.text"

    def __call__(self, doc: Document, *, fitz_doc, config: Config) -> None:
        prov = Provenance(extractor=self.name)

        # First pass: collect all spans to compute document-level font stats.
        all_spans = list(self._iter_all_spans(fitz_doc))
        stats = font_stats(all_spans)
        doc.body_font_size = stats["median"]

        # Z-score threshold over the body distribution.
        std = stats["std"] or 1.0
        z = config.heuristic.heading_zscore_threshold

        # Second pass: per-page extraction.
        for page_idx, fpage in enumerate(fitz_doc):
            page = Page(
                index=page_idx,
                width=float(fpage.rect.width),
                height=float(fpage.rect.height),
                rotation=int(fpage.rotation),
            )
            doc.pages.append(page)

            links = self._collect_links(fpage)

            raw = fpage.get_text("dict")
            for block in raw.get("blocks", []):
                if block.get("type") != 0:           # text only here
                    continue
                self._handle_text_block(
                    block,
                    page=page,
                    prov=prov,
                    page_idx=page_idx,
                    body_size=stats["median"],
                    std=std,
                    z=z,
                    links=links,
                )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _iter_all_spans(fitz_doc) -> list[Span]:
        out: list[Span] = []
        for fpage in fitz_doc:
            raw = fpage.get_text("dict")
            for block in raw.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        out.append(
                            Span.from_pymupdf(
                                text=span.get("text", ""),
                                bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                                font=span.get("font", ""),
                                font_size=float(span.get("size", 0.0)),
                                flags=int(span.get("flags", 0)),
                                color=int(span.get("color", 0)),
                            )
                        )
        return out

    @staticmethod
    def _collect_links(fpage):
        """Return list of {rect, uri} for URI-type links on the page."""
        try:
            return [
                {"rect": link["from"], "uri": link["uri"]}
                for link in fpage.get_links()
                if link.get("kind") == 2 and link.get("uri")
            ]
        except Exception as e:
            _logger.debug("link collection failed: %s", e)
            return []

    def _handle_text_block(
        self,
        block,
        *,
        page: Page,
        prov: Provenance,
        page_idx: int,
        body_size: float,
        std: float,
        z: float,
        links: list[dict],
    ) -> None:
        bbox = BBox.from_tuple(tuple(block["bbox"]))

        # Build a per-line view of spans.
        lines: list[list[Span]] = []
        for line in block.get("lines", []):
            line_spans: list[Span] = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                ir_span = Span.from_pymupdf(
                    text=text,
                    bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                    font=span.get("font", ""),
                    font_size=float(span.get("size", 0.0)),
                    flags=int(span.get("flags", 0)),
                    color=int(span.get("color", 0)),
                    link=_match_link(span.get("bbox"), links),
                )
                line_spans.append(ir_span)
            if line_spans:
                lines.append(line_spans)

        if not lines:
            return

        # Heading detection: single-line block whose dominant font size is
        # >= body + z * std (or whose spans are mostly bold and noticeably
        # larger than body).
        is_single_line = len(lines) == 1
        avg_size = sum(s.font_size for line in lines for s in line) / max(
            1, sum(len(line) for line in lines)
        )
        is_dominant_bold = (
            sum(1 for line in lines for s in line if s.is_bold)
            / max(1, sum(len(line) for line in lines))
            > 0.6
        )

        # Heading detection. Two paths:
        #   - Single-line block whose font is significantly larger / bolder than body.
        #   - Multi-line block where every line is dominantly bold and font is >= 1.10 *
        #     body (covers paper titles wrapped over 2-3 lines).
        if body_size > 0 and avg_size > body_size:
            ratio = avg_size / body_size
            zscore = (avg_size - body_size) / max(std, 0.5)
            qualifies_by_zscore = zscore >= z or (is_dominant_bold and zscore >= z * 0.5)
            qualifies_by_ratio = ratio >= 1.10
            multi_line_title = (
                not is_single_line
                and is_dominant_bold
                and ratio >= 1.10
                and len(lines) <= 3
            )
            if multi_line_title or (
                is_single_line and (qualifies_by_zscore or qualifies_by_ratio)
            ):
                level = _heading_level_from_ratio(ratio, bold=is_dominant_bold)
                heading_spans: list[Span] = []
                for i, line in enumerate(lines):
                    heading_spans.extend(line)
                    if i < len(lines) - 1:
                        heading_spans.append(
                            Span(
                                text=" ",
                                bbox=line[-1].bbox,
                                font_size=line[-1].font_size,
                            )
                        )
                page.blocks.append(
                    HeadingBlock(
                        page_idx=page_idx,
                        bbox=bbox,
                        provenance=prov,
                        level=level,
                        spans=heading_spans,
                    )
                )
                return

        # List detection: every line starts with a bullet or numbered marker.
        list_block = self._try_build_list(lines, bbox, prov, page_idx)
        if list_block is not None:
            page.blocks.append(list_block)
            return

        # Default: a TextBlock containing all spans, with line breaks preserved
        # as literal "\n" spans for the renderer to honor.
        flat: list[Span] = []
        for i, line in enumerate(lines):
            flat.extend(line)
            if i < len(lines) - 1:
                # Encode a soft line break as a synthetic span whose text is "\n".
                flat.append(
                    Span(
                        text="\n",
                        bbox=line[-1].bbox,
                        font="",
                        font_size=line[-1].font_size,
                    )
                )

        page.blocks.append(
            TextBlock(
                page_idx=page_idx,
                bbox=bbox,
                provenance=prov,
                spans=flat,
            )
        )

    @staticmethod
    def _try_build_list(
        lines: list[list[Span]],
        bbox: BBox,
        prov: Provenance,
        page_idx: int,
    ) -> ListBlock | None:
        items: list[ListItem] = []
        ordered: bool | None = None  # None until we see at least one marker

        for line in lines:
            joined = "".join(s.text for s in line)
            num_m = _NUMBERED_RE.match(joined)
            bul_m = _BULLET_RE.match(joined)

            if num_m:
                if ordered is False:
                    return None                  # mixed: reject
                ordered = True
                raw = num_m.group("paren") or num_m.group("plain")
                ordinal = _parse_ordinal(raw)
                items.append(
                    ListItem(
                        spans=_strip_marker_from_spans(line, num_m.end()),
                        ordinal=ordinal,
                    )
                )
            elif bul_m:
                if ordered is True:
                    return None
                ordered = False
                items.append(
                    ListItem(
                        spans=_strip_marker_from_spans(line, bul_m.end()),
                        ordinal=None,
                    )
                )
            else:
                # Not every line has a marker => not a clean list block.
                return None

        if not items or ordered is None:
            return None

        return ListBlock(
            page_idx=page_idx,
            bbox=bbox,
            provenance=prov,
            ordered=ordered,
            items=items,
        )


def _parse_ordinal(raw: str) -> int | None:
    if raw.isdigit():
        return int(raw)
    return None  # alpha / roman ordinals get re-numbered by the renderer


def _heading_level_from_ratio(ratio: float, *, bold: bool = False) -> int:
    """Map (font_size / body_size) to markdown heading levels 1..6.

    Ratios are calibrated against typical document scales (body 10-12 pt,
    title 20-28, H1 16-18, H2 14-15, H3 12-13, H4-H5 ~11.5).
    """
    if ratio >= 1.80:
        return 1
    if ratio >= 1.50:
        return 2
    if ratio >= 1.30:
        return 3
    if ratio >= 1.18:
        return 4
    if ratio >= 1.10:
        return 5
    # Bold but barely larger than body still warrants the lowest heading level.
    return 6


def _strip_marker_from_spans(spans: list[Span], cut_chars: int) -> list[Span]:
    """Drop the first ``cut_chars`` characters across the line's spans."""
    remaining = cut_chars
    out: list[Span] = []
    for s in spans:
        if remaining >= len(s.text):
            remaining -= len(s.text)
            continue
        if remaining > 0:
            out.append(s.model_copy(update={"text": s.text[remaining:]}))
            remaining = 0
        else:
            out.append(s)
    return out


def _match_link(bbox, links: list[dict]):
    """Return a Link object if any registered URI link intersects ``bbox``."""
    if bbox is None or not links:
        return None
    try:
        import fitz  # type: ignore

        rect = fitz.Rect(bbox)
    except Exception:
        return None
    for link in links:
        if rect.intersects(link["rect"]):
            from pdf_converter.ir.span import BBox as IRBBox
            from pdf_converter.ir.span import Link  # local import to avoid cycle

            r = link["rect"]
            return Link(
                uri=link["uri"],
                rect=IRBBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1),
            )
    return None


__all__ = ["TextExtractor", "is_monospace_font"]
