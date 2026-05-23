"""Surya layout refinement.

For each page we run the Surya layout predictor and use the results to:

1. **Reclassify** heuristic blocks whose surface kind disagrees with the
   layout label. The most useful upgrade is TextBlock -> EquationBlock,
   which lets the math OCR pass produce LaTeX. Headings detected by the
   layout model also upgrade non-heading text whose font heuristic missed
   the size signal.
2. **Override reading order**. Surya's layout `position` field is a
   reading-order index across all blocks of the page; we use it directly,
   replacing the column-based heuristic.
3. Optionally **inject** EquationBlocks for layout regions classified as
   "Formula" that have no overlapping heuristic block.

We deliberately don't try to *replace* the heuristic blocks wholesale --
the heuristic side has the actual text content (with font/style info) that
Surya can't recover from a rendered page image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pdf_converter.config import Config
from pdf_converter.ir import (
    BBox,
    Document,
    EquationBlock,
    HeadingBlock,
    Provenance,
    TextBlock,
)
from pdf_converter.ir.blocks import BlockKind
from pdf_converter.utils.logging import get_logger

if TYPE_CHECKING:
    pass

_logger = get_logger("extractors.ml.surya_layout")

# Surya layout labels (observed): Picture, SectionHeader, Text, Footnote,
# Caption, ListItem, Table, Formula, Code, PageHeader, PageFooter, Title.
_LABEL_TO_KIND = {
    "Text": BlockKind.TEXT,
    "Caption": BlockKind.TEXT,
    "ListItem": BlockKind.LIST,
    "Title": BlockKind.HEADING,
    "SectionHeader": BlockKind.HEADING,
    "PageHeader": BlockKind.TEXT,                    # chrome detector handles removal
    "PageFooter": BlockKind.TEXT,
    "Footnote": BlockKind.FOOTNOTE,
    "Picture": BlockKind.FIGURE,
    "Figure": BlockKind.FIGURE,
    "Table": BlockKind.TABLE,
    "Formula": BlockKind.EQUATION,
    "Code": BlockKind.CODE,
}


class SuryaLayout:
    name = "ml.surya_layout"

    def __init__(self, *, scale: float = 2.0, min_overlap: float = 0.30) -> None:
        self.scale = scale
        self.min_overlap = min_overlap

    def __call__(self, doc: Document, *, fitz_doc, config: Config) -> None:
        from pdf_converter.extractors.ml.loader import get_layout_predictor
        from pdf_converter.extractors.ml.page_render import render_page

        predictor = get_layout_predictor()
        if predictor is None:
            return

        # Render all pages once and run inference in a single batch.
        images = []
        for page_idx in range(doc.n_pages):
            try:
                fpage = fitz_doc[page_idx]
            except IndexError:
                _logger.warning("page %d missing in fitz doc", page_idx)
                continue
            images.append(render_page(fpage, scale=self.scale))

        if not images:
            return

        results = predictor(images)                  # type: ignore[operator]

        for page_idx, layout in enumerate(results):
            if page_idx >= doc.n_pages:
                break
            self._apply_layout(doc, page_idx, layout)

    def _apply_layout(self, doc: Document, page_idx: int, layout) -> None:
        page = doc.pages[page_idx]
        prov = Provenance(extractor=self.name)

        # Build PDF-space bboxes from Surya polygons (which are in pixel space).
        # We divide back by `scale` to compare against PDF-space heuristic bboxes.
        layout_items: list[tuple[BBox, str, int, float]] = []
        for lb in layout.bboxes:
            try:
                xs = [pt[0] for pt in lb.polygon]
                ys = [pt[1] for pt in lb.polygon]
            except (TypeError, IndexError):
                continue
            bbox = BBox(
                x0=min(xs) / self.scale,
                y0=min(ys) / self.scale,
                x1=max(xs) / self.scale,
                y1=max(ys) / self.scale,
            )
            layout_items.append((bbox, lb.label, lb.position, float(lb.confidence)))

        # FIRST PASS: for every Formula region from Surya, collapse all
        # contained text/equation fragments into a single new EquationBlock.
        # This is the only reliable way to clean up the fragmented spans
        # PyMuPDF returns inside equations.
        #
        # Surya's Formula bbox is often tight on the central symbols and
        # excludes scattered LaTeX-glyph fragments around it (sums, integrals,
        # script bounds rendered as separate text spans). We expand the bbox
        # before cleanup/OCR so those fragments are absorbed.
        for bbox, label, _position, confidence in layout_items:
            if label != "Formula" or confidence < 0.5:
                continue
            expanded = _expand_for_formula(bbox, page.width, page.height)
            insert_order = _nearest_order(page, expanded)
            fallback = _drop_contained_text_blocks(page, expanded)
            page.blocks.append(
                EquationBlock(
                    page_idx=page_idx,
                    bbox=expanded,
                    reading_order_idx=insert_order,
                    provenance=prov,
                    latex="",
                    display=True,
                    fallback_text=fallback,
                )
            )

        # SECOND PASS: heading / etc. refinement for non-formula labels.
        for bbox, label, _position, confidence in layout_items:
            if label == "Formula":
                continue
            best_block = None
            best_overlap = 0.0
            for b in page.blocks:
                ov = _overlap_ratio(b.bbox, bbox)
                if ov > best_overlap:
                    best_overlap = ov
                    best_block = b
            if best_block is None or best_overlap < self.min_overlap:
                # Other unseen labels (loose decorations) are ignored to avoid
                # duplicating content the heuristic captured under a wider bbox.
                continue

            # Optional kind upgrade when label disagrees usefully (non-formula).
            target_kind = _LABEL_TO_KIND.get(label)
            if target_kind is None:
                continue

            if (
                target_kind is BlockKind.HEADING
                and best_block.kind is BlockKind.TEXT
                and confidence >= 0.7
            ):
                _upgrade_text_to_heading(page, best_block, prov)   # type: ignore[arg-type]


_CONTAINMENT_THRESHOLD = 0.55

# Surya's Formula bboxes are usually a tight box around the central glyphs
# and miss scattered LaTeX-glyph fragments. Expand vertically modestly and
# horizontally to (roughly) the surrounding column.
_FORMULA_MARGIN_X = 250.0    # PDF points -- enough to span a typical column
_FORMULA_MARGIN_Y = 18.0


def _expand_for_formula(bbox: BBox, page_w: float, page_h: float) -> BBox:
    """Return a bbox expanded by formula margins, clipped to the page.

    Horizontal expansion is anchored to the column containing the formula:
    we extend leftwards no further than the column's left margin and
    rightwards no further than its right margin. For a 2-column page,
    the formula stays within one column; for single-column, it can grow
    across the full text width.
    """
    half = page_w / 2.0
    if bbox.cx < half:
        # left column
        left_bound = 0.0
        right_bound = half + 20.0
    else:
        left_bound = half - 20.0
        right_bound = page_w
    return BBox(
        x0=max(left_bound, bbox.x0 - _FORMULA_MARGIN_X),
        y0=max(0.0, bbox.y0 - _FORMULA_MARGIN_Y),
        x1=min(right_bound, bbox.x1 + _FORMULA_MARGIN_X),
        y1=min(page_h, bbox.y1 + _FORMULA_MARGIN_Y),
    )


def _drop_contained_text_blocks(page, region: BBox) -> str:
    """Remove TextBlocks/EquationBlocks heavily contained in ``region``.

    Returns the concatenated text of the removed blocks for use as
    ``EquationBlock.fallback_text`` when LaTeX OCR fails.
    """
    kept = []
    removed_texts: list[str] = []
    for b in page.blocks:
        if (
            b.kind in (BlockKind.TEXT, BlockKind.EQUATION)
            and _contained_ratio(b.bbox, region) >= _CONTAINMENT_THRESHOLD
        ):
            if b.kind is BlockKind.TEXT:
                removed_texts.append(
                    "".join(s.text for s in b.spans)             # type: ignore[attr-defined]
                )
            elif b.fallback_text:                                # type: ignore[attr-defined]
                removed_texts.append(b.fallback_text)            # type: ignore[attr-defined]
            continue
        kept.append(b)
    page.blocks = kept
    return " ".join(t.strip() for t in removed_texts if t.strip())[:300]


def _contained_ratio(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner`` that lies inside ``outer``."""
    if not inner.intersects(outer):
        return 0.0
    ix0 = max(inner.x0, outer.x0)
    iy0 = max(inner.y0, outer.y0)
    ix1 = min(inner.x1, outer.x1)
    iy1 = min(inner.y1, outer.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    return inter / max(1e-6, inner.area)


def _nearest_order(page, bbox: BBox) -> int:
    """Pick a reading-order index near the heuristic block closest in Y."""
    nearest = None
    best_dy = float("inf")
    for b in page.blocks:
        dy = abs(b.bbox.cy - bbox.cy)
        if dy < best_dy and b.reading_order_idx >= 0:
            best_dy = dy
            nearest = b
    return nearest.reading_order_idx if nearest is not None else 0


def _overlap_ratio(a: BBox, b: BBox) -> float:
    """Intersection-over-min-area. Returns 0 when disjoint."""
    if not a.intersects(b):
        return 0.0
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    min_area = max(1e-6, min(a.area, b.area))
    return inter / min_area


def _upgrade_text_to_heading(page, block: TextBlock, prov: Provenance) -> None:
    """Promote a TextBlock to a HeadingBlock (level 3 default)."""
    level = 3
    # Refine using font size if a span is taller than body.
    if block.spans and block.spans[0].font_size > 14:
        level = 2
    if block.spans and block.spans[0].font_size > 18:
        level = 1
    new_block = HeadingBlock(
        page_idx=block.page_idx,
        bbox=block.bbox,
        reading_order_idx=block.reading_order_idx,
        provenance=prov,
        level=level,
        spans=block.spans,
    )
    _swap_block(page, block, new_block)


def _swap_block(page, old, new) -> None:
    for i, b in enumerate(page.blocks):
        if b is old:
            page.blocks[i] = new
            return


__all__ = ["SuryaLayout"]
