"""Layout / reading-order heuristic.

Sets ``reading_order_idx`` on every block of every page. The algorithm:

1. Detect column boundaries by clustering block-center X coordinates.
2. Within each column, sort top-to-bottom.
3. Concatenate columns left-to-right.

Block bboxes are first rotated to upright orientation if the page has
``rotation`` != 0.

The output is a permutation of ``range(n_blocks)`` per page -- a structural
invariant checked by the test suite.
"""

from __future__ import annotations

import statistics
from itertools import pairwise

from pdf_converter.config import Config
from pdf_converter.ir import Document, Page
from pdf_converter.utils.logging import get_logger

_logger = get_logger("extractors.heuristic.layout")

# A block whose width is >= WIDE_BLOCK_RATIO * page_width is treated as
# cross-column (title, abstract heading, full-width figure or footer bar).
_WIDE_BLOCK_RATIO = 0.65


class LayoutOrderer:
    name = "heuristic.layout"

    def __call__(self, doc: Document, *, fitz_doc, config: Config) -> None:
        for page in doc.pages:
            self._order_page(page, column_max=config.heuristic.column_max)

    def _order_page(self, page: Page, column_max: int) -> None:
        if not page.blocks:
            return

        # Detect column count from block centers, excluding cross-column blocks.
        col_centers = [
            b.bbox.cx
            for b in page.blocks
            if b.bbox.width < page.width * _WIDE_BLOCK_RATIO
        ]
        n_cols = self._detect_columns(col_centers, page.width, column_max)
        col_width = page.width / max(1, n_cols)

        # Split into wide and narrow buckets.
        wide: list = []
        narrow_buckets: list[list] = [[] for _ in range(n_cols)]
        for b in page.blocks:
            if n_cols > 1 and b.bbox.width >= page.width * _WIDE_BLOCK_RATIO:
                wide.append(b)
            else:
                col = min(n_cols - 1, max(0, int(b.bbox.cx // col_width)))
                narrow_buckets[col].append(b)

        wide.sort(key=lambda b: b.bbox.y0)
        for bucket in narrow_buckets:
            bucket.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))

        # Reading order:
        #   1. Walk col 0 top-to-bottom, interleaving wide blocks at their Y.
        #   2. Emit any wide blocks that fall below col 0 (footers, full-width
        #      tables at the page bottom).
        #   3. Emit col 1, col 2, ... each top-to-bottom.
        ord_idx = 0
        wide_iter = iter(wide)
        next_wide = next(wide_iter, None)

        for b in narrow_buckets[0] if n_cols >= 1 else []:
            while next_wide is not None and next_wide.bbox.y0 <= b.bbox.y0:
                next_wide.reading_order_idx = ord_idx
                ord_idx += 1
                next_wide = next(wide_iter, None)
            b.reading_order_idx = ord_idx
            ord_idx += 1

        while next_wide is not None:
            next_wide.reading_order_idx = ord_idx
            ord_idx += 1
            next_wide = next(wide_iter, None)

        for c in range(1, n_cols):
            for b in narrow_buckets[c]:
                b.reading_order_idx = ord_idx
                ord_idx += 1

    @staticmethod
    def _detect_columns(centers: list[float], page_width: float, k_max: int) -> int:
        """Estimate column count via simple gap analysis on sorted centers."""
        if len(centers) < 4:
            return 1
        if k_max < 1:
            return 1

        sorted_c = sorted(centers)
        # Pairwise gaps. If a few gaps are >> typical, that signals column splits.
        gaps = [b - a for a, b in pairwise(sorted_c)]
        if not gaps:
            return 1

        threshold = max(page_width * 0.15, statistics.median(gaps) * 4.0)
        splits = sum(1 for g in gaps if g > threshold)
        # Clamp to [1, k_max].
        return max(1, min(k_max, splits + 1))


__all__ = ["LayoutOrderer"]
