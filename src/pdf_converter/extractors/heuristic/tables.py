"""Table extractor using pdfplumber.

Tables are inserted into the IR with their actual bounding box, which lets
``LayoutOrderer`` position them correctly within the reading order. This
fixes the reference's bug of appending tables at the end of every page.

A safety cap on tables per page is kept (was 128 in the reference) -- 64
here, more than enough for any sane document and a defense against
adversarial PDFs that nominate every paragraph as a table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pdf_converter.config import Config
from pdf_converter.ir import BBox, Document, Provenance, TableBlock
from pdf_converter.utils.logging import get_logger

if TYPE_CHECKING:
    pass

_logger = get_logger("extractors.heuristic.tables")
_MAX_TABLES_PER_PAGE = 64


class TableExtractor:
    name = "heuristic.tables"

    def __call__(self, doc: Document, *, fitz_doc, config: Config) -> None:
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            _logger.warning("pdfplumber not installed; skipping table extraction")
            return

        prov = Provenance(extractor=self.name)

        try:
            with pdfplumber.open(doc.source) as pdf:
                for page_idx, ppage in enumerate(pdf.pages):
                    if page_idx >= doc.n_pages:
                        break
                    try:
                        tables = ppage.find_tables() or []
                    except Exception as e:
                        _logger.debug("pdfplumber find_tables failed on page %d: %s", page_idx, e)
                        continue

                    if len(tables) > _MAX_TABLES_PER_PAGE:
                        _logger.warning(
                            "page %d has %d candidate tables; capping at %d",
                            page_idx,
                            len(tables),
                            _MAX_TABLES_PER_PAGE,
                        )
                        tables = tables[:_MAX_TABLES_PER_PAGE]

                    for t in tables:
                        cells = t.extract()
                        if not cells:
                            continue
                        cells = [
                            ["" if c is None else str(c) for c in row] for row in cells
                        ]
                        if not _is_meaningful_table(cells):
                            continue
                        bbox = BBox(
                            x0=float(t.bbox[0]),
                            y0=float(t.bbox[1]),
                            x1=float(t.bbox[2]),
                            y1=float(t.bbox[3]),
                        )
                        # Drop any TextBlocks that are wholly contained in this table's bbox --
                        # the table content has now been captured structurally.
                        _drop_text_inside(doc.pages[page_idx], bbox)

                        doc.pages[page_idx].blocks.append(
                            TableBlock(
                                page_idx=page_idx,
                                bbox=bbox,
                                provenance=prov,
                                cells=cells,
                                header_rows=1 if len(cells) > 1 else 0,
                            )
                        )
        except Exception as e:
            _logger.error("table extraction failed: %s", e)


def _is_meaningful_table(cells: list[list[str]]) -> bool:
    """Reject pdfplumber 'tables' that are really labels or empty grids.

    Heuristics:
    * Needs at least 2 rows OR at least 2 columns and 2 non-empty cells.
    * Total non-empty cells must be >= 4 (drops single-row/column captions).
    * If only one row, treat as misclassified text and reject.
    """
    n_rows = len(cells)
    if n_rows == 0:
        return False
    n_cols = max(len(r) for r in cells)
    non_empty = sum(1 for r in cells for c in r if c.strip())
    if non_empty < 4:
        return False
    return not (n_rows == 1 or n_cols == 1)


def _drop_text_inside(page, table_bbox: BBox) -> None:
    """Remove text/heading/list blocks entirely contained by ``table_bbox``."""
    from pdf_converter.ir.blocks import BlockKind

    keep = []
    for b in page.blocks:
        if b.kind in (BlockKind.TEXT, BlockKind.HEADING, BlockKind.LIST):
            bb = b.bbox
            if (
                bb.x0 >= table_bbox.x0 - 1
                and bb.y0 >= table_bbox.y0 - 1
                and bb.x1 <= table_bbox.x1 + 1
                and bb.y1 <= table_bbox.y1 + 1
            ):
                continue
        keep.append(b)
    page.blocks = keep


__all__ = ["TableExtractor"]
