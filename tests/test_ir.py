"""IR construction & validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pdf_converter.ir import (
    BBox,
    CodeBlock,
    Document,
    EquationBlock,
    FigureBlock,
    HeadingBlock,
    ListBlock,
    ListItem,
    Page,
    Provenance,
    Span,
    TableBlock,
    TextBlock,
)


def test_bbox_validates_x_order() -> None:
    with pytest.raises(ValidationError):
        BBox(x0=10, y0=0, x1=5, y1=20)


def test_bbox_validates_y_order() -> None:
    with pytest.raises(ValidationError):
        BBox(x0=0, y0=20, x1=10, y1=5)


def test_bbox_geometry() -> None:
    bb = BBox(x0=0, y0=0, x1=10, y1=20)
    assert bb.width == 10
    assert bb.height == 20
    assert bb.area == 200
    assert bb.cx == 5
    assert bb.cy == 10


def test_bbox_union_and_intersect() -> None:
    a = BBox(x0=0, y0=0, x1=10, y1=10)
    b = BBox(x0=5, y0=5, x1=15, y1=15)
    assert a.intersects(b)
    u = a.union(b)
    assert u == BBox(x0=0, y0=0, x1=15, y1=15)


def test_span_from_pymupdf_flags() -> None:
    bold = 1 << 4
    italic = 1 << 1
    mono = 1 << 3
    s = Span.from_pymupdf(
        text="x",
        bbox=(0, 0, 1, 1),
        font="Helv-Bold",
        font_size=12.0,
        flags=bold | italic | mono,
    )
    assert s.is_bold and s.is_italic and s.is_monospace


def _bbox() -> BBox:
    return BBox(x0=0, y0=0, x1=10, y1=10)


def _prov() -> Provenance:
    return Provenance(extractor="test")


def test_all_block_kinds_constructible() -> None:
    blocks = [
        TextBlock(page_idx=0, bbox=_bbox(), provenance=_prov(), spans=[]),
        HeadingBlock(page_idx=0, bbox=_bbox(), provenance=_prov(), level=2, spans=[]),
        ListBlock(
            page_idx=0,
            bbox=_bbox(),
            provenance=_prov(),
            ordered=True,
            items=[ListItem(spans=[], ordinal=1)],
        ),
        TableBlock(
            page_idx=0,
            bbox=_bbox(),
            provenance=_prov(),
            cells=[["a", "b"], ["c", "d"]],
        ),
        FigureBlock(
            page_idx=0,
            bbox=_bbox(),
            provenance=_prov(),
            image_path="fig.png",
        ),
        EquationBlock(
            page_idx=0,
            bbox=_bbox(),
            provenance=_prov(),
            latex="x = y",
        ),
        CodeBlock(
            page_idx=0,
            bbox=_bbox(),
            provenance=_prov(),
            lines=["print(1)"],
        ),
    ]
    kinds = [b.kind.value for b in blocks]
    assert set(kinds) == {"text", "heading", "list", "table", "figure", "equation", "code"}


def test_heading_level_bounds() -> None:
    with pytest.raises(ValidationError):
        HeadingBlock(page_idx=0, bbox=_bbox(), provenance=_prov(), level=7)
    with pytest.raises(ValidationError):
        HeadingBlock(page_idx=0, bbox=_bbox(), provenance=_prov(), level=0)


def test_document_iter_blocks_in_reading_order() -> None:
    doc = Document(source="/tmp/x.pdf")
    p = Page(index=0, width=100, height=100)
    doc.pages.append(p)
    p.blocks.append(
        TextBlock(page_idx=0, bbox=_bbox(), provenance=_prov(), reading_order_idx=2)
    )
    p.blocks.append(
        TextBlock(page_idx=0, bbox=_bbox(), provenance=_prov(), reading_order_idx=0)
    )
    p.blocks.append(
        TextBlock(page_idx=0, bbox=_bbox(), provenance=_prov(), reading_order_idx=1)
    )
    ords = [b.reading_order_idx for b in doc.iter_blocks()]
    assert ords == [0, 1, 2]


def test_table_dimensions() -> None:
    t = TableBlock(
        page_idx=0,
        bbox=_bbox(),
        provenance=_prov(),
        cells=[["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"]],
    )
    assert t.n_rows == 3
    assert t.n_cols == 3
