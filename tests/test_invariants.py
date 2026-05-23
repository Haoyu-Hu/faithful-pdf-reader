"""Structural invariants enforced across all conversions."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_converter.api import convert_pdf
from pdf_converter.config import Config
from pdf_converter.ir.blocks import BlockKind


@pytest.fixture
def doc(sample_pdf: Path, tmp_path: Path):
    out = tmp_path / "out"
    return convert_pdf(sample_pdf, out_dir=out, config=Config.default(), write=False)


def test_reading_order_is_permutation_per_page(doc) -> None:
    for page in doc.pages:
        if not page.blocks:
            continue
        ords = sorted(b.reading_order_idx for b in page.blocks)
        # No duplicates and contiguous from 0.
        assert ords == list(range(len(ords))), (page.index, ords)


def test_every_block_has_provenance(doc) -> None:
    for page in doc.pages:
        for b in page.blocks:
            assert b.provenance.extractor != ""


def test_heading_levels_in_range(doc) -> None:
    for b in doc.iter_blocks():
        if b.kind is BlockKind.HEADING:
            assert 1 <= b.level <= 6   # type: ignore[attr-defined]


def test_table_rows_have_equal_columns_after_render(doc) -> None:
    # If any TableBlock is in the doc, each row in its cells must have the
    # same column count after rendering (the renderer pads short rows).
    from pdf_converter.render.tables import render_table

    for b in doc.iter_blocks():
        if b.kind is BlockKind.TABLE:
            md = render_table(b.cells, header_rows=b.header_rows)  # type: ignore[attr-defined]
            rows = [line for line in md.splitlines() if line.startswith("|")]
            col_counts = {r.count("|") for r in rows}
            assert len(col_counts) == 1, (b.cells, col_counts)


def test_no_chrome_text_in_output(doc) -> None:
    """Detected chrome strings should not appear as standalone blocks."""
    for b in doc.iter_blocks():
        if b.kind is BlockKind.TEXT:
            text = "".join(s.text for s in b.spans).strip().lower()   # type: ignore[attr-defined]
            assert text != "" and text not in doc.chrome_texts


def test_markdown_nonempty_for_sample(doc, sample_pdf) -> None:
    md = doc.markdown
    assert md.strip() != ""
    assert "Test Document" in md
    assert "Introduction" in md
    assert "Methods" in md
    # Page numbers should be stripped.
    assert "\n1\n" not in md
    assert "\n2\n" not in md
