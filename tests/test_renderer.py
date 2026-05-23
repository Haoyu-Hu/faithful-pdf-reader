"""Markdown renderer exhaustiveness and per-block-type output checks."""

from __future__ import annotations

from pdf_converter.ir import (
    BBox,
    CodeBlock,
    Document,
    EquationBlock,
    FigureBlock,
    FootnoteBlock,
    HeadingBlock,
    ListBlock,
    ListItem,
    Page,
    Provenance,
    Span,
    TextBlock,
)
from pdf_converter.render import render_document
from pdf_converter.render.tables import render_table


def _bbox() -> BBox:
    return BBox(x0=0, y0=0, x1=10, y1=10)


def _prov() -> Provenance:
    return Provenance(extractor="test")


def _doc_with(blocks: list) -> Document:
    doc = Document(source="/tmp/x.pdf")
    page = Page(index=0, width=100, height=100)
    for i, b in enumerate(blocks):
        b.reading_order_idx = i
        page.blocks.append(b)
    doc.pages.append(page)
    return doc


def test_render_heading_h1_to_h6() -> None:
    for level in range(1, 7):
        doc = _doc_with(
            [
                HeadingBlock(
                    page_idx=0,
                    bbox=_bbox(),
                    provenance=_prov(),
                    level=level,
                    spans=[Span(text=f"H{level}", bbox=_bbox(), font_size=10.0)],
                ),
            ]
        )
        md = render_document(doc, format_with_mdformat=False)
        assert md.startswith("#" * level + " ")


def test_render_paragraph() -> None:
    doc = _doc_with(
        [
            TextBlock(
                page_idx=0,
                bbox=_bbox(),
                provenance=_prov(),
                spans=[
                    Span(text="hello ", bbox=_bbox(), font_size=10.0),
                    Span(text="world", bbox=_bbox(), font_size=10.0, is_bold=True),
                ],
            )
        ]
    )
    md = render_document(doc, format_with_mdformat=False)
    assert "hello" in md and "world" in md
    assert "**world**" in md


def test_render_italic_and_bold_italic() -> None:
    doc = _doc_with(
        [
            TextBlock(
                page_idx=0,
                bbox=_bbox(),
                provenance=_prov(),
                spans=[
                    Span(text="i", bbox=_bbox(), font_size=10.0, is_italic=True),
                    Span(
                        text="bi",
                        bbox=_bbox(),
                        font_size=10.0,
                        is_bold=True,
                        is_italic=True,
                    ),
                ],
            )
        ]
    )
    md = render_document(doc, format_with_mdformat=False)
    assert "*i*" in md
    assert "***bi***" in md


def test_render_list_unordered() -> None:
    doc = _doc_with(
        [
            ListBlock(
                page_idx=0,
                bbox=_bbox(),
                provenance=_prov(),
                ordered=False,
                items=[
                    ListItem(spans=[Span(text="alpha", bbox=_bbox(), font_size=10.0)]),
                    ListItem(spans=[Span(text="beta", bbox=_bbox(), font_size=10.0)]),
                ],
            )
        ]
    )
    md = render_document(doc, format_with_mdformat=False)
    assert "- alpha" in md
    assert "- beta" in md


def test_render_list_ordered() -> None:
    doc = _doc_with(
        [
            ListBlock(
                page_idx=0,
                bbox=_bbox(),
                provenance=_prov(),
                ordered=True,
                items=[
                    ListItem(spans=[Span(text="one", bbox=_bbox(), font_size=10.0)], ordinal=1),
                    ListItem(spans=[Span(text="two", bbox=_bbox(), font_size=10.0)], ordinal=2),
                ],
            )
        ]
    )
    md = render_document(doc, format_with_mdformat=False)
    assert "1. one" in md
    assert "2. two" in md


def test_render_table_markdown() -> None:
    md = render_table(
        [["A", "B"], ["1", "2"]], header_rows=1, alignments=["left", "right"]
    )
    # one header row, one separator, one body row
    lines = [line for line in md.splitlines() if line.strip()]
    assert len(lines) == 3
    assert lines[1].endswith(":|") or "-:" in lines[1]   # right alignment marker


def test_render_table_escapes_pipes() -> None:
    md = render_table([["a|b", "c"], ["1", "2"]])
    assert "a\\|b" in md


def test_render_equation_display_vs_inline() -> None:
    doc = _doc_with(
        [
            EquationBlock(
                page_idx=0,
                bbox=_bbox(),
                provenance=_prov(),
                latex="x = y",
                display=True,
            ),
            EquationBlock(
                page_idx=0,
                bbox=_bbox(),
                provenance=_prov(),
                latex="z",
                display=False,
            ),
        ]
    )
    md = render_document(doc, format_with_mdformat=False)
    assert "$$\nx = y\n$$" in md
    assert "$z$" in md


def test_render_code_with_language_fence() -> None:
    doc = _doc_with(
        [
            CodeBlock(
                page_idx=0,
                bbox=_bbox(),
                provenance=_prov(),
                lines=["print('hi')"],
                language="python",
            )
        ]
    )
    md = render_document(doc, format_with_mdformat=False)
    assert "```python" in md
    assert "print('hi')" in md
    assert md.count("```") == 2


def test_render_figure_with_caption() -> None:
    doc = _doc_with(
        [
            FigureBlock(
                page_idx=0,
                bbox=_bbox(),
                provenance=_prov(),
                image_path="images/x.png",
                caption="Figure 1: hello.",
            )
        ]
    )
    md = render_document(doc, format_with_mdformat=False)
    # Short alt label captured from "Figure 1: ..." prefix.
    assert "![Figure 1](images/x.png)" in md
    assert "*Figure 1: hello.*" in md


def test_render_footnote_appended_at_end() -> None:
    doc = _doc_with(
        [
            TextBlock(
                page_idx=0,
                bbox=_bbox(),
                provenance=_prov(),
                spans=[Span(text="body", bbox=_bbox(), font_size=10.0)],
            ),
            FootnoteBlock(
                page_idx=0,
                bbox=_bbox(),
                provenance=_prov(),
                marker="1",
                spans=[Span(text="footnote body", bbox=_bbox(), font_size=10.0)],
            ),
        ]
    )
    md = render_document(doc, format_with_mdformat=False)
    assert "footnote body" in md
    assert "[^1]: footnote body" in md
    # body comes before footnotes section
    assert md.index("body") < md.index("Footnotes")
