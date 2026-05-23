"""Intermediate representation: typed document model populated by extractors."""

from pdf_converter.ir.blocks import (
    Block,
    BlockKind,
    CodeBlock,
    EquationBlock,
    FigureBlock,
    FootnoteBlock,
    HeadingBlock,
    ListBlock,
    ListItem,
    Provenance,
    TableBlock,
    TextBlock,
)
from pdf_converter.ir.document import Document, Page
from pdf_converter.ir.span import BBox, Link, Span

__all__ = [
    "BBox",
    "Block",
    "BlockKind",
    "CodeBlock",
    "Document",
    "EquationBlock",
    "FigureBlock",
    "FootnoteBlock",
    "HeadingBlock",
    "Link",
    "ListBlock",
    "ListItem",
    "Page",
    "Provenance",
    "Span",
    "TableBlock",
    "TextBlock",
]
