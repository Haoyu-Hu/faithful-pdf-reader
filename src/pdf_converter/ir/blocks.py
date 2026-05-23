"""Tagged-union block types making up the IR.

Each block carries the bounding box, page index, reading-order index, and
provenance (which extractor produced it). Renderers exhaustively switch on
``Block.kind`` (a Literal-typed discriminator).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from pdf_converter.ir.span import BBox, Span


class BlockKind(StrEnum):
    TEXT = "text"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    CODE = "code"
    FOOTNOTE = "footnote"


class Provenance(BaseModel):
    """How a block was produced. Set by every extractor."""

    model_config = ConfigDict(frozen=True)

    extractor: str                          # e.g. "heuristic.text", "ml.surya_layout"
    confidence: float = 1.0                 # in [0, 1]; heuristic defaults to 1.0
    notes: str = ""                         # free-form for debugging


class _BlockBase(BaseModel):
    """Common fields. Block subclasses set their own ``kind`` literal."""

    model_config = ConfigDict(frozen=False)  # mutable so post-processors can patch order

    page_idx: int = Field(..., ge=0)
    bbox: BBox
    reading_order_idx: int = Field(default=-1, ge=-1)
    provenance: Provenance


# ----- concrete block types ------------------------------------------------


class TextBlock(_BlockBase):
    """A paragraph of body text."""

    kind: Literal[BlockKind.TEXT] = BlockKind.TEXT
    spans: list[Span] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)


class HeadingBlock(_BlockBase):
    """A heading. ``level`` is 1..6 (markdown semantics)."""

    kind: Literal[BlockKind.HEADING] = BlockKind.HEADING
    level: int = Field(..., ge=1, le=6)
    spans: list[Span] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)


class ListItem(BaseModel):
    """A single item in a list. ``ordinal`` is None for unordered lists."""

    model_config = ConfigDict(frozen=False)

    spans: list[Span] = Field(default_factory=list)
    ordinal: int | None = None
    nested: list[ListItem] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)


class ListBlock(_BlockBase):
    """A bullet or ordered list. ``ordered`` reflects the outermost list type."""

    kind: Literal[BlockKind.LIST] = BlockKind.LIST
    ordered: bool = False
    items: list[ListItem] = Field(default_factory=list)


class TableBlock(_BlockBase):
    """A table extracted from the PDF.

    ``cells`` is a 2-D matrix indexed [row][col]. ``header_rows`` and
    ``header_cols`` describe header bands (defaults: row 0 is header, no header
    cols).
    """

    kind: Literal[BlockKind.TABLE] = BlockKind.TABLE
    cells: list[list[str]] = Field(default_factory=list)
    header_rows: int = 1
    header_cols: int = 0
    caption: str = ""
    alignments: list[Literal["left", "right", "center"]] = Field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.cells)

    @property
    def n_cols(self) -> int:
        return max((len(row) for row in self.cells), default=0)


class FigureBlock(_BlockBase):
    """An image / figure extracted from the PDF.

    ``image_path`` is a path relative to the markdown output directory.
    """

    kind: Literal[BlockKind.FIGURE] = BlockKind.FIGURE
    image_path: str
    caption: str = ""
    alt_text: str = ""


class EquationBlock(_BlockBase):
    """A math equation block. ``latex`` may be empty if heuristic fallback only."""

    kind: Literal[BlockKind.EQUATION] = BlockKind.EQUATION
    latex: str = ""
    display: bool = True               # True = display math ($$...$$), False = inline ($...$)
    fallback_text: str = ""            # raw Unicode text when LaTeX is unknown


class CodeBlock(_BlockBase):
    """A monospace code block."""

    kind: Literal[BlockKind.CODE] = BlockKind.CODE
    lines: list[str] = Field(default_factory=list)
    language: str = ""                 # empty => no language tag in fence

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class FootnoteBlock(_BlockBase):
    """A footnote referenced from body text."""

    kind: Literal[BlockKind.FOOTNOTE] = BlockKind.FOOTNOTE
    marker: str                        # e.g. "1", "*", "a"
    spans: list[Span] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)


# Discriminated union: ``Block`` is one of the concrete types above.
Block = Annotated[
    TextBlock | HeadingBlock | ListBlock | TableBlock | FigureBlock | EquationBlock | CodeBlock | FootnoteBlock,
    Field(discriminator="kind"),
]
