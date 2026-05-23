"""Top-level IR containers: Document and Page."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pdf_converter.ir.blocks import Block


class Page(BaseModel):
    """A single PDF page."""

    model_config = ConfigDict(frozen=False)

    index: int = Field(..., ge=0)            # 0-based
    width: float
    height: float
    rotation: int = 0                        # 0, 90, 180, 270
    blocks: list[Block] = Field(default_factory=list)

    def add_block(self, block: Block) -> None:
        self.blocks.append(block)


class Document(BaseModel):
    """A document: pages of blocks plus rendered markdown."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    source: Path
    pages: list[Page] = Field(default_factory=list)
    markdown: str = ""

    # Detected document-level metadata (filled by post-processors)
    title: str = ""
    body_font_size: float = 0.0              # median across all spans, used for heading z-score
    chrome_texts: set[str] = Field(default_factory=set)  # detected header/footer strings

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def blocks(self) -> list[Block]:
        """All blocks, in (page_idx, reading_order_idx) order."""
        out: list[Block] = []
        for p in self.pages:
            out.extend(sorted(p.blocks, key=lambda b: b.reading_order_idx))
        return out

    def iter_blocks(self) -> Iterator[Block]:
        for p in self.pages:
            yield from sorted(p.blocks, key=lambda b: b.reading_order_idx)
