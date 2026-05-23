"""Low-level IR primitives: bounding boxes, spans, links."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator


class BBox(BaseModel):
    """Axis-aligned bounding box in PDF user-space coordinates (origin top-left)."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @field_validator("x1")
    @classmethod
    def _check_x_order(cls, x1: float, info) -> float:
        if "x0" in info.data and x1 < info.data["x0"]:
            raise ValueError(f"bbox x1 ({x1}) < x0 ({info.data['x0']})")
        return x1

    @field_validator("y1")
    @classmethod
    def _check_y_order(cls, y1: float, info) -> float:
        if "y0" in info.data and y1 < info.data["y0"]:
            raise ValueError(f"bbox y1 ({y1}) < y0 ({info.data['y0']})")
        return y1

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float, float]) -> Self:
        return cls(x0=t[0], y0=t[1], x1=t[2], y1=t[3])

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    def union(self, other: BBox) -> BBox:
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def intersects(self, other: BBox) -> bool:
        return not (
            self.x1 < other.x0
            or other.x1 < self.x0
            or self.y1 < other.y0
            or other.y1 < self.y0
        )

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


class Link(BaseModel):
    """A hyperlink target attached to a span."""

    model_config = ConfigDict(frozen=True)

    uri: str
    rect: BBox


class Span(BaseModel):
    """A run of text with uniform font / styling, originating from a single PDF span."""

    model_config = ConfigDict(frozen=True)

    text: str
    bbox: BBox
    font: str = ""
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False
    is_monospace: bool = False
    is_superscript: bool = False
    is_subscript: bool = False
    color: int = 0          # packed sRGB int as returned by PyMuPDF
    link: Link | None = None

    @classmethod
    def from_pymupdf(
        cls,
        text: str,
        bbox: tuple[float, float, float, float],
        font: str,
        font_size: float,
        flags: int,
        color: int = 0,
        link: Link | None = None,
    ) -> Self:
        """Build a Span from PyMuPDF's per-span dict output.

        Flag bits per PyMuPDF: 0=superscript, 1=italic, 2=serif, 3=monospace, 4=bold, 5=smallcaps.
        """
        return cls(
            text=text,
            bbox=BBox.from_tuple(bbox),
            font=font,
            font_size=font_size,
            is_superscript=bool(flags & (1 << 0)),
            is_italic=bool(flags & (1 << 1)),
            is_monospace=bool(flags & (1 << 3)),
            is_bold=bool(flags & (1 << 4)),
            is_subscript=False,
            color=color,
            link=link,
        )
