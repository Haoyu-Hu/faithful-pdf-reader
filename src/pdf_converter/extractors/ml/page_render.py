"""Helpers to rasterize PDF pages for Surya inputs.

Surya predictors take PIL Images. PyMuPDF is already open in our pipeline,
but pypdfium2 is faster for raw rendering. We use whichever is already
loaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    pass


def render_page(fitz_page, scale: float = 2.0) -> Image.Image:
    """Render a PyMuPDF page to a PIL RGB image."""
    import fitz  # type: ignore

    mat = fitz.Matrix(scale, scale)
    pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def crop_region(image: Image.Image, bbox_pdf, page_size, scale: float = 2.0) -> Image.Image:
    """Crop ``image`` to a PDF-space bbox.

    ``image`` was rendered from ``page_size`` (PDF user-space WxH) at ``scale``.
    ``bbox_pdf`` is (x0, y0, x1, y1) in PDF user-space coordinates.
    """
    px = lambda v: round(v * scale)                  # noqa: E731 - tight scope helper
    x0, y0, x1, y1 = bbox_pdf
    pad = 4
    box = (
        max(0, px(x0) - pad),
        max(0, px(y0) - pad),
        min(image.width, px(x1) + pad),
        min(image.height, px(y1) + pad),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return image.crop((0, 0, 1, 1))              # tiny placeholder
    return image.crop(box)


__all__ = ["crop_region", "render_page"]
