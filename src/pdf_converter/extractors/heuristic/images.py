"""Image / figure extractor.

* Render each image bbox to a PNG via PyMuPDF (clip + matrix zoom).
* Locate the nearest "Figure N: ..." or "Fig. N: ..." text block beneath the
  image and use it as the caption (much more reliable than a generic ML
  captioner).
* Tiny images (width or height < 20px after zoom) are filtered out -- those
  are usually decorative rules / dividers, not figures.
"""

from __future__ import annotations

import re
from pathlib import Path

from pdf_converter.config import Config
from pdf_converter.ir import BBox, Document, FigureBlock, Provenance
from pdf_converter.ir.blocks import BlockKind
from pdf_converter.utils.logging import get_logger

_logger = get_logger("extractors.heuristic.images")

_CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?|table|chart|plot|scheme|diagram)\s*[\dA-Za-z]+[:.\s]",
    re.IGNORECASE,
)
_CAPTION_PROXIMITY_PX = 60.0
_MIN_PIXEL_DIM = 20


class ImageExtractor:
    name = "heuristic.images"

    def __init__(self, *, image_dir: Path, pdf_stem: str, zoom: float = 2.0) -> None:
        self.image_dir = Path(image_dir)
        self.pdf_stem = pdf_stem
        self.zoom = zoom

    def __call__(self, doc: Document, *, fitz_doc, config: Config) -> None:
        import fitz  # type: ignore
        from PIL import Image  # type: ignore

        prov = Provenance(extractor=self.name)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        for page_idx, fpage in enumerate(fitz_doc):
            if page_idx >= doc.n_pages:
                break

            # PyMuPDF's get_text("dict") gives us image *blocks* with bboxes.
            raw = fpage.get_text("dict")
            for blk in raw.get("blocks", []):
                if blk.get("type") != 1:         # image
                    continue
                bbox_t = tuple(blk.get("bbox", (0, 0, 0, 0)))
                bbox = BBox.from_tuple(bbox_t)
                if bbox.width < 1 or bbox.height < 1:
                    continue

                try:
                    mat = fitz.Matrix(self.zoom, self.zoom)
                    pix = fpage.get_pixmap(clip=bbox_t, matrix=mat, alpha=False)
                except Exception as e:
                    _logger.debug("pixmap failed on page %d: %s", page_idx, e)
                    continue

                if pix.width < _MIN_PIXEL_DIM or pix.height < _MIN_PIXEL_DIM:
                    continue

                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                fname = f"{self.pdf_stem}_p{page_idx + 1}_img{blk.get('number', 0)}.png"
                fpath = self.image_dir / fname
                img.save(fpath, "PNG", optimize=True)

                caption = self._find_caption(doc, page_idx, bbox)

                doc.pages[page_idx].blocks.append(
                    FigureBlock(
                        page_idx=page_idx,
                        bbox=bbox,
                        provenance=prov,
                        image_path=str(fpath.relative_to(self.image_dir.parent)),
                        caption=caption,
                        alt_text=caption or f"{self.pdf_stem} figure (page {page_idx + 1})",
                    )
                )

    @staticmethod
    def _find_caption(doc: Document, page_idx: int, fig_bbox: BBox) -> str:
        """Return nearby caption text (below the figure), or empty."""
        page = doc.pages[page_idx]
        candidates = []
        for b in page.blocks:
            if b.kind not in (BlockKind.TEXT, BlockKind.HEADING):
                continue
            txt = "".join(s.text for s in b.spans)         # type: ignore[attr-defined]
            if not _CAPTION_RE.match(txt):
                continue
            gap = b.bbox.y0 - fig_bbox.y1
            if 0 < gap < _CAPTION_PROXIMITY_PX:
                candidates.append((gap, txt.strip()))
        if not candidates:
            return ""
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]


__all__ = ["ImageExtractor"]
