"""Math equation -> LaTeX via Surya RecognitionPredictor.

For every ``EquationBlock`` whose ``latex`` field is empty, render the
block's bbox as an image and feed it through the recognition predictor in
math mode. The resulting LaTeX is written back to the block.

Surya batches well, so we accumulate all eligible blocks across the
document and call the predictor in a single batch.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pdf_converter.config import Config
from pdf_converter.ir import Document, EquationBlock
from pdf_converter.ir.blocks import BlockKind
from pdf_converter.utils.logging import get_logger

# Surya returns LaTeX wrapped in MathML tags when math_mode is on.
# Strip <math ...>...</math> wrappers but keep the inner LaTeX intact.
_MATHML_WRAPPER_RE = re.compile(
    r"^\s*<math\b[^>]*>(?P<body>.*?)</math>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_TRAILING_DISPLAY_ATTR_RE = re.compile(r'\s*display\s*=\s*["\']?\w+["\']?', re.IGNORECASE)


def _unwrap_mathml(raw: str) -> tuple[str, bool]:
    """Return (latex, display_mode). ``display_mode`` is True for block math."""
    if not raw:
        return "", True
    m = _MATHML_WRAPPER_RE.match(raw.strip())
    if not m:
        return raw.strip(), True
    body = m.group("body").strip()
    display = "display=" in raw and 'display="block"' in raw.lower()
    # Surya sometimes also adds attribute strings inside body -- defensive strip.
    body = _TRAILING_DISPLAY_ATTR_RE.sub("", body)
    return body, bool(display) or len(body) > 30

if TYPE_CHECKING:
    pass

_logger = get_logger("extractors.ml.math_ocr")


class MathLatexOCR:
    name = "ml.math_ocr"

    def __init__(self, *, scale: float = 3.0, max_blocks: int = 200) -> None:
        self.scale = scale                          # higher scale than layout for sharper glyphs
        self.max_blocks = max_blocks

    def __call__(self, doc: Document, *, fitz_doc, config: Config) -> None:
        from pdf_converter.extractors.ml.loader import get_recognition_predictor
        from pdf_converter.extractors.ml.page_render import crop_region, render_page

        # Collect equation blocks needing LaTeX.
        targets: list[EquationBlock] = []
        for page in doc.pages:
            for b in page.blocks:
                if b.kind is BlockKind.EQUATION and not b.latex.strip():   # type: ignore[attr-defined]
                    targets.append(b)                # type: ignore[arg-type]
        if not targets:
            return
        if len(targets) > self.max_blocks:
            _logger.warning(
                "math_ocr: capping equation OCR at %d (have %d)",
                self.max_blocks,
                len(targets),
            )
            targets = targets[: self.max_blocks]

        _logger.info("math_ocr: %d equation blocks queued for LaTeX OCR", len(targets))

        # Render and crop each region. Rasterize each page once.
        page_cache: dict[int, object] = {}
        crops: list = []
        for b in targets:
            if b.page_idx not in page_cache:
                page_cache[b.page_idx] = render_page(fitz_doc[b.page_idx], scale=self.scale)
            page_img = page_cache[b.page_idx]
            page = doc.pages[b.page_idx]
            crops.append(
                crop_region(
                    page_img,
                    (b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1),
                    (page.width, page.height),
                    scale=self.scale,
                )
            )

        predictor = get_recognition_predictor()
        if predictor is None:
            return

        from surya.common.surya.schema import TaskNames

        tasks = [TaskNames.block_without_boxes] * len(crops)
        bboxes = [[[0, 0, img.width, img.height]] for img in crops]

        try:
            results = predictor(                     # type: ignore[operator]
                crops,
                task_names=tasks,
                bboxes=bboxes,
                math_mode=True,
            )
        except Exception as e:
            _logger.error("math OCR inference failed: %s", e)
            return

        prov_name = self.name
        for block, res in zip(targets, results, strict=False):
            try:
                raw = res.text_lines[0].text
            except (AttributeError, IndexError):
                continue
            latex, display = _unwrap_mathml(raw)
            if not latex:
                continue
            block.latex = latex
            block.display = display
            block.provenance = block.provenance.model_copy(
                update={"extractor": prov_name, "confidence": 0.9, "notes": "ml.math_ocr"}
            )


__all__ = ["MathLatexOCR"]
