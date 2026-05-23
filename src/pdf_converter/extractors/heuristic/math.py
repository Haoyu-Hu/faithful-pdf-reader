"""Heuristic math/equation detection.

We can't extract LaTeX without ML, but we *can* detect blocks that are
likely equations (high density of math glyphs / Greek letters / operators)
and emit them as ``EquationBlock`` with the original Unicode in
``fallback_text``. When the ML pipeline is enabled, ``ml.textify`` walks
these blocks and fills in ``latex``.

This keeps math from being silently dropped (the reference's behavior) even
in the CPU-only path.
"""

from __future__ import annotations

from pdf_converter.config import Config
from pdf_converter.ir import Document, EquationBlock, Provenance, TextBlock
from pdf_converter.utils.fonts import math_density
from pdf_converter.utils.logging import get_logger

_logger = get_logger("extractors.heuristic.math")

# A block needs both a math-glyph density above threshold AND a small number
# of words. (Body text full of inline math otherwise gets misclassified.)
_MAX_WORD_COUNT = 25


class MathDetector:
    name = "heuristic.math"

    def __call__(self, doc: Document, *, fitz_doc, config: Config) -> None:
        prov = Provenance(extractor=self.name)
        threshold = config.heuristic.math_op_density

        for page in doc.pages:
            new_blocks = []
            for b in page.blocks:
                if isinstance(b, TextBlock) and self._looks_like_equation(b, threshold):
                    text = "".join(s.text for s in b.spans).strip()
                    word_count = len(text.split())
                    if word_count <= _MAX_WORD_COUNT:
                        new_blocks.append(
                            EquationBlock(
                                page_idx=b.page_idx,
                                bbox=b.bbox,
                                provenance=prov,
                                latex="",
                                display=True,
                                fallback_text=text,
                            )
                        )
                        continue
                new_blocks.append(b)
            page.blocks = new_blocks

    @staticmethod
    def _looks_like_equation(b: TextBlock, threshold: float) -> bool:
        text = "".join(s.text for s in b.spans)
        if not text.strip():
            return False
        density = math_density(text)
        if density < threshold:
            return False
        # Avoid flagging headings / titles with one or two math glyphs.
        return len(text.replace(" ", "")) >= 4


__all__ = ["MathDetector"]
