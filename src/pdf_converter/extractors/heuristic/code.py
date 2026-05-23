"""Code block detector.

The reference relied on per-language regex matching (Python/JS/HTML/...) of
the *content*, which is fragile, English-centric, and misses many real code
samples. Here we detect code by a robust structural signal: clusters of
spans in a monospace font sharing a consistent left margin.

A ``CodeBlock`` replaces the underlying ``TextBlock``s on the page.
Optional Pygments lexer-detection picks a language tag for the fence.
"""

from __future__ import annotations

from collections import defaultdict

from pdf_converter.config import Config
from pdf_converter.ir import BBox, CodeBlock, Document, Provenance, TextBlock
from pdf_converter.utils.fonts import is_monospace_font
from pdf_converter.utils.logging import get_logger

_logger = get_logger("extractors.heuristic.code")
_X_TOLERANCE = 3.0           # px tolerance for "same left margin"
_Y_GAP_TOLERANCE = 2.5       # multiplier of font size to count as adjacent line


class CodeExtractor:
    name = "heuristic.code"

    def __call__(self, doc: Document, *, fitz_doc, config: Config) -> None:
        prov = Provenance(extractor=self.name)
        min_lines = config.heuristic.monospace_min_lines

        for page in doc.pages:
            # Identify TextBlocks whose dominant font is monospace.
            mono_blocks: list[TextBlock] = []
            other: list = []
            for b in page.blocks:
                if isinstance(b, TextBlock) and _is_mono_block(b):
                    mono_blocks.append(b)
                else:
                    other.append(b)

            if not mono_blocks:
                continue

            # Group mono blocks by left margin (x0) and vertical adjacency.
            groups = _group_by_left_margin(mono_blocks)
            new_blocks = list(other)

            for group in groups:
                if len(group) < min_lines and not _has_multi_line(group):
                    new_blocks.extend(group)
                    continue

                lines: list[str] = []
                for blk in group:
                    text = _block_text(blk)
                    for line in text.splitlines() or [text]:
                        lines.append(line)

                bbox = _union_bbox([b.bbox for b in group])
                lang = _detect_language(lines)
                new_blocks.append(
                    CodeBlock(
                        page_idx=group[0].page_idx,
                        bbox=bbox,
                        provenance=prov,
                        lines=lines,
                        language=lang,
                    )
                )

            page.blocks = new_blocks


def _is_mono_block(b: TextBlock) -> bool:
    if not b.spans:
        return False
    mono = sum(
        1
        for s in b.spans
        if s.text.strip() and (s.is_monospace or is_monospace_font(s.font, s.is_monospace))
    )
    nontrivial = sum(1 for s in b.spans if s.text.strip())
    return nontrivial > 0 and (mono / nontrivial) >= 0.7


def _group_by_left_margin(blocks: list[TextBlock]) -> list[list[TextBlock]]:
    """Bucket blocks whose left margins agree (within X tolerance) and that
    are vertically adjacent (next block within Y_GAP_TOLERANCE * font size)."""
    blocks = sorted(blocks, key=lambda b: (round(b.bbox.x0 / _X_TOLERANCE), b.bbox.y0))
    groups: list[list[TextBlock]] = []
    by_margin: dict[float, list[TextBlock]] = defaultdict(list)
    for b in blocks:
        by_margin[round(b.bbox.x0 / _X_TOLERANCE) * _X_TOLERANCE].append(b)

    for _margin, mblocks in by_margin.items():
        mblocks.sort(key=lambda b: b.bbox.y0)
        cur: list[TextBlock] = []
        prev_y1: float | None = None
        for b in mblocks:
            font_size = b.spans[0].font_size if b.spans else 12.0
            if prev_y1 is None or (b.bbox.y0 - prev_y1) <= font_size * _Y_GAP_TOLERANCE:
                cur.append(b)
            else:
                if cur:
                    groups.append(cur)
                cur = [b]
            prev_y1 = b.bbox.y1
        if cur:
            groups.append(cur)
    return groups


def _has_multi_line(group: list[TextBlock]) -> bool:
    return any(
        sum(1 for s in b.spans if s.text == "\n") >= 1 for b in group
    )


def _block_text(b: TextBlock) -> str:
    return "".join(s.text for s in b.spans).rstrip()


def _union_bbox(bboxes: list[BBox]) -> BBox:
    out = bboxes[0]
    for bb in bboxes[1:]:
        out = out.union(bb)
    return out


def _detect_language(lines: list[str]) -> str:
    """Best-effort language detection via pygments. Empty on failure."""
    try:
        from pygments.lexers import guess_lexer  # type: ignore
        from pygments.util import ClassNotFound  # type: ignore
    except ImportError:
        return ""

    src = "\n".join(lines)
    if not src.strip():
        return ""
    try:
        lex = guess_lexer(src)
    except ClassNotFound:
        return ""
    except Exception:
        return ""

    aliases = getattr(lex, "aliases", []) or []
    if aliases:
        return aliases[0]
    name = (lex.name or "").lower().replace(" ", "-")
    return name


__all__ = ["CodeExtractor"]
