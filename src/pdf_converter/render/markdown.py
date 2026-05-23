"""IR -> Markdown rendering.

Renderer is exhaustive over ``BlockKind``: adding a new block kind requires
a new branch in ``_render_block`` (mypy will flag a missing branch via the
``assert_never`` call).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import assert_never

from pdf_converter.ir import (
    Block,
    BlockKind,
    CodeBlock,
    Document,
    EquationBlock,
    FigureBlock,
    HeadingBlock,
    ListBlock,
    ListItem,
    Span,
    TableBlock,
    TextBlock,
)
from pdf_converter.render.postprocess import mdformat_text, normalize_markdown
from pdf_converter.render.tables import render_table


@dataclass(frozen=True)
class _Style:
    """Inline formatting flags for a span group."""

    is_bold: bool = False
    is_italic: bool = False
    is_monospace: bool = False
    is_superscript: bool = False
    is_subscript: bool = False
    link_uri: str = ""
    linebreak_only: bool = False

    @classmethod
    def from_span(cls, s: Span) -> _Style:
        return cls(
            is_bold=s.is_bold,
            is_italic=s.is_italic,
            is_monospace=s.is_monospace,
            is_superscript=s.is_superscript,
            is_subscript=s.is_subscript,
            link_uri=s.link.uri if s.link else "",
        )

    @classmethod
    def linebreak(cls) -> _Style:
        return cls(linebreak_only=True)


# CommonMark special characters that need escaping in inline text.
# `.` and `-` are intentionally omitted -- they only matter at start of line,
# and over-escaping them in URLs / sentences was producing artifacts like
# "\.com" leaking into auto-links.
_MD_ESCAPE_CHARS = r"\\`*_{}[]()#+!|<>"
_MD_ESCAPE_RE = re.compile(f"([{re.escape(_MD_ESCAPE_CHARS)}])")
_LINK_AUTODETECT_RE = re.compile(r"(?P<url>https?://[^\s)\]<>]+)")


def render_document(doc: Document, *, format_with_mdformat: bool = True) -> str:
    """Render the full document and store the result on ``doc.markdown``."""
    return MarkdownRenderer(format_with_mdformat=format_with_mdformat).render(doc)


class MarkdownRenderer:
    """Stateful markdown renderer. ``footnotes`` collects across the document."""

    def __init__(self, *, format_with_mdformat: bool = True) -> None:
        self._format = format_with_mdformat
        self._footnotes: list[tuple[str, str]] = []  # (marker, body)

    # --- public ------------------------------------------------------------

    def render(self, doc: Document) -> str:
        chunks: list[str] = []

        # Title (first level-1 heading on the first page, if any).
        if doc.title:
            chunks.append(f"# {self._escape(doc.title)}\n")

        for page in doc.pages:
            blocks = sorted(page.blocks, key=lambda b: b.reading_order_idx)
            for b in blocks:
                rendered = self._render_block(b)
                if rendered:
                    chunks.append(rendered)

        if self._footnotes:
            chunks.append(self._render_footnote_section())

        md = "\n\n".join(c for c in chunks if c.strip())
        md = normalize_markdown(md)
        if self._format:
            md = mdformat_text(md)
        doc.markdown = md
        return md

    # --- dispatch ----------------------------------------------------------

    def _render_block(self, b: Block) -> str:
        kind = b.kind
        if kind is BlockKind.TEXT:
            return self._render_text(b)              # type: ignore[arg-type]
        if kind is BlockKind.HEADING:
            return self._render_heading(b)           # type: ignore[arg-type]
        if kind is BlockKind.LIST:
            return self._render_list(b)              # type: ignore[arg-type]
        if kind is BlockKind.TABLE:
            return self._render_table_block(b)       # type: ignore[arg-type]
        if kind is BlockKind.FIGURE:
            return self._render_figure(b)            # type: ignore[arg-type]
        if kind is BlockKind.EQUATION:
            return self._render_equation(b)          # type: ignore[arg-type]
        if kind is BlockKind.CODE:
            return self._render_code(b)              # type: ignore[arg-type]
        if kind is BlockKind.FOOTNOTE:
            self._footnotes.append(
                (b.marker, self._render_spans(b.spans))   # type: ignore[attr-defined]
            )
            return ""                                # body deferred
        assert_never(kind)

    # --- per-type renderers ------------------------------------------------

    def _render_text(self, b: TextBlock) -> str:
        return self._render_spans(b.spans).strip()

    def _render_heading(self, b: HeadingBlock) -> str:
        level = max(1, min(6, b.level))
        text = self._render_spans(b.spans).strip()
        if not text:
            return ""
        return f"{'#' * level} {text}"

    def _render_list(self, b: ListBlock) -> str:
        lines: list[str] = []
        self._collect_list_lines(b.items, b.ordered, lines, depth=0)
        return "\n".join(lines)

    def _collect_list_lines(
        self,
        items: list[ListItem],
        ordered: bool,
        out: list[str],
        depth: int,
    ) -> None:
        indent = "  " * depth
        for i, item in enumerate(items, start=1):
            marker = f"{i}." if ordered else "-"
            text = self._render_spans(item.spans).strip()
            out.append(f"{indent}{marker} {text}")
            if item.nested:
                # Nested children inherit the ordered flag from the parent for now.
                self._collect_list_lines(item.nested, ordered, out, depth + 1)

    def _render_table_block(self, b: TableBlock) -> str:
        tbl = render_table(b.cells, header_rows=b.header_rows, alignments=b.alignments)
        if b.caption:
            return f"{tbl}\n\n*{self._escape(b.caption)}*"
        return tbl

    def _render_figure(self, b: FigureBlock) -> str:
        # Keep alt text short. Long alt text in `![alt](path)` is awkward in
        # markdown source, and the caption is shown verbatim below anyway.
        alt = _short_alt(b.caption, b.alt_text)
        line = f"![{self._escape(alt)}]({b.image_path})"
        if b.caption:
            return f"{line}\n\n*{self._escape(b.caption)}*"
        return line

    def _render_equation(self, b: EquationBlock) -> str:
        body = b.latex.strip() if b.latex else b.fallback_text.strip()
        if not body:
            return ""
        if b.display:
            return f"$$\n{body}\n$$"
        return f"${body}$"

    def _render_code(self, b: CodeBlock) -> str:
        lang = b.language or ""
        body = "\n".join(b.lines).rstrip()
        return f"```{lang}\n{body}\n```"

    def _render_footnote_section(self) -> str:
        lines = ["", "---", "", "### Footnotes", ""]
        for marker, body in self._footnotes:
            lines.append(f"[^{marker}]: {body.strip()}")
        return "\n".join(lines)

    # --- helpers -----------------------------------------------------------

    def _render_spans(self, spans: list[Span]) -> str:
        # Group consecutive spans that share inline-formatting flags so that
        # markdown emphasis wraps whole runs rather than each PyMuPDF span
        # individually ("*hello world*" instead of "*hello* *world*").
        groups: list[tuple[_Style, list[Span]]] = []
        for s in spans:
            if s.text == "\n":
                groups.append((_Style.linebreak(), [s]))
                continue
            style = _Style.from_span(s)
            if groups and groups[-1][0] == style and not groups[-1][0].linebreak_only:
                groups[-1][1].append(s)
            else:
                groups.append((style, [s]))

        parts: list[str] = []
        for style, group in groups:
            if style.linebreak_only:
                parts.append("\n")
                continue
            raw = "".join(g.text for g in group)
            if style.is_monospace:
                # Don't escape inside code.
                parts.append(f"`{raw}`")
                continue
            # Auto-link bare URLs FIRST (on raw text) so escaping doesn't
            # corrupt the URL contents.
            inner = self._escape_with_urls(raw)
            if style.is_superscript:
                inner = f"<sup>{inner}</sup>"
            elif style.is_subscript:
                inner = f"<sub>{inner}</sub>"
            if style.is_bold and style.is_italic:
                inner = f"***{inner}***"
            elif style.is_bold:
                inner = f"**{inner}**"
            elif style.is_italic:
                inner = f"*{inner}*"
            if style.link_uri:
                inner = f"[{inner.strip()}]({style.link_uri})"
            parts.append(inner)

        text = "".join(parts)
        text = re.sub(r"[ \t]+", " ", text)
        return text

    @classmethod
    def _escape_with_urls(cls, text: str) -> str:
        """Escape markdown specials but pass URLs through verbatim inside <...>."""
        if not text:
            return ""
        out: list[str] = []
        last = 0
        for m in _LINK_AUTODETECT_RE.finditer(text):
            out.append(_MD_ESCAPE_RE.sub(r"\\\1", text[last : m.start()]))
            out.append(f"<{m.group('url')}>")
            last = m.end()
        out.append(_MD_ESCAPE_RE.sub(r"\\\1", text[last:]))
        return "".join(out)

    @staticmethod
    def _escape(text: str) -> str:
        """Escape markdown special chars in plain text. Newlines pass through."""
        if not text:
            return ""
        # Don't escape inside an already-formatted string; this is for raw text only.
        return _MD_ESCAPE_RE.sub(r"\\\1", text)


_SHORT_ALT_RE = re.compile(r"^\s*((?:figure|fig\.?|table|chart|plot|scheme|diagram)\s*[\dA-Za-z]+)", re.IGNORECASE)


def _short_alt(caption: str, fallback: str) -> str:
    """Return a short alt label: e.g. 'Figure 2' from 'Figure 2. Statistics of ...'."""
    if caption:
        m = _SHORT_ALT_RE.match(caption)
        if m:
            return m.group(1).strip()
        # No "Figure N" prefix -- take the first few words.
        words = caption.strip().split()[:4]
        return " ".join(words) if words else "figure"
    return fallback or "figure"


__all__ = ["MarkdownRenderer", "render_document"]
