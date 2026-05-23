"""Markdown table rendering."""

from __future__ import annotations

from typing import Literal

Align = Literal["left", "right", "center"]


def render_table(
    cells: list[list[str]],
    header_rows: int = 1,
    alignments: list[Align] | None = None,
) -> str:
    """Render a 2-D cell matrix as a GitHub-Flavored Markdown table.

    * Cells are stripped, ``|`` inside cells is escaped, newlines collapsed to ``<br>``.
    * Column widths are padded for human readability.
    * If ``header_rows`` < 1, a synthetic empty header is emitted (md tables
      require a header separator).
    """
    if not cells:
        return ""

    n_cols = max(len(row) for row in cells)
    if n_cols == 0:
        return ""

    norm: list[list[str]] = []
    for row in cells:
        normed = [_clean(c) for c in row] + [""] * (n_cols - len(row))
        norm.append(normed)

    aligns: list[Align] = list(alignments or [])
    if len(aligns) < n_cols:
        aligns.extend(["left"] * (n_cols - len(aligns)))

    if header_rows < 1:
        # GFM needs a header row; synthesize a blank one.
        norm.insert(0, [""] * n_cols)
        header_rows = 1

    # Cap cell padding so cells with long content don't explode the
    # separator row into 1000s of dashes; readers still see the full text.
    _MAX_PAD = 60
    col_widths = [
        max(len(norm[r][c]) for r in range(len(norm))) for c in range(n_cols)
    ]
    col_widths = [min(_MAX_PAD, max(3, w)) for w in col_widths]

    def fmt_row(row: list[str]) -> str:
        cells_str = []
        for i, cell in enumerate(row):
            # If the actual content is longer than the cap, emit it verbatim --
            # markdown tables don't require equal column widths, only equal column counts.
            pad_to = max(col_widths[i], len(cell))
            if aligns[i] == "right":
                cells_str.append(cell.rjust(pad_to))
            elif aligns[i] == "center":
                cells_str.append(cell.center(pad_to))
            else:
                cells_str.append(cell.ljust(pad_to))
        return "| " + " | ".join(cells_str) + " |"

    def separator() -> str:
        seps: list[str] = []
        for i in range(n_cols):
            w = col_widths[i]
            if aligns[i] == "right":
                seps.append("-" * (w - 1) + ":")
            elif aligns[i] == "center":
                seps.append(":" + "-" * (w - 2) + ":")
            else:
                seps.append("-" * w)
        return "| " + " | ".join(seps) + " |"

    out_lines: list[str] = []
    for r in range(header_rows):
        out_lines.append(fmt_row(norm[r]))
    out_lines.append(separator())
    for r in range(header_rows, len(norm)):
        out_lines.append(fmt_row(norm[r]))
    return "\n".join(out_lines)


def _clean(cell: str | None) -> str:
    if cell is None:
        return ""
    return (
        str(cell)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", "<br>")
        .replace("\r", "")
        .strip()
    )


__all__ = ["render_table"]
