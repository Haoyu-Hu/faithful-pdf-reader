"""PDF utilities: hashing, optional pikepdf repair, helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pdf_converter.utils.logging import get_logger

_logger = get_logger("utils.pdf")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def repair_pdf_inplace(path: Path) -> bool:
    """Attempt to repair a malformed PDF with pikepdf. Returns True on rewrite.

    Best-effort: silently returns False if pikepdf is unavailable or the file
    is already valid.
    """
    try:
        import pikepdf  # type: ignore
    except ImportError:
        _logger.debug("pikepdf not installed; skipping repair")
        return False

    try:
        with pikepdf.open(path, allow_overwriting_input=True) as pdf:
            pdf.save(path)
        return True
    except Exception as e:
        _logger.debug("pikepdf repair failed for %s: %s", path, e)
        return False
