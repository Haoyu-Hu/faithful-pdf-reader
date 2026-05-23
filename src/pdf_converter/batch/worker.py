"""Per-PDF worker entry. Runs in a child process via ProcessPoolExecutor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pdf_converter.config import Config


def process_one(
    args: tuple[str, str, dict[str, Any]],
) -> dict[str, Any]:
    """Convert one PDF in a subprocess.

    Inputs are picklable: paths as strings and config as a dict so the worker
    rehydrates from pydantic. The return shape is also a plain dict so the
    parent doesn't need to import IR types just to record results.
    """
    src_str, out_str, cfg_dict = args

    from pdf_converter.api import convert_pdf  # lazy import to keep parent light

    cfg = Config.model_validate(cfg_dict)
    src = Path(src_str)
    out_dir = Path(out_str)

    try:
        doc = convert_pdf(src, out_dir=out_dir, config=cfg, write=True)
        return {
            "source": str(src),
            "status": "ok",
            "output": str(out_dir / f"{src.stem}.md"),
            "n_pages": doc.n_pages,
            "n_blocks": sum(len(p.blocks) for p in doc.pages),
            "error": "",
        }
    except Exception as e:
        return {
            "source": str(src),
            "status": "failed",
            "output": "",
            "n_pages": 0,
            "n_blocks": 0,
            "error": f"{type(e).__name__}: {e}",
        }


__all__ = ["process_one"]
