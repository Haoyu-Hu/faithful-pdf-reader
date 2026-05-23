"""High-level public API: ``convert_pdf`` and ``convert_batch``."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import fitz  # type: ignore

from pdf_converter.config import Config, MLMode
from pdf_converter.extractors.base import Extractor
from pdf_converter.extractors.heuristic import (
    ChromeDetector,
    CodeExtractor,
    ImageExtractor,
    LayoutOrderer,
    MathDetector,
    TableExtractor,
    TextExtractor,
)
from pdf_converter.ir import Document
from pdf_converter.render import render_document
from pdf_converter.utils.logging import get_logger

_logger = get_logger("api")


def convert_pdf(
    path: str | Path,
    *,
    out_dir: str | Path | None = None,
    config: Config | None = None,
    write: bool = True,
) -> Document:
    """Convert a single PDF to Markdown.

    Returns the populated ``Document`` (with ``doc.markdown`` filled in). If
    ``write`` and ``out_dir`` are set, the markdown is also written to
    ``<out_dir>/<stem>.md`` and any extracted images go under
    ``<out_dir>/<image_subdir>/``.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(src)

    cfg = config or Config.default()
    out = Path(out_dir) if out_dir is not None else cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    image_dir = out / cfg.image_subdir
    image_dir.mkdir(parents=True, exist_ok=True)

    doc = Document(source=src)
    extractors = _build_pipeline(cfg, image_dir=image_dir, pdf_stem=src.stem)

    fitz_doc = fitz.open(src)
    try:
        for ext in extractors:
            _logger.debug("running extractor %s on %s", ext.name, src.name)
            ext(doc, fitz_doc=fitz_doc, config=cfg)
    finally:
        fitz_doc.close()

    render_document(doc)

    if write:
        out_path = out / f"{src.stem}.md"
        out_path.write_text(doc.markdown, encoding="utf-8")
        _logger.info("wrote %s", out_path)

    return doc


def convert_batch(
    paths: Iterable[str | Path],
    *,
    out_dir: str | Path,
    workers: int = 0,
    config: Config | None = None,
) -> list[Document]:
    """Convert multiple PDFs. Lazy-imports the batch runner so the heuristic
    path doesn't pay for ProcessPoolExecutor setup when called as a library.
    """
    from pdf_converter.batch.runner import run_batch

    return run_batch(
        list(paths),
        out_dir=Path(out_dir),
        workers=workers,
        config=config or Config.default(),
    )


def _build_pipeline(
    cfg: Config,
    *,
    image_dir: Path,
    pdf_stem: str,
) -> Sequence[Extractor]:
    """Compose the extractor pipeline for the given config.

    Order matters: text first (to compute font stats), then tables / images /
    code / math (which may depend on or remove text blocks), then layout
    ordering, then chrome removal (frequency-based, needs all pages loaded).
    """
    pipeline: list[Extractor] = [
        TextExtractor(),
        TableExtractor(),
        ImageExtractor(image_dir=image_dir, pdf_stem=pdf_stem),
        CodeExtractor(),
        MathDetector(),
        LayoutOrderer(),
        ChromeDetector(),
    ]

    if cfg.ml in (MLMode.ON, MLMode.AUTO):
        # ML extractors are optional and lazy-imported. ``AUTO`` falls back to
        # heuristic silently when surya-ocr or torch are not installed.
        try:
            from pdf_converter.extractors.ml.loader import build_ml_pipeline

            ml_extractors = build_ml_pipeline(cfg)
            if ml_extractors:
                # ML extractors run after heuristic so they can refine.
                pipeline = list(pipeline) + list(ml_extractors)
            elif cfg.ml is MLMode.ON:
                raise RuntimeError("--ml on but ML extras not installed")
        except ImportError:
            if cfg.ml is MLMode.ON:
                raise RuntimeError(
                    "--ml on but `pip install pdf-converter[ml]` is required"
                ) from None
            _logger.debug("ML extras not available; running heuristic-only")

    return pipeline


__all__ = ["convert_batch", "convert_pdf"]
