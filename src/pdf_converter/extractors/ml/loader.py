"""Lazy, per-process predictor cache.

Surya predictors hold large CUDA tensors and are not picklable, so they
cannot be passed across `ProcessPoolExecutor` workers. Instead, we keep a
module-level dict in each process and build predictors on first use.

This module also decides whether the ML pipeline is *available* given the
runtime: surya-ocr + torch + a usable GPU. If `--ml on` is requested but
either is missing, ``build_ml_pipeline`` raises so the failure is loud; if
the mode is `auto`, the empty list returned here causes the orchestrator
to fall back to the heuristic-only path.

**HPC note**: this module clears ``LD_LIBRARY_PATH`` for the rest of the
process when torch is loaded with NVIDIA pip wheels but a system CUDA
toolkit is present on ``LD_LIBRARY_PATH``. Cluster modules that pre-load
mismatched CUDA libs would otherwise shadow torch's bundled CUDA runtime
and cause ``undefined symbol`` errors on import. The user can opt out by
setting ``PDF_CONVERTER_KEEP_LD_LIBRARY_PATH=1``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from pdf_converter.config import Config, MLMode
from pdf_converter.extractors.base import Extractor
from pdf_converter.utils.logging import get_logger

_logger = get_logger("extractors.ml.loader")


def _maybe_clear_ld_library_path() -> None:
    """Clear ``LD_LIBRARY_PATH`` to avoid conflicts with system CUDA on HPC."""
    if os.environ.get("PDF_CONVERTER_KEEP_LD_LIBRARY_PATH"):
        return
    if os.environ.get("LD_LIBRARY_PATH"):
        _logger.debug(
            "clearing LD_LIBRARY_PATH (set PDF_CONVERTER_KEEP_LD_LIBRARY_PATH=1 to keep)"
        )
        os.environ.pop("LD_LIBRARY_PATH", None)


# Per-process predictor singletons. Keyed by predictor name.
_PREDICTORS: dict[str, object] = {}


# ----- availability checks ------------------------------------------------


def ml_available() -> tuple[bool, str]:
    """Return (available, reason). ``reason`` is "" when available."""
    _maybe_clear_ld_library_path()
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch not installed"
    try:
        import surya  # noqa: F401
    except ImportError:
        return False, "surya-ocr not installed"
    try:
        import torch as _t
        if not _t.cuda.is_available():
            return True, "cuda unavailable; running on CPU (slow)"
    except Exception as e:
        return False, f"torch.cuda init failed: {e}"
    return True, ""


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----- predictor accessors (one-off lazy load per process) ----------------


def get_layout_predictor() -> object:
    if "layout" not in _PREDICTORS:
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.settings import settings

        _logger.info("loading Surya layout model (device=%s)", _device())
        fp = FoundationPredictor(
            checkpoint=settings.LAYOUT_MODEL_CHECKPOINT,
            device=_device(),
        )
        _PREDICTORS["layout"] = LayoutPredictor(fp)
    return _PREDICTORS["layout"]


def get_recognition_predictor() -> object:
    """Recognition predictor used for math LaTeX OCR (math_mode default True)."""
    if "recognition" not in _PREDICTORS:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        _logger.info("loading Surya recognition model (device=%s)", _device())
        fp = FoundationPredictor(device=_device())   # default recognition checkpoint
        _PREDICTORS["recognition"] = RecognitionPredictor(fp)
    return _PREDICTORS["recognition"]


def get_detection_predictor() -> object:
    if "detection" not in _PREDICTORS:
        from surya.detection import DetectionPredictor

        _logger.info("loading Surya detection model (device=%s)", _device())
        _PREDICTORS["detection"] = DetectionPredictor()
    return _PREDICTORS["detection"]


def get_table_predictor() -> object:
    if "table" not in _PREDICTORS:
        from surya.table_rec import TableRecPredictor

        _logger.info("loading Surya table_rec model (device=%s)", _device())
        _PREDICTORS["table"] = TableRecPredictor()
    return _PREDICTORS["table"]


# ----- pipeline assembly --------------------------------------------------


def build_ml_pipeline(config: Config) -> Sequence[Extractor]:
    """Return the ML extractors selected by ``config.ml``.

    Returns an empty list when ML extras are unavailable (and mode is AUTO).
    """
    available, reason = ml_available()
    if not available:
        if config.ml is MLMode.ON:
            raise RuntimeError(f"--ml on requested but not available: {reason}")
        _logger.info("ML extras unavailable (%s); using heuristic-only", reason)
        return []
    if reason:                                      # available but with caveat (CPU mode)
        _logger.warning("ML available with caveat: %s", reason)

    from pdf_converter.extractors.ml.math_ocr import MathLatexOCR
    from pdf_converter.extractors.ml.surya_layout import SuryaLayout

    return [
        SuryaLayout(),
        MathLatexOCR(),
    ]


__all__ = [
    "build_ml_pipeline",
    "get_detection_predictor",
    "get_layout_predictor",
    "get_recognition_predictor",
    "get_table_predictor",
    "ml_available",
]
