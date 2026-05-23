"""Logging setup. One root logger named ``pdf_converter``."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the package-level logger idempotently and return it."""
    logger = logging.getLogger("pdf_converter")
    logger.setLevel(level)

    # Prevent duplicate handlers when called repeatedly (per-worker imports).
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of ``pdf_converter``."""
    return logging.getLogger(f"pdf_converter.{name}")
