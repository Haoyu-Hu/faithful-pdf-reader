"""Extractor protocol shared by heuristic and ML backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pdf_converter.config import Config
from pdf_converter.ir import Document


@runtime_checkable
class Extractor(Protocol):
    """Anything that mutates ``Document`` in place during conversion.

    Extractors are composed in a pipeline. Each extractor reads what previous
    extractors added (via ``doc.pages[i].blocks``) and appends or refines.

    The ``name`` field is recorded into ``Provenance.extractor`` so that
    downstream debugging can identify the source of every block.
    """

    name: str

    def __call__(self, doc: Document, *, fitz_doc, config: Config) -> None:
        """Mutate ``doc`` in place. ``fitz_doc`` is the open PyMuPDF Document."""
        ...
