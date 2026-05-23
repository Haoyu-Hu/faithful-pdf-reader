"""pdf-converter: faithful PDF -> Markdown conversion."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pdf-converter")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

from pdf_converter.api import convert_batch, convert_pdf

__all__ = ["__version__", "convert_batch", "convert_pdf"]
