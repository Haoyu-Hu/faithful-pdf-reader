"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_pdf(fixtures_dir: Path) -> Path:
    """A small synthetic PDF generated on-demand for end-to-end tests.

    Generated lazily so tests aren't broken by missing checked-in binaries.
    """
    p = fixtures_dir / "sample.pdf"
    if not p.exists():
        _generate_sample_pdf(p)
    return p


def _generate_sample_pdf(path: Path) -> None:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test Document", fontsize=24, fontname="helv")
    page.insert_text((72, 120), "1. Introduction", fontsize=16, fontname="helv")
    page.insert_text(
        (72, 150),
        "This is body text. It mentions alpha and beta as variables.",
        fontsize=11,
        fontname="helv",
    )
    page.insert_text((72, 200), "- First bullet item", fontsize=11, fontname="helv")
    page.insert_text((72, 215), "- Second bullet item", fontsize=11, fontname="helv")
    page.insert_text((280, 750), "1", fontsize=10, fontname="helv")

    page2 = doc.new_page()
    page2.insert_text((72, 120), "2. Methods", fontsize=16, fontname="helv")
    page2.insert_text((72, 150), "Another paragraph here.", fontsize=11, fontname="helv")
    page2.insert_text((280, 750), "2", fontsize=10, fontname="helv")

    doc.save(path)
    doc.close()
