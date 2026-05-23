"""ChromeDetector unit tests against synthetic Documents."""

from __future__ import annotations

import fitz

from pdf_converter.api import convert_pdf


def _make_pdf_with_running_header(path, n_pages: int = 5) -> None:
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        # Running header: same text at top of every page.
        page.insert_text((72, 40), "Confidential Draft", fontsize=9, fontname="helv")
        # Page number at bottom.
        page.insert_text((280, 760), str(i + 1), fontsize=10, fontname="helv")
        # Body.
        page.insert_text((72, 200), f"Section {i + 1} content here.", fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()


def test_running_header_dropped(tmp_path) -> None:
    pdf = tmp_path / "running.pdf"
    _make_pdf_with_running_header(pdf, n_pages=5)
    doc = convert_pdf(pdf, out_dir=tmp_path / "out", write=False)
    md = doc.markdown
    # The running header should appear in chrome_texts (after normalize).
    assert "confidential draft" in doc.chrome_texts
    # And should NOT appear in the rendered markdown.
    assert "Confidential Draft" not in md
    # Page numbers gone.
    for n in range(1, 6):
        assert f"\n{n}\n" not in md
    # Body sections preserved.
    for n in range(1, 6):
        assert f"Section {n} content here." in md


def test_short_doc_does_not_strip_title(tmp_path) -> None:
    """A 2-page doc with a title on page 1 only must NOT classify it as chrome."""
    pdf = tmp_path / "short.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), "My Title", fontsize=20, fontname="helv")
    page.insert_text((72, 200), "First page body.", fontsize=11, fontname="helv")
    page.insert_text((280, 760), "1", fontsize=10, fontname="helv")
    page2 = doc.new_page()
    page2.insert_text((72, 200), "Second page body.", fontsize=11, fontname="helv")
    page2.insert_text((280, 760), "2", fontsize=10, fontname="helv")
    doc.save(pdf)
    doc.close()

    result = convert_pdf(pdf, out_dir=tmp_path / "out", write=False)
    assert "My Title" in result.markdown
    # Page numbers stripped.
    assert "\n1\n" not in result.markdown
    assert "\n2\n" not in result.markdown
