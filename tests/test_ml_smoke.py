"""ML smoke test. Skipped by default; run with ``pytest -m ml``.

Verifies that the ML pipeline assembles and the layout predictor runs end
to end. Does not assert on the output (Surya's behavior is what it is);
just that no exception is raised.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.ml


def test_ml_available_status() -> None:
    """If --ml on is requested but extras absent, build_ml_pipeline must raise."""
    from pdf_converter.config import Config, MLMode
    from pdf_converter.extractors.ml.loader import build_ml_pipeline, ml_available

    available, reason = ml_available()
    if not available:
        # Hard requirement: --ml on must fail loudly with a clear message.
        with pytest.raises(RuntimeError) as ei:
            build_ml_pipeline(Config(ml=MLMode.ON))
        assert reason in str(ei.value)
    else:
        pipeline = build_ml_pipeline(Config(ml=MLMode.ON))
        names = [e.name for e in pipeline]
        assert "ml.surya_layout" in names
        assert "ml.math_ocr" in names


def test_ml_end_to_end(sample_pdf: Path, tmp_path: Path) -> None:
    """Run the full ML pipeline on the small synthetic sample PDF."""
    from pdf_converter.api import convert_pdf
    from pdf_converter.config import Config, MLMode
    from pdf_converter.extractors.ml.loader import ml_available

    available, _ = ml_available()
    if not available:
        pytest.skip("ML extras not installed")

    doc = convert_pdf(
        sample_pdf,
        out_dir=tmp_path / "out",
        config=Config(ml=MLMode.ON),
        write=False,
    )
    assert doc.n_pages > 0
    assert doc.markdown
