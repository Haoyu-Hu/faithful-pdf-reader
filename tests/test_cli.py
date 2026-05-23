"""CLI smoke tests using typer's CliRunner."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pdf_converter.cli import app


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "pdf-convert" in result.stdout


def test_cli_run_single_pdf(sample_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["run", str(sample_pdf), "-o", str(out)])
    assert result.exit_code == 0, result.stdout
    assert (out / f"{sample_pdf.stem}.md").is_file()


def test_cli_batch_resumes_skip(sample_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    # First run.
    r1 = CliRunner().invoke(app, ["run", str(sample_pdf), "-o", str(out), "--workers", "1"])
    assert r1.exit_code == 0
    assert (out / "manifest.json").is_file()
    # Second run: should skip via resume.
    r2 = CliRunner().invoke(app, ["run", str(sample_pdf), "-o", str(out), "--workers", "1"])
    assert r2.exit_code == 0
