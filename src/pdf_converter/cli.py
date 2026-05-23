"""``pdf-convert`` CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pdf_converter import __version__
from pdf_converter.api import convert_batch
from pdf_converter.config import Config, Domain, MLMode
from pdf_converter.utils.logging import setup_logging


def _maybe_reexec_for_ml() -> None:
    """On HPC clusters the user's shell often pre-loads CUDA libraries via
    ``LD_LIBRARY_PATH`` (modulefiles, conda envs, etc.). When torch ships
    its own CUDA runtime via pip wheels, the system libs shadow them and
    torch fails to import with "undefined symbol:
    cudaGetDriverEntryPointByVersion".

    The dynamic linker reads ``LD_LIBRARY_PATH`` only at process startup,
    so clearing ``os.environ`` after we're already running doesn't help.
    Instead, we re-exec the CLI once with a cleared ``LD_LIBRARY_PATH``.
    The re-exec is a no-op when ``LD_LIBRARY_PATH`` is unset.

    Users can opt out by setting ``PDF_CONVERTER_KEEP_LD_LIBRARY_PATH=1``.
    """
    if (
        os.environ.get("LD_LIBRARY_PATH")
        and not os.environ.get("PDF_CONVERTER_LD_CLEARED")
        and not os.environ.get("PDF_CONVERTER_KEEP_LD_LIBRARY_PATH")
    ):
        new_env = dict(os.environ)
        new_env.pop("LD_LIBRARY_PATH", None)
        new_env["PDF_CONVERTER_LD_CLEARED"] = "1"
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "pdf_converter.cli", *sys.argv[1:]],
            new_env,
        )


app = typer.Typer(
    name="pdf-convert",
    add_completion=False,
    help="Faithful PDF -> Markdown conversion.",
    no_args_is_help=True,
)
_console = Console()


def _version_callback(value: bool) -> None:
    if value:
        _console.print(f"pdf-convert {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Print version and exit.", callback=_version_callback, is_eager=True),
    ] = False,
) -> None:
    pass


@app.command("run")
def run(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            ...,
            exists=True,
            help="PDF file(s) or directory(ies). Directories are walked recursively.",
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option("-o", "--out", help="Output directory."),
    ] = Path("outputs"),
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help="Parallel worker count. 0 = auto (cpu/2 for heuristic, 1 for ML).",
        ),
    ] = 0,
    ml: Annotated[
        MLMode,
        typer.Option("--ml", help="ML augmentation mode."),
    ] = MLMode.AUTO,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip files whose output already exists."),
    ] = True,
    domain: Annotated[
        Domain,
        typer.Option("--domain", help="Domain hint to bias extractors."),
    ] = Domain.AUTO,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logger level (DEBUG/INFO/WARNING/ERROR)."),
    ] = "INFO",
) -> None:
    """Convert one or many PDFs to Markdown."""
    setup_logging(level=log_level)

    pdf_paths = _collect_pdfs(inputs)
    if not pdf_paths:
        _console.print("[yellow]No PDFs found.[/yellow]")
        raise typer.Exit(1)

    cfg = Config(
        output_dir=out_dir,
        workers=workers,
        ml=ml,
        resume=resume,
        domain=domain,
        log_level=log_level,
    )

    # Always route through the batch runner so the manifest (and thus
    # resume support) is consistent regardless of input count.
    convert_batch(pdf_paths, out_dir=out_dir, workers=workers, config=cfg)


@app.command("convert")
def convert(
    inputs: Annotated[
        list[Path],
        typer.Argument(..., exists=True, help="PDF file(s) or directory(ies)."),
    ],
    out_dir: Annotated[
        Path,
        typer.Option("-o", "--out", help="Output directory."),
    ] = Path("outputs"),
    workers: Annotated[int, typer.Option("--workers", "-w")] = 0,
    ml: Annotated[MLMode, typer.Option("--ml")] = MLMode.AUTO,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    domain: Annotated[Domain, typer.Option("--domain")] = Domain.AUTO,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Alias for ``run`` (matches the README docs)."""
    run(
        inputs=inputs,
        out_dir=out_dir,
        workers=workers,
        ml=ml,
        resume=resume,
        domain=domain,
        log_level=log_level,
    )


def _collect_pdfs(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in inputs:
        if p.is_dir():
            out.extend(sorted(p.rglob("*.pdf")))
            out.extend(sorted(p.rglob("*.PDF")))
        elif p.suffix.lower() == ".pdf":
            out.append(p)
    return sorted(set(out))


def main() -> None:
    """Console-script entry point. Performs HPC fixup, then runs the CLI."""
    _maybe_reexec_for_ml()
    app()


if __name__ == "__main__":
    main()
