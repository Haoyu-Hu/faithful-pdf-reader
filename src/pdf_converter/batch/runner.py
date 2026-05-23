"""Parallel batch driver.

Uses ProcessPoolExecutor so each worker has its own ML model state (one
copy per process, lazy-loaded). Progress is reported via rich.

Resumability: a JSON manifest under ``out_dir/manifest.json`` records the
``(source_sha, config_hash)`` of each conversion. On restart we skip
entries whose status is ``ok`` and whose inputs still hash to the same
value -- if either the source file or the config changes, the file is
reprocessed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pdf_converter.batch.manifest import Manifest, config_hash
from pdf_converter.batch.worker import process_one
from pdf_converter.config import Config, MLMode
from pdf_converter.ir import Document
from pdf_converter.utils.logging import get_logger, setup_logging
from pdf_converter.utils.pdf import sha256

_logger = get_logger("batch.runner")
_console = Console()


def _resolve_workers(requested: int, ml_on: bool) -> int:
    if requested > 0:
        return requested
    if ml_on:
        return 1                                 # one A40, one ML pipeline
    return max(1, (os.cpu_count() or 2) // 2)


def run_batch(
    paths: Sequence[Path],
    *,
    out_dir: Path,
    workers: int,
    config: Config,
) -> list[Document]:
    """Run the converter over ``paths`` in parallel.

    Returns an empty list: child processes can't return Document instances
    cheaply (large pydantic models), so the parent records only summary
    metadata via the manifest. Library callers who need full Documents
    should iterate ``convert_pdf`` themselves.
    """
    setup_logging(level=config.log_level)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    manifest = Manifest.load_or_new(manifest_path)
    cfg_h = config_hash(config)

    ml_on = config.ml in (MLMode.ON, MLMode.AUTO)
    n_workers = _resolve_workers(workers, ml_on)
    _console.print(f"[bold]pdf-convert[/bold]: {len(paths)} PDFs, {n_workers} worker(s)")

    # Build the to-do list, honoring --resume.
    todo: list[tuple[str, str, dict]] = []
    skipped = 0
    cfg_dict = config.model_dump(mode="json")
    for p in paths:
        p = Path(p).resolve()
        key = str(p)
        sha = sha256(p)
        if config.resume and manifest.should_skip(key, sha, cfg_h):
            skipped += 1
            continue
        manifest.record(
            key,
            source_sha=sha,
            cfg_hash=cfg_h,
            output=str(out_dir / f"{p.stem}.md"),
            status="pending",
        )
        todo.append((str(p), str(out_dir), cfg_dict))

    if skipped:
        _console.print(f"[dim]skipped {skipped} (resume)[/dim]")

    if not todo:
        manifest.save()
        return []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=_console,
    ) as progress:
        task = progress.add_task("converting", total=len(todo))

        if n_workers == 1:
            # Synchronous in-process for easier debugging and to keep memory low.
            for triple in todo:
                result = process_one(triple)
                _record_result(manifest, result, cfg_h)
                manifest.save()
                progress.advance(task)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(process_one, t) for t in todo]
                for fut in as_completed(futures):
                    result = fut.result()
                    _record_result(manifest, result, cfg_h)
                    manifest.save()
                    progress.advance(task)

    failed = sum(1 for e in manifest.entries.values() if e.status == "failed")
    ok = sum(1 for e in manifest.entries.values() if e.status == "ok")
    _console.print(f"[green]done[/green] ok={ok} failed={failed} skipped={skipped}")
    return []


def _record_result(manifest: Manifest, result: dict, cfg_h: str) -> None:
    src = result["source"]
    entry = manifest.entries.get(src)
    sha = entry.source_sha if entry else ""
    manifest.record(
        src,
        source_sha=sha,
        cfg_hash=cfg_h,
        output=result.get("output", ""),
        status=result["status"],
        error=result.get("error", ""),
    )


__all__ = ["run_batch"]
