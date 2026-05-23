"""Manifest for batch resumability.

Records ``(source_path, source_sha, config_hash, output_path, status)``
per converted PDF. ``run_batch`` consults this on startup to skip work that
has already been done with the same inputs and configuration.

JSON on disk so it's diff-able and human-inspectable.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pdf_converter.config import Config

Status = Literal["pending", "ok", "failed", "skipped"]


def config_hash(cfg: Config) -> str:
    """Stable short hash of the config that affects extraction output."""
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=False)

    source: str                                  # absolute path
    source_sha: str = ""
    config_hash: str = ""
    output: str = ""
    status: Status = "pending"
    error: str = ""
    updated_at: str = ""

    def mark(self, status: Status, *, error: str = "") -> None:
        self.status = status
        self.error = error
        self.updated_at = datetime.now(UTC).isoformat()


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=False)

    path: Path
    entries: dict[str, ManifestEntry] = Field(default_factory=dict)

    @classmethod
    def load_or_new(cls, path: Path) -> Manifest:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entries = {
                    k: ManifestEntry.model_validate(v) for k, v in data.get("entries", {}).items()
                }
                return cls(path=path, entries=entries)
            except Exception:
                # Corrupt manifest -> start over but keep a sidecar copy.
                backup = path.with_suffix(path.suffix + ".bak")
                with contextlib.suppress(OSError):
                    path.rename(backup)
        return cls(path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "updated_at": datetime.now(UTC).isoformat(),
            "entries": {k: v.model_dump(mode="json") for k, v in self.entries.items()},
        }
        # Write atomically.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def should_skip(self, key: str, source_sha: str, cfg_hash: str) -> bool:
        e = self.entries.get(key)
        if e is None:
            return False
        return e.status == "ok" and e.source_sha == source_sha and e.config_hash == cfg_hash

    def record(
        self,
        key: str,
        *,
        source_sha: str,
        cfg_hash: str,
        output: str,
        status: Status,
        error: str = "",
    ) -> None:
        e = self.entries.get(key) or ManifestEntry(source=key)
        e.source_sha = source_sha
        e.config_hash = cfg_hash
        e.output = output
        e.mark(status, error=error)
        self.entries[key] = e


__all__ = ["Manifest", "ManifestEntry", "Status", "config_hash"]
