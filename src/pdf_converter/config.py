"""Runtime configuration. Defaults come from pyproject; CLI flags override."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict


class MLMode(StrEnum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"


class Domain(StrEnum):
    AUTO = "auto"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    REPORT = "report"


class HeuristicConfig(BaseModel):
    """Tuning knobs for the heuristic extractors."""

    model_config = ConfigDict(frozen=True)

    heading_zscore_threshold: float = 1.0
    header_band_ratio: float = 0.10
    footer_band_ratio: float = 0.15
    chrome_freq_threshold: float = 0.30
    chrome_min_pages: int = 2             # require >= this many distinct pages to count as chrome
    column_max: int = 3
    math_op_density: float = 0.10
    monospace_min_lines: int = 2          # min lines for a monospace cluster to count as code


class Config(BaseModel):
    """Top-level runtime config."""

    model_config = ConfigDict(frozen=True)

    output_dir: Path = Path("outputs")
    image_subdir: str = "images"
    page_delimiter: str = ""
    workers: int = 0                      # 0 => auto
    ml: MLMode = MLMode.AUTO
    resume: bool = True
    domain: Domain = Domain.AUTO
    log_level: str = "INFO"
    heuristic: HeuristicConfig = HeuristicConfig()

    @classmethod
    def default(cls) -> Self:
        return cls()

    def with_overrides(self, **kwargs) -> Self:
        return self.model_copy(update=kwargs)
