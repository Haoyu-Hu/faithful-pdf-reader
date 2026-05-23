# faithful-pdf-reader

A faithful PDF → Markdown converter built around a typed intermediate
representation. Designed for batch conversion of mixed corpora — academic
papers, technical documentation, business reports, medical/biological
literature — where preserving structure, math, and specialized terminology
verbatim is the goal.

The pipeline runs heuristically on the CPU by default and optionally augments
the output with GPU-accelerated ML models for layout understanding and LaTeX
equation OCR.

## Features

- **Faithful by design.** Headings, lists, tables, code blocks, equations,
  figures, footnotes, and inline emphasis are preserved with their semantic
  type rather than collapsed to plain text.
- **Hybrid extraction.** A heuristic CPU baseline (PyMuPDF, pdfplumber,
  pikepdf) handles 90 % of real documents with no ML dependencies. ML
  augmentation (Surya layout + recognition) is opt-in for math, scanned
  pages, and complex layouts.
- **Math support.** Equations are emitted as LaTeX inside `$$ … $$`
  fences when ML mode is on, and preserved as Unicode fallback otherwise
  (instead of being silently dropped).
- **Multi-column reading order.** Two- and three-column layouts are
  detected and read in the conventional left-to-right, top-to-bottom
  order, with full-width blocks (titles, abstracts, full-page figures)
  interleaved correctly.
- **Frequency-based chrome removal.** Running headers, footers, and page
  numbers are detected by recurrence across pages rather than hardcoded
  Y-coordinates, so document-specific layouts work out of the box.
- **Batch + resumable.** A `ProcessPoolExecutor`-backed runner converts
  directories in parallel with a JSON manifest that lets interrupted runs
  resume cleanly.
- **Typed intermediate representation.** Every block carries its bounding
  box, reading-order index, and provenance (which extractor produced it
  and with what confidence), enabling deterministic post-processing and
  A/B comparison between heuristic and ML output.

## Requirements

- Python 3.11 or later
- For ML mode: an NVIDIA GPU with CUDA 12.6 or later (4 GB VRAM minimum,
  8 GB recommended). CPU-only ML works but is slow.

## Installation

```bash
git clone https://github.com/Haoyu-Hu/faithful-pdf-reader.git
cd faithful-pdf-reader

# heuristic-only (CPU, MIT-licensed)
pip install -e .

# + Surya layout & math LaTeX OCR (GPU recommended)
pip install -e '.[ml]'

# + test & lint tooling
pip install -e '.[dev]'

# everything
pip install -e '.[all]'
```

The base package is MIT-licensed. The `[ml]` extra pulls **surya-ocr**,
which is GPL-3.0-or-later — your runtime is then subject to GPL-3 terms.
Skip the `[ml]` extra to keep the runtime pure MIT.

## Quick start

### Command line

```bash
# Single PDF
pdf-convert run paper.pdf -o outputs/

# A directory of PDFs in parallel
pdf-convert run ./papers/ -o outputs/ --workers 4

# Enable ML augmentation (requires GPU + the [ml] extra)
pdf-convert run paper.pdf -o outputs/ --ml on

# Resume an interrupted batch (default: --resume)
pdf-convert run ./papers/ -o outputs/ --workers 4

# Force a clean rerun
pdf-convert run ./papers/ -o outputs/ --no-resume
```

Useful flags:

| Flag | Default | What it does |
|---|---|---|
| `-o`, `--out` | `outputs/` | Output directory. Each PDF becomes `<stem>.md`, with figures under `<out>/images/`. |
| `--workers`, `-w` | `0` (auto) | Parallel worker count. `0` = `cpu_count // 2` for heuristic, `1` for ML. |
| `--ml` | `auto` | `auto` falls back to heuristic if ML is unavailable; `on` requires it; `off` disables. |
| `--resume / --no-resume` | `--resume` | Skip files already converted with the same source SHA and config. |
| `--domain` | `auto` | Domain hint (`academic`, `technical`, `medical`, `report`) that biases extractor tuning. |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

### Library API

```python
from pdf_converter import convert_pdf, convert_batch
from pdf_converter.config import Config, MLMode

# Single file
doc = convert_pdf("paper.pdf")
print(doc.markdown)

# Walk the intermediate representation
for block in doc.iter_blocks():
    print(block.kind, block.bbox, block.provenance.extractor)

# With ML enabled
doc = convert_pdf("paper.pdf", config=Config(ml=MLMode.ON))

# Parallel batch
convert_batch(["a.pdf", "b.pdf"], out_dir="outputs/", workers=4)
```

## Design

```
PDF  ─► load (PyMuPDF)
     ─► extract  (heuristic + optional ML, per block kind)
     ─► IR       (Document / Page / Block tagged union)
                 each block has bbox + reading-order index + provenance
     ─► sort by reading order
     ─► post-process  (frequency-based chrome filter, span merging)
     ─► render        (markdown builder, exhaustive over block kinds)
     ─► write          .md + sidecar  images/
```

### Block kinds

`text` · `heading` · `list` · `table` · `figure` · `equation` · `code` ·
`footnote`

Each kind has a dedicated Pydantic model, and the renderer dispatches
exhaustively over a discriminated union — adding a new block kind triggers
a static type error in `render/markdown.py` until a new branch is added.

### Extractor pipeline

Heuristic extractors (always available):

1. `text` — PyMuPDF blocks → text/heading/list with relative font-size
   heading detection.
2. `tables` — pdfplumber tables, positioned in the IR by bounding box.
3. `images` — extracted at 2× scale, captioned from nearby "Figure N:" text.
4. `code` — monospace-font runs with consistent left margins.
5. `math` — text blocks with high math-glyph density become `EquationBlock`
   with Unicode fallback.
6. `layout` — column-aware reading-order sort.
7. `chrome` — frequency-based header/footer removal.

ML extractors (`[ml]` extra, opt-in via `--ml on|auto`):

1. `surya_layout` — region classification (Title, SectionHeader, Formula,
   Table, Picture, Caption, Footnote, …) used to upgrade heuristic blocks
   and inject missed equations.
2. `math_ocr` — Surya recognition in math mode renders each `EquationBlock`
   region to an image and fills in LaTeX (with MathML wrapper stripping).

## ML mode notes

On first run, Surya downloads ~3 GB of model weights to
`~/.cache/datalab/models/`. Subsequent runs reuse the cache.

End-to-end latency with a warm cache and an NVIDIA A40 GPU:

| Document size | Heuristic | + ML |
|---|---|---|
| 2 pages | ~0.3 s | ~5 s |
| 11 pages | ~5 s | ~25 s |
| 50 pages | ~25 s | ~2 min |

### HPC clusters and `LD_LIBRARY_PATH`

Cluster modulefiles often pre-load CUDA libraries via `LD_LIBRARY_PATH`.
PyTorch wheels ship their own CUDA runtime, and the system libraries can
shadow them — producing a confusing
`undefined symbol: cudaGetDriverEntryPointByVersion` import error.

The CLI auto-detects this and re-execs itself with `LD_LIBRARY_PATH`
cleared. If you intentionally want to keep the system CUDA libraries
visible (e.g., sharing a shell with a cluster-built ML framework), set:

```bash
PDF_CONVERTER_KEEP_LD_LIBRARY_PATH=1 pdf-convert run ...
```

Library callers (using `convert_pdf(...)` directly from Python) must clear
`LD_LIBRARY_PATH` themselves before importing if ML is enabled.

## Project layout

```
pdf-converter/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/pdf_converter/
│   ├── api.py              # convert_pdf, convert_batch
│   ├── cli.py              # pdf-convert entry point
│   ├── config.py           # pydantic settings
│   ├── ir/                 # Document / Page / Block tagged union
│   ├── extractors/
│   │   ├── heuristic/      # CPU baseline
│   │   └── ml/             # Surya-backed augmentation (optional)
│   ├── render/             # IR → markdown
│   ├── batch/              # ProcessPoolExecutor runner + manifest
│   └── utils/              # font / pdf / logging helpers
└── tests/
    ├── conftest.py         # generates a synthetic sample PDF
    └── test_*.py           # IR, renderer, invariants, CLI, ML smoke
```

## Development

```bash
pip install -e '.[dev]'

# Run the test suite (32 tests, ~2 s)
pytest

# Run the ML smoke tests (requires the [ml] extra and a GPU)
LD_LIBRARY_PATH= pytest -m ml

# Lint
ruff check src tests

# Type-check (strict)
mypy src
```

Contributions are welcome via pull requests on
[GitHub](https://github.com/Haoyu-Hu/faithful-pdf-reader). Please run
`pytest` and `ruff check` before opening one.

## License

The pdf-converter package is released under the MIT License — see
[LICENSE](LICENSE).

When installed with the `[ml]` extra, the runtime depends on
[surya-ocr](https://github.com/datalab-to/surya), which is licensed under
GPL-3.0-or-later. Choose the install path that matches your distribution
requirements.
