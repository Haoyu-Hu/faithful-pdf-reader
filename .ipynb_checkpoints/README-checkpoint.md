# pdf-converter

A faithful PDF → Markdown converter built around a typed intermediate representation, with a heuristic CPU baseline and an optional GPU-accelerated ML augmentation layer (Surya).

Designed for batch conversion of mixed corpora (academic papers, technical docs, reports, medical/biological documents) where structural fidelity, math/formula preservation, and verbatim handling of specialized terminology all matter.

## Status

Pre-alpha. v1 ships with:

- Hybrid heuristic + optional Surya ML extractors driven by a shared IR
- Batch / parallel processing with progress and resumability
- Structural invariants enforced by a property-test suite

See `/u/hhu4/.claude/plans/we-are-developing-a-soft-fiddle.md` for the full design.

## Install

```bash
pip install -e .            # heuristic-only (CPU, MIT-licensed)
pip install -e '.[ml]'      # + Surya / torch (see "ML mode" below)
pip install -e '.[dev]'     # + test & lint tooling
pip install -e '.[all]'     # everything
```

The base package is MIT-licensed. The `[ml]` extra pulls **surya-ocr** which is GPL-3.0-or-later — your runtime is then subject to GPL-3 terms. Installing without the `[ml]` extra keeps everything pure MIT.

## Quick start

```bash
# single PDF, heuristic-only
pdf-convert run path/to/doc.pdf -o outputs/

# directory, parallel (heuristic only)
pdf-convert run path/to/dir/ -o outputs/ --workers 4

# with ML augmentation (requires GPU + ml extras)
pdf-convert run path/to/dir/ -o outputs/ --ml on
```

Library API:

```python
from pdf_converter import convert_pdf, convert_batch

doc = convert_pdf("paper.pdf")
print(doc.markdown)
for block in doc.iter_blocks():
    print(block.kind, block.bbox, block.provenance.extractor)

convert_batch(["a.pdf", "b.pdf"], out_dir="outputs/", workers=4)
```

## ML mode

`--ml on` enables Surya layout detection and LaTeX equation OCR. On first run, Surya downloads ~3 GB of models to `~/.cache/datalab/models/`.

**GPU is required for reasonable speed**. With an A40 the recognition + layout pipeline adds ~25-60 s per 10-page document (model load amortized across files in a batch).

### HPC clusters and `LD_LIBRARY_PATH`

Cluster modulefiles often pre-load CUDA libraries via `LD_LIBRARY_PATH`. PyTorch wheels ship their own CUDA runtime, and the system libs can shadow them, producing a confusing `undefined symbol: cudaGetDriverEntryPointByVersion` import error.

The CLI auto-detects this and **re-execs itself with `LD_LIBRARY_PATH` cleared**. If you need to keep the system libraries (e.g., you're using both pdf-convert and a cluster-built ML framework in the same shell), set:

```bash
PDF_CONVERTER_KEEP_LD_LIBRARY_PATH=1 pdf-convert run ...
```

Library callers (`convert_pdf(...)` from Python) must clear `LD_LIBRARY_PATH` themselves before invocation when calling with ML enabled.

## Design

```
PDF -> load (PyMuPDF) -> extract (heuristic | ml per block-type)
    -> IR (Document/Page/Block, tagged union, bbox + provenance + reading_order)
    -> reading-order sort
    -> post-process (header/footer freq filter, math normalize)
    -> render (markdown builder)
    -> write (.md + sidecar images/)
```

Block kinds: `text` · `heading` · `list` · `table` · `figure` · `equation` · `code` · `footnote`.

ML extractors live in `pdf_converter.extractors.ml` and are lazy-loaded per-process. They refine the heuristic IR rather than replacing it, so any reason ML is unavailable (no GPU, no extras installed) falls back cleanly to heuristic-only output.

## License

MIT for the core package. See `LICENSE`. Note that `[ml]` brings in GPL-3.0-licensed dependencies (Surya); see "Install" above.
