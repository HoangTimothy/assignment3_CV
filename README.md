# assignment3_CV ? Document detection and OCR

CPU-only Python project for detecting a planar document, estimating corners, correcting perspective, and evaluating Tesseract OCR. SmartDoc real-photo geometry and controlled FUNSD OCR are separate evaluation tracks.

## Handoff: completed work

- Implemented the `cvdoc` package and CLI: image I/O with EXIF correction, preprocessing, scored contour detection, validated corner ordering, robust line refinement, full-resolution homography, rectification, OCR, metrics and reports.
- Implemented secure public-dataset preparation, document-disjoint SmartDoc sampling, official FUNSD splits, deterministic perspective views, provenance and source hashes.
- Completed local validation and frozen test benchmarks. All result rows were audited against source annotations and metrics were recomputed.
- Verified **85 tests**, Ruff lint/format, editable installation, wheel installation, CLI demos and PDF export on Windows/Python 3.12.10/OpenCV 4.11/Tesseract 5.4.
- Added CPU Colab/Kaggle runners. An offline GitHub Actions workflow is preserved locally at `.local/ci/tests.yml`; the current GitHub token lacks `workflow` scope, so it is not published. Hosted execution has not been verified.

| Frozen test | Verified result |
| --- | --- |
| SmartDoc | 120 frames per method; 55 valid outputs and 65 detection failures per method |
| Refined geometry | Native-image IoU 0.423889 over all frames; mean NCE 0.057602 over valid outputs |
| FUNSD | 50 pages, 150 derived views, 34,050 region/branch predictions audited |
| ROI-assisted CER, raw ? refined | Frontal 0.174574 ? 0.190507; mild 0.489757 ? 0.212181; strong 0.659323 ? 0.238862 |

OCR scores use published entity boxes for region assistance. They do not measure automatic text localization or real-camera OCR improvement. Frontal recognition worsened; refinement did not recover missed SmartDoc detections. Detailed results and caveats are in [results](docs/RESULTS.md), [failure analysis](docs/FAILURE_ANALYSIS.md) and the [report](report/report.md). Small aggregate tables, plots, hashes and audits are committed under `report/artifacts/`.

## Handoff: remaining work

1. Add actual team names, student IDs and contributions to the report; no identities have been invented.
2. Review the course instruction PDFs and required submission filename/format. Those checks were not completed during the original implementation. Course reference files, if present locally, are not published.
3. Execute the notebooks on Colab/Kaggle. To enable CI, restore `.local/ci/tests.yml` to `.github/workflows/tests.yml` and push using a credential with `workflow` scope, or create the workflow through GitHub. Local notebook structure/code syntax passed; hosted environments remain unverified.
4. Review detection misses and wrong-object selection on **development/validation data**. Keep the existing test results as the frozen baseline; use a fresh evaluation protocol for subsequent tuning.
5. Before submission, regenerate the PDF if the report changes and inspect every rendered page. The original four-page PDF was generated and visually checked locally.

## Setup and checks

Windows PowerShell (no environment activation required):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -c requirements-lock.txt -e '.[dev,report]'
.venv\Scripts\python -m cvdoc doctor
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m ruff check src tests scripts
```

Linux: create/activate a virtual environment, install `tesseract-ocr` and `tesseract-ocr-eng`, then use `python` for the same commands. The lock records the tested Windows/Python 3.12 environment. `doctor` exits 0 when OCR is ready and 1 when unavailable; malformed CLI commands exit 2.

## Continue the project

```bash
# Review FUNSD terms and personally confirm age 18+ before accepting:
python -m cvdoc data prepare-funsd --accept-terms --download
python -m cvdoc data prepare-smartdoc --download
python -m cvdoc data make-splits --seed 42
python -m cvdoc benchmark geometry --split val --method refined --max-items 12
python -m cvdoc benchmark ocr --split val --variants frontal,mild,strong --max-items 10
python -m cvdoc infer --input path/to/image.png --method refined --out results/demo --save-stages
```

Defaults: `src/cvdoc/defaults.yaml`. Final benchmark settings and rationale: `configs/frozen.yaml` and `configs/frozen.json`. Historical run IDs and source hashes: `report/artifacts/run_index.json`. Local full runs remain under `results/`; they are not downloaded with a clone. Do not rerun the final test split to choose parameters.

Rebuild a report from an existing local run with `python -m cvdoc report build --run-dir results/<run-id>`. Export the academic PDF with `python scripts/export_report.py report/report.md output/pdf/report.pdf`. See [reproduction commands](docs/REPRODUCE.md), [pipeline](docs/PIPELINE.md), [dataset terms](docs/DATASETS.md), and the [presentation outline](report/presentation_outline.md).

## Repository boundaries

Keep source, tests, configs, runners, report source/references and small aggregate findings. `.gitignore` excludes datasets/derived images/transcripts (`data/`), full runs and debug images (`results/`), generated PDFs/previews/ZIP exports (`output/`), virtual environments, caches, builds, secrets and local course/prompt material. The generated mathematical fixture under `tests/fixtures/` is safe to version and is not a research dataset.

Original prompt/spec documents are preserved locally under `.local/reference/` after cleanup. They are not needed to run the package. Raw public datasets must not be uploaded: FUNSD requires personal acceptance of its noncommercial research/education and age terms; SmartDoc requires author attribution. Dataset sources and citations are recorded in [DATASETS.md](docs/DATASETS.md) and [references.bib](report/references.bib).
