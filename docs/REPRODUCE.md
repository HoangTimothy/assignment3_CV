# Reproduce

Use an isolated environment. Windows commands replace `python` with `.venv\Scripts\python` after `python -m venv .venv`; Linux uses `source .venv/bin/activate`.

```bash
python -m pip install -c requirements-lock.txt -e '.[dev,report]'
python -m cvdoc doctor
python -m pytest -q
python -m ruff check src tests scripts
# Only after personal terms review and age confirmation:
python -m cvdoc data prepare-funsd --accept-terms --download
python -m cvdoc data prepare-smartdoc --download
python -m cvdoc data make-splits --seed 42
python -m cvdoc data inspect --dataset smartdoc --n 8
python -m cvdoc data inspect --dataset funsd --n 8
python -m cvdoc benchmark geometry --split val --method baseline --max-items 12
python -m cvdoc benchmark geometry --split val --method refined --max-items 12
python -m cvdoc benchmark ocr --split val --variants frontal,mild,strong --max-items 10
```

Freeze after validation with `python scripts/freeze_config.py --validation-run results/<geometry-val-run> --validation-run results/<ocr-val-run>`. Then run `python -m cvdoc benchmark all --split test --config configs/frozen.yaml` once. The CLI prints actual run paths. Rebuild tables/plots with `python -m cvdoc report build --run-dir results/<run-id>`.

Every run stores exact config, environment, selected/input manifests, source fingerprint, failures and status. Image IDs retain leading zeros. Run IDs use UTC; client dates are Asia/Saigon. The original benchmark preceded Git initialization; its recorded source hashes provide traceability. New runs also record the Git commit. Tests use generated mathematical fixtures, not extra research data.

The lock records the tested Python 3.12 Windows environment. Linux/Colab/Kaggle execution must be verified separately. Colab and Kaggle runners are under `notebooks/`. Export only aggregate artifacts with `python scripts/export_metrics.py results/<run-id> output/metrics.zip`. Do not redistribute the raw/derived images or entity transcripts.

PDF: install the `report` extra, then `python scripts/export_report.py report/report.md output/pdf/report.pdf`. Review the rendered output with Poppler. Markdown can also be printed to PDF from a local Markdown viewer; no course submission filename is assumed.
