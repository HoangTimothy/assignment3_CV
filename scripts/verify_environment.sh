#!/usr/bin/env bash
set -euo pipefail
python -m cvdoc --help
python -m cvdoc doctor
python -m pytest -q -m 'not data and not ocr'
python -m ruff check src tests scripts
