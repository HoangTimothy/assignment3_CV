#!/usr/bin/env bash
set -euo pipefail
: "${1:?Usage: run_demo.sh INPUT_IMAGE [OUTPUT_DIR]}"
python -m cvdoc infer --input "$1" --method refined --out "${2:-results/demo}" --save-stages
