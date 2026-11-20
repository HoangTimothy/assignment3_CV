#!/usr/bin/env bash
set -euo pipefail
: "${1:?Usage: run_test.sh FROZEN_CONFIG_YAML}"
python -m cvdoc benchmark all --split test --config "$1"
