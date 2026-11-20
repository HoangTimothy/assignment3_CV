#!/usr/bin/env bash
set -euo pipefail
python -m cvdoc data make-splits --seed 42
python -m cvdoc data inspect --dataset smartdoc --n 8
python -m cvdoc data inspect --dataset funsd --n 8
python -m cvdoc benchmark all --split val --max-items "${MAX_ITEMS:-10}"
