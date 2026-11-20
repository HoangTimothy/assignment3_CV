"""Freeze validation-selected settings before executing the final test manifests."""

import argparse
from pathlib import Path

import yaml

from cvdoc.config import load_config
from cvdoc.datasets.manifest import sha256
from cvdoc.experiments import source_fingerprint
from cvdoc.io import write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("configs/frozen.yaml"))
    parser.add_argument("--validation-run", type=Path, action="append", required=True)
    args = parser.parse_args()
    for run in args.validation_run:
        if not (run / "summary.json").is_file():
            parser.error(f"Validation run is incomplete: {run}")
        import json

        metadata = json.loads((run / "run.json").read_text())
        if metadata["split"] != "val" or metadata["status"] != "complete":
            parser.error("Only completed validation runs may justify a freeze")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        parser.error(f"Refusing to overwrite existing frozen config: {args.out}")
    args.out.write_text(yaml.safe_dump(load_config(args.config), sort_keys=True), encoding="utf-8")
    write_json(
        args.out.with_suffix(".json"),
        {
            "config_sha256": sha256(args.out),
            "source_sha256": source_fingerprint(),
            "validation_runs": [str(p) for p in args.validation_run],
            "rationale": "Retain specified defaults plus fixed lower-Canny "
            "fallback for weak edges. "
            "No test results were used to select settings. PSM 3/6 and normalization fixed.",
            "no_test_tuning": True,
        },
    )
    print(args.out)
