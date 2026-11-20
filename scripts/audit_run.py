"""Independently audit benchmark coverage and recompute metrics from saved predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cvdoc.corners import validate_quad
from cvdoc.datasets.funsd import parse_annotation
from cvdoc.datasets.manifest import sha256
from cvdoc.experiments import BRANCHES
from cvdoc.io import write_json
from cvdoc.metrics import geometry_summary, normalized_corner_error, polygon_iou, text_metrics
from cvdoc.report import validate_rows


def close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=1e-9, atol=1e-10):
        raise ValueError(f"Metric mismatch {label}: {actual} != {expected}")


def audit(run: Path) -> dict:
    state = json.loads((run / "run.json").read_text())
    validate_rows(run, state)
    config = yaml.safe_load((run / "config.yaml").read_text())
    if sha256(run / "config.yaml") != state["config_sha256"]:
        raise ValueError("Run config was modified")
    if sha256(run / "input_manifest.csv") != state["manifest_sha256"]:
        raise ValueError("Input manifest was modified")
    selected = pd.read_csv(
        run / "selected_manifest.csv", dtype={"document_id": str, "image_id": str}
    )
    selected_by_id = selected.set_index("document_id")
    geometry = pd.read_csv(
        run / "geometry_metrics.csv", dtype={"document_id": str, "image_id": str}
    )
    geometric_checks = 0
    for row in geometry.to_dict("records"):
        if row["status"] != "ok":
            if row["polygon_iou"] != 0 or not pd.isna(row["NCE"]):
                raise ValueError("Failure lacks zero IoU / absent conditional NCE")
            continue
        prediction = np.array(json.loads(row["predicted_quad"]), float)
        reference = np.array(json.loads(row["gt_quad"]), float)
        original = selected_by_id.loc[row["document_id"]]
        if isinstance(original, pd.DataFrame):
            original = original[original.image_id == row["image_id"]].iloc[0]
        if state["kind"] == "ocr":
            dimensions = (round(1.6 * original.width), round(1.6 * original.height))
        else:
            dimensions = (int(original.width), int(original.height))
        valid, reason = validate_quad(prediction, dimensions, config)
        if not valid:
            raise ValueError(f"Invalid successful prediction: {reason}")
        close(row["polygon_iou"], polygon_iou(prediction, reference), "polygon IoU")
        close(row["NCE"], normalized_corner_error(prediction, reference), "NCE")
        geometric_checks += 1
    output = {
        "coverage": "verified",
        "geometry_rows": len(geometry),
        "valid_quads_checked": geometric_checks,
        "config_and_manifest_hashes": "verified",
    }
    if state["kind"].startswith("geometry"):
        expected = geometry_summary(geometry.to_dict("records"))
        summary = json.loads((run / "summary.json").read_text())
        for key, value in expected.items():
            if value is not None:
                close(summary[key], value, key)
    else:
        regions = pd.read_csv(
            run / "ocr_regions.csv",
            keep_default_na=False,
            dtype={"document_id": str, "entity_id": str},
        )
        summary = json.loads((run / "summary.json").read_text())
        expected_keys = set()
        references = {}
        for record in selected.to_dict("records"):
            entities, _ = parse_annotation(Path(record["annotation_path"]))
            for entity in entities:
                if entity["box"][2] > record["width"] or entity["box"][3] > record["height"]:
                    continue
                references[(record["document_id"], entity["entity_id"])] = entity["text"]
                for condition in summary["variants"]:
                    for method in BRANCHES:
                        expected_keys.add(
                            (record["document_id"], condition, entity["entity_id"], method)
                        )
        actual_keys = set()
        for row in regions.to_dict("records"):
            key = (row["document_id"], row["condition"], row["entity_id"], row["method"])
            actual_keys.add(key)
            if references[(row["document_id"], row["entity_id"])] != row["reference_text"]:
                raise ValueError("Reference differs from published annotation")
            measured = text_metrics(row["reference_text"], row["prediction_text"])
            for field, value in measured.items():
                close(row[field], value, field)
            if row["status"] != "ok" and row["prediction_text"]:
                raise ValueError("Failed prediction contains text")
        if expected_keys != actual_keys or len(actual_keys) != len(regions):
            raise ValueError("Missing, duplicated or extra entity/condition/branch rows")
        aggregates = pd.read_csv(run / "ocr_summary.csv")
        for row in aggregates.to_dict("records"):
            subset = regions[regions.method == row["method"]]
            if row["condition"] != "all":
                subset = subset[subset.condition == row["condition"]]
            if row["denominator"] == "conditional_success":
                subset = subset[subset.status == "ok"]
            if len(subset) != row["entities"]:
                raise ValueError("Aggregate entity denominator mismatch")
            if len(subset):
                for rate, error, length in (
                    ("CER", "char_errors", "char_count"),
                    ("WER", "word_errors", "word_count"),
                ):
                    close(row[rate], subset[error].sum() / subset[length].sum(), rate)
        output.update(
            ocr_regions=len(regions),
            exact_annotation_pairing="verified",
            text_edit_counts_and_pooled_aggregates="verified",
        )
    write_json(run / "audit.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run), indent=2))
