"""Build final artifacts from audited runs without rerunning benchmarks or OCR."""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from assemble_report import assemble
from audit_run import audit
from export_metrics import export as export_metrics
from export_report import export as export_pdf

from cvdoc.io import write_json
from cvdoc.report import markdown_table


def build(geometry_runs: list[Path], ocr_run: Path) -> None:
    all_runs = geometry_runs + [ocr_run]
    frozen = json.loads(Path("configs/frozen.json").read_text())
    for run in all_runs:
        state = json.loads((run / "run.json").read_text())
        if state["status"] != "complete" or state["split"] != "test":
            raise ValueError(f"Expected a completed frozen test run: {run}")
        if state["config_sha256"] != frozen["config_sha256"]:
            raise ValueError(f"Run config differs from frozen settings: {run}")
        if state["source_sha256"] != frozen["source_sha256"]:
            raise ValueError(f"Run source differs from frozen source: {run}")
        audit(run)
    ocr_metadata = json.loads((ocr_run / "summary.json").read_text())
    if ocr_metadata["source_documents"] != 50 or ocr_metadata["variants"] != [
        "frontal",
        "mild",
        "strong",
    ]:
        raise ValueError("Delivery requires all 50 official FUNSD test pages and three conditions")
    if len(list((ocr_run / "masks").glob("*.png"))) != 150:
        raise ValueError("Expected 150 saved synthesis masks")
    assemble(geometry_runs, ocr_run, Path("report/report.md"))
    artifacts = Path("report/artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    index = []
    geometry_rows = []
    for run in all_runs:
        state = json.loads((run / "run.json").read_text())
        summary = json.loads((run / "summary.json").read_text())
        index.append(
            {
                "directory": str(run),
                "run": state,
                "summary": summary,
                "audit": json.loads((run / "audit.json").read_text()),
            }
        )
    for run in geometry_runs:
        summary = json.loads((run / "summary.json").read_text())
        if summary["samples"] != 120:
            raise ValueError("Delivery requires all 120 SmartDoc test frames per method")
        geometry_rows.append(
            {
                "Method": summary["method"],
                "Frames": summary["samples"],
                "Valid": summary["valid"],
                "IoU, all": summary["complete_mean_iou"],
                "NCE, valid": summary["conditional_mean_nce"],
                "NCE median, valid": summary["conditional_median_nce"],
                "Runtime ms": summary["mean_runtime_ms"],
            }
        )
        for name in ("geometry_summary.csv", "runtime_summary.csv", "failure_summary.csv"):
            path = run / name
            if path.exists():
                shutil.copy2(path, artifacts / f"{summary['method']}_{name}")
    geometry_table = pd.DataFrame(geometry_rows)
    geometry_table.to_csv(artifacts / "geometry_test.csv", index=False)
    write_json(artifacts / "run_index.json", index)
    for name in ("ocr_summary.csv", "ocr_paired.csv", "ocr_strata.csv", "ocr_runtime_summary.csv"):
        shutil.copy2(ocr_run / name, artifacts / name)
    figure_sources = {
        "ocr_by_condition.png": ocr_run / "figures/ocr_by_condition.png",
        "nce_vs_ocr_delta.png": ocr_run / "figures/nce_vs_ocr_delta.png",
        "geometry_by_class.png": geometry_runs[-1] / "figures/geometry_by_class.png",
        "geometry_by_background.png": geometry_runs[-1] / "figures/geometry_by_background.png",
    }
    report_path = Path("report/report.md")
    report = report_path.read_text(encoding="utf-8")
    for name, path in figure_sources.items():
        if not path.is_file():
            raise ValueError(f"Missing required computed figure: {path}")
        shutil.copy2(path, artifacts / name)
        previous = os.path.relpath(path, report_path.parent).replace("\\", "/")
        report = report.replace(previous, "artifacts/" + name)
    report += (
        "\n\nAggregate-only artifacts and run/source hashes: "
        "[run index](artifacts/run_index.json).\n"
    )
    report_path.write_text(report, encoding="utf-8")
    export_pdf(report_path, Path("output/pdf/report.pdf"))
    export_metrics(ocr_run, Path("output/ocr-test-metrics.zip"))
    ocr = pd.read_csv(ocr_run / "ocr_summary.csv")
    complete = ocr[(ocr.denominator == "complete") & (ocr.condition != "all")]
    Path("docs/RESULTS.md").write_text(
        "# Verified results\n\nFrozen SmartDoc test (native-image Polygon IoU; NCE uses the GT "
        "bounding-box diagonal):\n\n"
        + markdown_table(geometry_table)
        + "\n\nIoU includes failures as zero. NCE is conditional on valid predictions. "
        "Detection failures remain in the denominator; valid geometry "
        "is not necessarily correct localization.\n\n"
        "Frozen FUNSD test: 50 source scans, 150 derived views, five branches. "
        "Primary metric: **ROI-assisted OCR (dataset text boxes provided)**. "
        "E4 is an oracle and is excluded from automatic-system success rates.\n\n"
        + markdown_table(
            complete[["condition", "method", "entities", "failed_entities", "CER", "WER"]]
        )
        + "\n\nComplete-sample scores include empty predictions for failed regions. "
        "Conditional metrics, document-pooled summaries, paired changes and region-size strata "
        "are retained in the run. Views share source documents; no significance claim is made. "
        "These controlled scan results do not establish real-camera OCR gains.\n\n"
        "Exact paths, hashes and independent audits: [run "
        "index](../report/artifacts/run_index.json). "
        "See [report](../report/report.md) and [limitations](LIMITATIONS.md).\n",
        encoding="utf-8",
    )
    examples = sorted((ocr_run / "examples").glob("*strong/branches.jpg"))
    if not examples:
        raise ValueError("No saved strong-view comparison")
    example = examples[0].as_posix()
    gallery = sorted((ocr_run / "error_cases").glob("*.jpg"))
    selected_error = gallery[0].as_posix() if gallery else example
    slides = [
        ("Problem and scope", "docs/PIPELINE.md"),
        ("Two evaluation tracks and licenses", "docs/DATASETS.md"),
        ("Document-disjoint splits and frozen protocol", "configs/frozen.json"),
        (
            "Preprocessing and explicit coordinate transforms",
            str(geometry_runs[-1] / "examples/0000/montage.jpg"),
        ),
        (
            "Scored contour candidates and masks",
            str(geometry_runs[-1] / "examples/0000/candidates.json"),
        ),
        (
            "Baseline versus robust line-refined corners",
            str(geometry_runs[-1] / "examples/0000/montage.jpg"),
        ),
        (
            "Four-point homography and full-resolution warp",
            str(geometry_runs[-1] / "examples/0000/transform.json"),
        ),
        ("Controlled E0-E4 comparison; oracle labeled separately", example),
        ("SmartDoc geometry and detection failures", "report/artifacts/geometry_by_class.png"),
        (
            "Background sensitivity and wrong-object examples",
            "report/artifacts/geometry_by_background.png",
        ),
        ("ROI-assisted CER/WER and paired outcomes", "report/artifacts/ocr_by_condition.png"),
        ("Failure/degradation review and limitations", selected_error),
        ("Measured conclusions, reproducibility and contributions", "report/report.md"),
    ]
    lines = [
        "# Presentation outline",
        "",
        "Thirteen slides; use the measured tables and exact local artifacts below. "
        "Dataset-image examples remain local and must not be committed or redistributed.",
        "",
    ]
    for number, (title, path) in enumerate(slides, 1):
        if not Path(path).is_file():
            raise ValueError(f"Presentation artifact missing: {path}")
        lines += [f"## {number}. {title}", "", f"Artifact: `{Path(path).as_posix()}`.", ""]
    Path("report/presentation_outline.md").write_text("\n".join(lines), encoding="utf-8")
    print("Final delivery generated; render and inspect the PDF before marking visual QA complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", action="append", type=Path, required=True)
    parser.add_argument("--ocr-run", type=Path, required=True)
    args = parser.parse_args()
    build(args.geometry_run, args.ocr_run)
