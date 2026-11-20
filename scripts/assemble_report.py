"""Assemble an academic report using completed, explicitly selected experiment runs."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from cvdoc.report import build_report, markdown_table


def assemble(geometry_runs: list[Path], ocr_run: Path, destination: Path) -> None:
    for run in geometry_runs + [ocr_run]:
        build_report(run)
    rows = []
    for run in geometry_runs:
        row = json.loads((run / "summary.json").read_text())
        rows.append(
            {
                "Method": row["method"],
                "Split": row["split"],
                "Samples": row["samples"],
                "Valid": row["valid"],
                "IoU (all)": row["complete_mean_iou"],
                "NCE (valid)": row["conditional_mean_nce"],
                "Runtime ms": row["mean_runtime_ms"],
            }
        )
    ocr = pd.read_csv(ocr_run / "ocr_summary.csv")
    ocr = ocr[(ocr.denominator == "complete") & (ocr.condition != "all")]
    metadata = json.loads((ocr_run / "summary.json").read_text())
    dataset_provenance = json.loads((geometry_runs[0] / "dataset_provenance.json").read_text())
    smartdoc = dataset_provenance["smartdoc"]
    source_rows = smartdoc["source_rows"]
    counts = smartdoc["manifest_counts"]
    environment = json.loads((ocr_run / "environment.json").read_text())
    paired = pd.read_csv(ocr_run / "ocr_paired.csv")

    geometry_values = [json.loads((run / "summary.json").read_text()) for run in geometry_runs]
    base = next(row for row in geometry_values if row["method"] == "baseline")
    refined = next(row for row in geometry_values if row["method"] == "refined")
    findings = (
        f"Both geometric methods produced {refined['valid']}/{refined['samples']} valid outputs. "
        f"Mean complete-sample IoU changed from {base['complete_mean_iou']:.6f} to "
        f"{refined['complete_mean_iou']:.6f}; conditional median NCE changed from "
        f"{base['conditional_median_nce']:.6f} to {refined['conditional_median_nce']:.6f}. "
        f"Conditional p90 NCE changed from {base['conditional_p90_nce']:.6f} to "
        f"{refined['conditional_p90_nce']:.6f}. The refinement did not recover missed detections. "
    )
    for condition in sorted(ocr.condition.unique()):
        raw = ocr[(ocr.condition == condition) & (ocr.method == "E0_raw")].iloc[0]
        proposed = ocr[(ocr.condition == condition) & (ocr.method == "E3_refined")].iloc[0]
        findings += (
            f"For {condition} views, pooled ROI-assisted CER changed from "
            f"{raw.CER:.6f} to {proposed.CER:.6f}. "
        )
    findings += (
        "Repeated resampling is one plausible reason a frontal oracle can worsen recognition; "
        "the oracle is a geometric reference, not a guaranteed recognition upper bound. "
        "Observed runtimes reflect this host's workload, "
        "rather than an isolated hardware comparison."
    )

    def relative(path: Path) -> str:
        return os.path.relpath(path, destination.parent).replace("\\", "/")

    content = [
        "# Intelligent Document Detection and Perspective Correction for OCR Enhancement",
        "",
        "## Problem and Scope",
        "",
        "Detect one approximately planar printed page, estimate four "
        "corners, and rectify perspective "
        "for fixed-engine English OCR. Real-photo geometry and controlled "
        "scan OCR address separate "
        "research questions. This study tests a classical contour "
        "baseline and robust boundary-line refinement.",
        "",
        "## Theory",
        "",
        "A homography maps planar points by s[u,v,1]^T = H[x,y,1]^T. Four noncollinear corner "
        "correspondences determine eight degrees of freedom up to scale. "
        "OpenCV solves the four-point "
        "transform; there are no redundant correspondences for RANSAC to reject a wrong corner. "
        "Axis-aligned cropping cannot remove converging page edges.",
        "",
        "Native-image Polygon IoU equals intersection area divided by union area. "
        "NCE is mean Euclidean error across the four semantic vertices divided by the diagonal "
        "of the GT axis-aligned bounding box. CER and WER equal (substitutions + deletions + "
        "insertions) divided by GT character count and whitespace-token count, respectively.",
        "",
        "## Dataset",
        "",
        "SmartDoc Challenge 1 is a real smartphone/tablet geometry dataset, licensed CC BY 4.0 "
        f"(Burie et al., 2015). The prepared source contains {source_rows:,} metadata rows; "
        f"the manifest selects {sum(counts.values())} frames: {counts['dev']} development, "
        f"{counts['val']} validation and {counts['test']} test, with document "
        "identities disjoint. This project reports native-image polygon IoU, not the official "
        "document-frame score, and makes no SmartDoc transcription claim.",
        "",
        "FUNSD (Jaume, Ekenel and Thiran, 2019) contains 199 annotated scanned forms. Its official "
        "149/50 train/test split is retained; training is subdivided into 119 development and "
        "30 validation pages using seed 42. The user personally accepted its noncommercial "
        "research/education and age terms. Derived views remain private local artifacts. Published "
        "entity annotations can contain noise.",
        "",
        "## Method",
        "",
        "EXIF-aware BGR input is resized to a 1400-pixel longest side "
        "with explicit x/y transforms. "
        "Gaussian 5x5 blur precedes Canny 50/150 and 3x3 closing. Only after no valid candidate, "
        "a fixed threshold grid 25/75 then 10/40 is attempted. External "
        "contours precede an all-contour "
        "fallback. Convex four-vertex candidates are scored by normalized area, edge support, "
        "convexity and border penalty. Polygon masks and numerical candidate scores are saved.",
        "",
        "The baseline uses approxPolyDP in a fixed epsilon order. Cyclic "
        "convex ordering selects an "
        "upright top edge without consulting GT. The proposed method collects normal-compatible "
        "Canny pixels in narrow edge corridors, fits Huber lines, removes gross outliers, refits, "
        "and intersects adjacent lines. Unsupported or unstable refinements retain the baseline "
        "with a rejection diagnostic. Corners are mapped back to full resolution before warping. "
        "SmartDoc model dimensions supply only a physical ratio prior. "
        "FUNSD E2-E4 preserve the scan size.",
        "",
        "## Experimental Protocol",
        "",
        "All runs use CPU and seed 42. Only validation data informed "
        "configuration selection; final "
        "test inference uses a frozen YAML. Controlled views use a neutral-gray canvas 1.6 times "
        "the scan dimensions. Per-coordinate displacements are 3-10% for "
        "mild and 10-22% for strong; "
        "nonconvex, off-canvas and too-small polygons are rejected. The exact H, polygon, size, "
        "source identity, seed and selected masks are logged. Views are "
        "not smartphone photographs.",
        "",
        "E0 uses the raw view; E1 uses the detected axis-aligned crop; E2 "
        "uses baseline rectification; "
        "E3 uses refined rectification; E4 uses known synthesis corners only as an oracle. "
        "Primary ROI-assisted OCR uses provided entity boxes composed "
        "through each branch transform "
        "and axis-aligned crops without a second local warp. E1 can coincide with E0 under this "
        "region-assisted protocol. Fixed Tesseract language is eng, page PSM 3, ROI PSM 6, with "
        "identical 5%-height/minimum-2-pixel padding and no ROI scaling. "
        "NFKC, casefold and whitespace "
        "collapse preserve punctuation and digits. WER uses whitespace tokens. CER/WER pool edit "
        "totals over reference lengths. Failed predictions are empty strings; conditional-success "
        "summaries are separate. Views are paired by source document; no "
        "significance test is claimed.",
        "",
        "## Results",
        "",
        markdown_table(pd.DataFrame(rows)),
        "",
        "IoU (all) includes failures as zero. NCE uses the GT bounding-box diagonal and is "
        "conditional on valid predictions. Validity indicates geometric output availability, "
        "not semantic correctness.",
        "",
        f"OCR run covers {metadata['source_documents']} source documents and "
        f"{len(metadata['variants'])} conditions on split {metadata['split']}. "
        "Primary metric: ROI-assisted OCR (dataset text boxes provided).",
        "",
        markdown_table(
            ocr[["condition", "method", "entities", "failed_entities", "CER", "WER"]].rename(
                columns={
                    "condition": "Condition",
                    "method": "Method",
                    "entities": "Entities",
                    "failed_entities": "Failures",
                }
            )
        ),
        "",
        "Paired complete-sample comparison against raw recognition:",
        "",
        markdown_table(
            paired[
                [
                    "condition",
                    "method",
                    "documents",
                    "absolute_delta_CER",
                    "improved",
                    "unchanged",
                    "worsened",
                ]
            ].rename(
                columns={
                    "condition": "Condition",
                    "method": "Method",
                    "documents": "Docs",
                    "absolute_delta_CER": "Delta CER",
                    "improved": "Better",
                    "unchanged": "Same",
                    "worsened": "Worse",
                }
            )
        ),
        "",
        "![ROI-assisted OCR (dataset text boxes provided)]"
        f"({relative(ocr_run / 'figures/ocr_by_condition.png')})",
        "",
        "![SmartDoc geometry by class]"
        f"({relative(geometry_runs[-1] / 'figures/geometry_by_class.png')})",
        "",
        "## Discussion",
        "",
        findings,
        "",
        "Compare the reported conditional geometry values with complete-sample IoU before "
        "attributing an effect to refinement. A valid quadrilateral can select the wrong object "
        "or semantic orientation. The paired OCR table reports whether E3 improves or worsens "
        "recognition for each source page; improvements are not assumed. The oracle measures "
        "known-transform correction and is excluded from automatic-system success rates. "
        "Raw page OCR and heuristic reading-order scores are exploratory artifacts only.",
        "",
        "## Limitations",
        "",
        "The model assumes a single planar roughly upright page. Shadows, clutter, weak/broken "
        "edges, occlusion and low contrast can defeat contour extraction; content edges can bias "
        "line fits. The boundary cannot identify 180-degree semantic orientation. Interpolation "
        "and original scan quality limit recoverable text; inaccurate rectification can reduce "
        "recognition quality. Known ROI boxes assist localization, so these scores are not "
        "end-to-end automated text detection. Controlled scan views do not establish real-camera "
        "OCR gains. Multiple views of a document are dependent. Colab/Kaggle execution requires "
        "separate hosted verification.",
        "",
        "## Conclusion",
        "",
        "The implementation provides independently tested preprocessing, contour detection, "
        "validated corner refinement, projective warping and auditable CPU OCR experiments. "
        "The measured tables define the evidence; they do not imply guaranteed accuracy gains. "
        "Detection coverage remains a material limitation to address on development data in a "
        "future experiment with a fresh final evaluation protocol.",
        "",
        "## Reproducibility",
        "",
        f"Python {environment['python'].split()[0]}; OpenCV "
        f"{environment['packages']['opencv-python-headless']}; Tesseract "
        f"{environment['tesseract_version']}; {environment['cpu_count']} logical CPUs. "
        "Measured stage runtimes, frozen configs, source hashes, manifests and failures are stored "
        "in each run. No GPU, training or proprietary OCR API is used.",
        "",
        *[f"- [{run.name}]({relative(run / 'report.md')})" for run in geometry_runs + [ocr_run]],
        "",
        "## References",
        "",
        "- Burie et al. ICDAR2015 Competition on Smartphone Document "
        "Capture and OCR (SmartDoc), 2015. "
        "[Source and CC BY 4.0 attribution](https://github.com/jchazalon/smartdoc15-ch1-dataset).",
        "- Jaume, Ekenel and Thiran. FUNSD: A Dataset for Form "
        "Understanding in Noisy Scanned Documents, "
        "2019. [Dataset](https://guillaumejaume.github.io/FUNSD/).",
        "- Canny. A Computational Approach to Edge Detection, IEEE TPAMI, 1986.",
        "- [OpenCV shape analysis and transforms](https://docs.opencv.org/4.x/).",
        "- [Tesseract OCR documentation](https://tesseract-ocr.github.io/tessdoc/).",
        "",
        "## Team Contributions",
        "",
        "Team names, student IDs and actual contribution/commit evidence are to be supplied by "
        "the team. Codex assisted implementation and local verification. No human contributions "
        "or official submission filename have been invented.",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(content), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", action="append", type=Path, required=True)
    parser.add_argument("--ocr-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("report/report.md"))
    args = parser.parse_args()
    assemble(args.geometry_run, args.ocr_run, args.out)
