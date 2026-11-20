"""Recompute aggregates from source rows; never fill missing results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .datasets.perturb import generate_view
from .experiments import prediction_branches, summarize_ocr
from .io import read_image, write_json
from .metrics import geometry_summary
from .visualize import montage


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value) -> str:
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value).replace("|", "/").replace("\n", " ")

    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    return "\n".join(
        [header, rule]
        + [
            "| " + " | ".join(cell(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def validate_rows(run: Path, state: dict) -> None:
    if state.get("status") != "complete":
        raise ValueError("Report requires a completed run; incomplete checkpoints are not results")
    selected_path = run / "selected_manifest.csv"
    if not selected_path.is_file():
        raise ValueError("Report requires selected_manifest.csv")
    selected = pd.read_csv(selected_path, dtype={"image_id": str, "document_id": str})
    geometry = pd.read_csv(
        run / "geometry_metrics.csv", dtype={"image_id": str, "document_id": str}
    )
    expected_ids = set(selected.image_id.astype(str))
    if state["kind"].startswith("geometry"):
        if len(geometry) != len(selected) or set(geometry.image_id.astype(str)) != expected_ids:
            raise ValueError("Geometry result rows do not match selected manifest")
        keys = ["image_id", "method"]
    else:
        regions = pd.read_csv(
            run / "ocr_regions.csv",
            keep_default_na=False,
            dtype={"document_id": str, "entity_id": str},
        )
        if set(regions.document_id) != set(selected.document_id.astype(str)):
            raise ValueError("OCR document coverage differs from selected manifest")
        if regions.duplicated(["view_id", "entity_id", "method"]).any():
            raise ValueError("Duplicate OCR region predictions")
        if not (regions.groupby(["view_id", "entity_id"]).method.nunique() == 5).all():
            raise ValueError("Unpaired OCR regions; expected E0-E4")
        for column in ("char_errors", "char_count", "word_errors", "word_count", "CER", "WER"):
            if not np.isfinite(pd.to_numeric(regions[column])).all():
                raise ValueError(f"Nonfinite OCR metric: {column}")
        if (pd.to_numeric(regions.char_count) <= 0).any():
            raise ValueError("Empty reference included in primary metrics")
        keys = ["image_id", "method"]
    if geometry.duplicated(keys).any():
        raise ValueError("Duplicate geometric result rows")
    good = geometry[geometry.status == "ok"]
    if not np.isfinite(good[["polygon_iou", "NCE", "runtime_ms"]].astype(float)).all().all():
        raise ValueError("Successful geometry rows contain nonfinite metrics")


def build_report(run: Path) -> Path:
    if not (run / "run.json").is_file() or not (run / "summary.json").is_file():
        raise ValueError("Missing completed run metadata or summary; execute a benchmark first")
    state = json.loads((run / "run.json").read_text())
    validate_rows(run, state)
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    geometry = pd.read_csv(
        run / "geometry_metrics.csv", dtype={"image_id": str, "document_id": str}
    )
    lines = [
        "# Verified experiment results",
        "",
        f"Track: {state['kind']}; split: {state['split']}.",
        "",
        "Configuration, environment, input manifest and provenance are stored beside this report.",
        "",
    ]
    if state["kind"].startswith("geometry"):
        summary = geometry_summary(geometry.to_dict("records"))
        write_json(
            run / "summary.json",
            {
                "track": "SmartDoc native-image geometry",
                "split": state["split"],
                "method": geometry.method.iloc[0],
                **summary,
            },
        )
        groups = []
        for field in ("class", "background"):
            for value, group in geometry.groupby(field):
                groups.append(
                    {"group": field, "value": value, **geometry_summary(group.to_dict("records"))}
                )
        pd.DataFrame(groups).to_csv(run / "geometry_summary.csv", index=False)
        lines += [
            "Native-image Polygon IoU; NCE uses the GT bounding-box diagonal.",
            "Missing predictions count as IoU=0; NCE averages are conditional on "
            "valid predictions.",
            "",
            markdown_table(
                pd.DataFrame(groups)[
                    [
                        "group",
                        "value",
                        "samples",
                        "valid",
                        "complete_mean_iou",
                        "conditional_mean_nce",
                    ]
                ]
            ),
            "",
        ]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, column, title in zip(
            axes, ("polygon_iou", "NCE"), ("Native-image Polygon IoU", "Conditional NCE")
        ):
            values = geometry.copy()
            values[column] = pd.to_numeric(values[column], errors="coerce")
            values.groupby("class")[column].mean().plot.bar(ax=ax)
            ax.set_title(title)
            ax.set_ylabel(column)
        fig.tight_layout()
        fig.savefig(figures / "geometry_by_class.png", dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, column in zip(axes, ("polygon_iou", "NCE")):
            geometry.groupby("background")[column].mean().plot.bar(ax=ax)
            ax.set_title("Native-image geometry by background")
            ax.set_ylabel(column)
        fig.tight_layout()
        fig.savefig(figures / "geometry_by_background.png", dpi=140)
        plt.close(fig)
    else:
        regions = pd.read_csv(
            run / "ocr_regions.csv",
            keep_default_na=False,
            dtype={"document_id": str, "entity_id": str},
        )
        aggregate = summarize_ocr(regions, run)
        previous = json.loads((run / "summary.json").read_text())
        write_json(run / "summary.json", {**previous, **aggregate})
        table = pd.read_csv(run / "ocr_summary.csv")
        selected = table[(table.denominator == "complete") & (table.condition != "all")]
        lines += [
            "Primary: ROI-assisted OCR (dataset text boxes provided). These are "
            "controlled views of scans.",
            "E4 is an oracle reference. Failed regions retain empty predictions "
            "and deletion penalties.",
            "Page metrics use a top/left entity ordering heuristic and are exploratory.",
            "",
            markdown_table(
                selected[["condition", "method", "entities", "failed_entities", "CER", "WER"]]
            ),
            "",
        ]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for ax, column in zip(axes, ("CER", "WER")):
            pivot = selected.pivot(index="condition", columns="method", values=column)
            pivot.plot.bar(ax=ax)
            ax.set_title(f"ROI-assisted OCR (dataset text boxes provided): {column}")
            ax.set_ylabel(column)
        fig.tight_layout()
        fig.savefig(figures / "ocr_by_condition.png", dpi=140)
        plt.close(fig)
        documents = pd.read_csv(run / "ocr_per_document.csv", dtype={"document_id": str})
        pairs = documents.pivot(
            index=["document_id", "condition"], columns="method", values="CER"
        ).reset_index()
        pairs["delta_CER"] = pairs.E3_refined - pairs.E0_raw
        points = geometry[geometry.method == "E3_refined"].merge(
            pairs, on=["document_id", "condition"]
        )
        points["NCE"] = pd.to_numeric(points.NCE, errors="coerce")
        points = points.dropna(subset=["NCE", "delta_CER"])
        if len(points) >= 3:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(points.NCE, points.delta_CER, s=20)
            ax.set(
                xlabel="NCE",
                ylabel="CER E3 - E0",
                title="ROI-assisted OCR (dataset text boxes provided)",
            )
            fig.tight_layout()
            fig.savefig(figures / "nce_vs_ocr_delta.png", dpi=140)
            plt.close(fig)
        timing = regions.groupby("method").runtime_ms.agg(["count", "mean", "median"])
        timing.to_csv(run / "ocr_runtime_summary.csv")
        worse = pairs[pairs.delta_CER > 0].copy()
        worse.to_csv(run / "worsened_documents.csv", index=False)
        gallery = []
        if len(worse):
            settings = yaml.safe_load((run / "config.yaml").read_text())
            manifest = pd.read_csv(run / "selected_manifest.csv", dtype={"document_id": str})
            for row in worse.sort_values("delta_CER", ascending=False).head(5).to_dict("records"):
                record = manifest[manifest.document_id == row["document_id"]].iloc[0]
                page = read_image(record["path"])
                view = generate_view(page, row["document_id"], row["condition"], settings["seed"])
                images, _ = prediction_branches(
                    view.image, (page.shape[1], page.shape[0]), settings, view.metadata["view_id"]
                )
                case_path = run / "error_cases" / f"{view.metadata['view_id']}.jpg"
                montage(
                    case_path,
                    [(name, images[name][0]) for name in ("E0_raw", "E2_basic", "E3_refined")],
                )
                gallery.append(
                    {
                        "document_id": row["document_id"],
                        "condition": row["condition"],
                        "cause": "observed_E3_CER_worse_than_E0",
                        "delta_CER": row["delta_CER"],
                        "figure": str(case_path.relative_to(run)),
                    }
                )
            pd.DataFrame(gallery).to_csv(run / "error_case_gallery.csv", index=False)
            lines += [
                "## Recognition degradation cases",
                "",
                "Cases selected by measured E3-minus-E0 CER; no new tuning is performed.",
                "",
            ]
            for case in gallery:
                lines += [f"![Measured CER change {case['delta_CER']:.4f}]({case['figure']})", ""]
    durations = [c for c in geometry if c.endswith("_ms")]
    if durations:
        times = (
            geometry[durations]
            .apply(pd.to_numeric, errors="coerce")
            .mean()
            .rename("mean_ms")
            .reset_index()
        )
        times.columns = ["stage", "mean_ms"]
        times.to_csv(run / "runtime_summary.csv", index=False)
        lines += ["## Measured runtime", "", markdown_table(times), ""]
    failures = pd.read_csv(run / "failures.csv", keep_default_na=False)
    if len(failures):
        counts = failures.groupby("status").size().reset_index(name="count")
        counts.to_csv(run / "failure_summary.csv", index=False)
        lines += ["## Observed failures", "", markdown_table(counts), ""]
    galleries = sorted((run / "examples").rglob("montage.jpg"))
    lines += ["## Case gallery", ""]
    for path in galleries[:8]:
        lines += [f"![{path.parent.name}]({path.relative_to(run).as_posix()})", ""]
    lines += [
        "No significance claim is made. Views share source documents; "
        "frame-level independence is not assumed.",
        "Interpolation, low resolution, wrong boundaries, orientation and "
        "annotation noise can affect recognition.",
    ]
    report = run / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
