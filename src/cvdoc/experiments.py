"""Reproducible geometry and ROI-assisted OCR experiments, with oracle isolation."""

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import yaml

from .datasets.funsd import parse_annotation
from .datasets.manifest import load_manifest, sha256
from .datasets.perturb import generate_view
from .environment import environment
from .geometry import output_size, rectify, transform_points
from .io import read_image, write_image, write_json
from .metrics import geometry_summary, normalized_corner_error, polygon_iou, text_metrics
from .ocr import recognize, require_ocr
from .pipeline import run_pipeline, save_debug
from .schemas import OCRResult, Status
from .visualize import montage, overlay

BRANCHES = ("E0_raw", "E1_crop", "E2_basic", "E3_refined", "E4_oracle")


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def new_run(config: dict, kind: str, split: str, manifest: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = Path(config["results_dir"]) / f"{stamp}_{kind}_{split}"
    run.mkdir(parents=True)
    (run / "figures").mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    write_json(run / "environment.json", environment(config))
    shutil.copy2(manifest, run / "input_manifest.csv")
    provenance = Path(config["data_dir"]) / "processed/dataset_provenance.json"
    if provenance.is_file():
        shutil.copy2(provenance, run / "dataset_provenance.json")
    write_json(
        run / "run.json",
        {
            "kind": kind,
            "split": split,
            "seed": config["seed"],
            "manifest_sha256": sha256(manifest),
            "config_sha256": sha256(run / "config.yaml"),
            "started_utc": stamp,
            "status": "running",
            "test_policy": "validation-selected config, no test tuning",
            "source_sha256": source_fingerprint(),
        },
    )
    print(f"Run directory: {run}", flush=True)
    return run


def write_rows(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty and columns:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False)


def geometry_row(record: dict, method: str, result, reference: np.ndarray, runtime: float) -> dict:
    quad = result.predicted_quad
    success = result.status == Status.OK and quad is not None
    return {
        "image_id": record["image_id"],
        "document_id": record["document_id"],
        "dataset": record.get("source", "funsd"),
        "class": record.get("class", "form"),
        "background": record.get("background", "controlled_gray"),
        "condition": record.get("condition", "real_photo"),
        "method": method,
        "status": result.status.value,
        "gt_quad": json.dumps(reference.tolist()),
        "predicted_quad": json.dumps(quad.tolist()) if quad is not None else "",
        "polygon_iou": polygon_iou(quad, reference) if success else 0.0,
        "NCE": normalized_corner_error(quad, reference) if success else None,
        "runtime_ms": runtime,
        "refinement_rejected": result.diagnostics.get("refinement_rejected", False),
        "failure_detail": result.diagnostics.get("reason", ""),
        **result.durations,
    }


def finish(run: Path, summary: dict) -> None:
    write_json(run / "summary.json", summary)
    state = json.loads((run / "run.json").read_text())
    state.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(run / "run.json", state)


def benchmark_geometry(config: dict, split: str, method: str, max_items: int | None = None) -> Path:
    data_dir = Path(config["data_dir"])
    frame = load_manifest(data_dir, "smartdoc", split)
    if max_items:
        frame = frame.head(max_items)
    run = new_run(config, f"geometry_{method}", split, data_dir / "processed/smartdoc_manifest.csv")
    write_rows(run / "selected_manifest.csv", frame.to_dict("records"))
    rows, predictions, failures = [], [], []
    for index, record in enumerate(frame.to_dict("records")):
        start = perf_counter()
        reference = np.array(json.loads(record["gt_quad"]), float)
        try:
            image = read_image(record["path"])
            size = output_size(
                None, config["output_width"], record["model_width"] / record["model_height"]
            )
            result, debug = run_pipeline(image, method, config, size, record["image_id"])
            row = geometry_row(record, method, result, reference, (perf_counter() - start) * 1000)
            predictions.append(
                {
                    "image_id": record["image_id"],
                    "method": method,
                    "status": result.status.value,
                    "predicted_quad": row["predicted_quad"],
                    "H": json.dumps(result.H.tolist()) if result.H is not None else "",
                    "diagnostics": json.dumps(result.diagnostics, default=lambda x: x.tolist()),
                }
            )
            if index < config["save_successes"] or result.status != Status.OK:
                location = run / "examples" / f"{index:04}"
                save_debug(location, image, result, debug)
                write_image(
                    location / "gt_overlay.jpg",
                    overlay(image, reference, (0, 0, 255), "SmartDoc GT P_raw"),
                )
        except (ValueError, OSError, cv2.error) as exc:
            row = {
                "image_id": record["image_id"],
                "document_id": record["document_id"],
                "class": record["class"],
                "background": record["background"],
                "method": method,
                "status": "image_read_error",
                "polygon_iou": 0.0,
                "NCE": None,
                "runtime_ms": (perf_counter() - start) * 1000,
                "failure_detail": str(exc),
            }
            predictions.append(
                {
                    "image_id": record["image_id"],
                    "method": method,
                    "status": "image_read_error",
                    "predicted_quad": "",
                    "H": "",
                }
            )
        rows.append(row)
        if row["status"] != "ok":
            failures.append(row)
        print(f"Geometry {method}: {index + 1}/{len(frame)} {row['status']}", flush=True)
    write_rows(run / "geometry_metrics.csv", rows)
    write_rows(run / "predictions.csv", predictions)
    write_rows(run / "failures.csv", failures, ["image_id", "method", "status", "failure_detail"])
    finish(
        run,
        {
            "track": "SmartDoc native-image geometry",
            "split": split,
            "method": method,
            **geometry_summary(rows),
        },
    )
    return run


def entity_corners(box: list[float]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def crop_region(
    image: np.ndarray, polygon: np.ndarray, config: dict
) -> tuple[np.ndarray | None, list[int]]:
    h, w = image.shape[:2]
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    padding = max(config["ocr_padding_min"], config["ocr_padding_ratio"] * (high[1] - low[1]))
    left, top = np.maximum([0, 0], np.floor(low - padding)).astype(int)
    right, bottom = np.minimum([w, h], np.ceil(high + padding)).astype(int)
    bounds = [int(left), int(top), int(right), int(bottom)]
    if right <= left or bottom <= top:
        return None, bounds
    return image[top:bottom, left:right].copy(), bounds


def prediction_branches(
    view_image: np.ndarray, source_size: tuple[int, int], config: dict, image_id: str
) -> tuple[dict, dict]:
    # This function intentionally accepts no synthetic GT transform or GT polygon.
    basic, basic_debug = run_pipeline(view_image, "baseline", config, source_size, image_id)
    refined, refined_debug = run_pipeline(view_image, "refined", config, source_size, image_id)
    images = {"E0_raw": (view_image, np.eye(3), "ok")}
    if basic.predicted_quad is not None:
        q = basic.predicted_quad
        h, w = view_image.shape[:2]
        x0, y0 = np.maximum([0, 0], np.floor(q.min(axis=0))).astype(int)
        x1, y1 = np.minimum([w, h], np.ceil(q.max(axis=0)) + 1).astype(int)
        if x1 > x0 and y1 > y0:
            translation = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
            images["E1_crop"] = (view_image[y0:y1, x0:x1], translation, "ok")
        else:
            images["E1_crop"] = (None, None, "invalid_quad")
    else:
        images["E1_crop"] = (None, None, basic.status.value)
    for name, result, debug in (
        ("E2_basic", basic, basic_debug),
        ("E3_refined", refined, refined_debug),
    ):
        rectification = debug["rectification"]
        images[name] = (
            rectification.image if rectification else None,
            result.H,
            result.status.value,
        )
    return images, {"E2_basic": (basic, basic_debug), "E3_refined": (refined, refined_debug)}


def oracle_branch(view, size: tuple[int, int], config: dict) -> tuple:
    # GT corners are used only here and for evaluation/ROI coordinate composition.
    oracle = rectify(view.image, view.polygon, size, config)
    return oracle.image, oracle.H, oracle.status.value


def summarize_ocr(regions: pd.DataFrame, run: Path) -> dict:
    summaries = []
    document_rows = []
    for (document, condition, method), group in regions.groupby(
        ["document_id", "condition", "method"], sort=True
    ):
        totals = group[["char_errors", "char_count", "word_errors", "word_count"]].sum()
        document_rows.append(
            {
                "document_id": document,
                "condition": condition,
                "method": method,
                **totals.to_dict(),
                "CER": totals.char_errors / totals.char_count,
                "WER": totals.word_errors / totals.word_count,
                "entities": len(group),
                "success_entities": int((group.status == "ok").sum()),
            }
        )
    per_doc = pd.DataFrame(document_rows)
    for condition in ["all"] + sorted(regions.condition.unique().tolist()):
        selected = regions if condition == "all" else regions[regions.condition == condition]
        for method, group in selected.groupby("method", sort=True):
            for denominator in ("complete", "conditional_success"):
                subset = group if denominator == "complete" else group[group.status == "ok"]
                if len(subset):
                    doc_totals = subset.groupby("document_id")[["char_errors", "char_count"]].sum()
                    doc_cer = doc_totals.char_errors / doc_totals.char_count
                else:
                    doc_cer = pd.Series(dtype=float)
                totals = subset[["char_errors", "char_count", "word_errors", "word_count"]].sum()
                summaries.append(
                    {
                        "condition": condition,
                        "method": method,
                        "denominator": denominator,
                        "oracle": method == "E4_oracle",
                        "entities": len(subset),
                        "documents": subset.document_id.nunique(),
                        "failed_entities": int((group.status != "ok").sum()),
                        **totals.to_dict(),
                        "CER": totals.char_errors / totals.char_count
                        if totals.char_count
                        else None,
                        "WER": totals.word_errors / totals.word_count
                        if totals.word_count
                        else None,
                        "mean_entity_CER": float(subset.CER.mean()) if len(subset) else None,
                        "median_entity_CER": float(subset.CER.median()) if len(subset) else None,
                        "mean_document_CER": float(doc_cer.mean()) if len(doc_cer) else None,
                        "median_document_CER": float(doc_cer.median()) if len(doc_cer) else None,
                    }
                )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(run / "ocr_summary.csv", index=False)
    per_doc.to_csv(run / "ocr_per_document.csv", index=False)
    paired = []
    for condition in sorted(per_doc.condition.unique()):
        selected = per_doc[per_doc.condition == condition]
        raw = selected[selected.method == "E0_raw"].set_index("document_id").CER
        for method in BRANCHES[1:]:
            other = selected[selected.method == method].set_index("document_id").CER
            delta = other - raw
            raw_total = (
                summary_frame[
                    (summary_frame.condition == condition)
                    & (summary_frame.method == "E0_raw")
                    & (summary_frame.denominator == "complete")
                ]
                .iloc[0]
                .CER
            )
            other_total = (
                summary_frame[
                    (summary_frame.condition == condition)
                    & (summary_frame.method == method)
                    & (summary_frame.denominator == "complete")
                ]
                .iloc[0]
                .CER
            )
            paired.append(
                {
                    "condition": condition,
                    "method": method,
                    "documents": len(delta),
                    "absolute_delta_CER": other_total - raw_total,
                    "relative_CER_reduction": (raw_total - other_total) / raw_total
                    if raw_total
                    else None,
                    "improved": int((delta < -1e-12).sum()),
                    "unchanged": int((abs(delta) <= 1e-12).sum()),
                    "worsened": int((delta > 1e-12).sum()),
                }
            )
    write_rows(run / "ocr_paired.csv", paired)
    # Region-size strata retain complete-sample deletion penalties.
    regions["region_size"] = np.where(
        regions.source_box_height < 25, "small_lt25px", "large_ge25px"
    )
    strata = (
        regions.groupby(["condition", "method", "region_size", "pipeline_status"], dropna=False)
        .agg(
            entities=("entity_id", "size"),
            char_errors=("char_errors", "sum"),
            char_count=("char_count", "sum"),
            word_errors=("word_errors", "sum"),
            word_count=("word_count", "sum"),
        )
        .reset_index()
    )
    strata["CER"] = strata.char_errors / strata.char_count
    strata["WER"] = strata.word_errors / strata.word_count
    strata.to_csv(run / "ocr_strata.csv", index=False)
    return {
        "documents": int(regions.document_id.nunique()),
        "regions": len(regions),
        "failed_region_predictions": int((regions.status != "ok").sum()),
        "metric_label": "ROI-assisted OCR (dataset text boxes provided)",
    }


def benchmark_ocr(
    config: dict, split: str, variants: list[str], max_items: int | None = None
) -> Path:
    require_ocr(config)
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    if (
        not variants
        or len(set(variants)) != len(variants)
        or any(v not in ("frontal", "mild", "strong", "stress") for v in variants)
    ):
        raise ValueError("Variants must be unique frontal,mild,strong[,stress]")
    data_dir = Path(config["data_dir"])
    frame = load_manifest(data_dir, "funsd", split)
    if max_items:
        frame = frame.head(max_items)
    if any(not Path(p).is_file() for p in frame.annotation_path):
        raise ValueError("Missing FUNSD annotations; prepare the complete source archive")
    run = new_run(config, "ocr", split, data_dir / "processed/funsd_manifest.csv")
    write_rows(run / "selected_manifest.csv", frame.to_dict("records"))
    regions, geometry, predictions, failures, derived, page_rows = [], [], [], [], [], []
    excluded = {"empty_text": 0, "invalid_box": 0, "out_of_page_box": 0}
    with ThreadPoolExecutor(max_workers=config["ocr_workers"]) as executor:
        for index, record in enumerate(frame.to_dict("records")):
            page = read_image(record["path"])
            entities, counters = parse_annotation(Path(record["annotation_path"]))
            for key, value in counters.items():
                excluded[key] += value
            valid_entities = []
            for entity in entities:
                if entity["box"][2] > page.shape[1] or entity["box"][3] > page.shape[0]:
                    excluded["out_of_page_box"] += 1
                else:
                    valid_entities.append(entity)
            if not valid_entities:
                failures.append(
                    {"document_id": record["document_id"], "status": "no_valid_entities"}
                )
            for condition in variants:
                view = generate_view(page, record["document_id"], condition, config["seed"])
                mask_path = run / "masks" / f"{view.metadata['view_id']}.png"
                write_image(mask_path, view.mask)
                derived.append(
                    {
                        "page_mask_path": str(mask_path),
                        **view.metadata,
                        "split": split,
                        "H_page_to_view": json.dumps(view.H_page_to_view.tolist()),
                        "polygon": json.dumps(view.polygon.tolist()),
                        "source_path": record["path"],
                    }
                )
                source_size = (page.shape[1], page.shape[0])
                images, results = prediction_branches(
                    view.image, source_size, config, view.metadata["view_id"]
                )
                images["E4_oracle"] = oracle_branch(view, source_size, config)
                for method, (result, debug) in results.items():
                    geo = geometry_row(
                        {**record, "condition": condition, "image_id": view.metadata["view_id"]},
                        method,
                        result,
                        view.polygon,
                        sum(result.durations.values()),
                    )
                    geometry.append(geo)
                    predictions.append(
                        {
                            "view_id": view.metadata["view_id"],
                            "method": method,
                            "predicted_quad": geo["predicted_quad"],
                            "status": result.status.value,
                            "H_view_to_rect": json.dumps(result.H.tolist())
                            if result.H is not None
                            else "",
                            "diagnostics": json.dumps(
                                result.diagnostics, default=lambda x: x.tolist()
                            ),
                        }
                    )
                    if index < config["save_successes"] or result.status != Status.OK:
                        save_debug(
                            run / "examples" / view.metadata["view_id"] / method,
                            view.image,
                            result,
                            debug,
                        )
                if index < config["save_successes"] or any(
                    r.status != Status.OK for r, _ in results.values()
                ):
                    sample = run / "examples" / view.metadata["view_id"]
                    write_image(sample / "mask.png", view.mask)
                    write_json(sample / "synthesis.json", view.metadata)
                    montage(
                        sample / "branches.jpg",
                        [
                            (name + " (oracle)" if name == "E4_oracle" else name, images[name][0])
                            for name in BRANCHES
                        ],
                    )
                page_reference = " ".join(
                    e["text"]
                    for e in sorted(
                        valid_entities, key=lambda e: (e["box"][1], e["box"][0], e["entity_id"])
                    )
                )
                # Cache exact duplicate crops only; E0/E1 region results often coincide.
                cache, tasks, pending = {}, [], {}
                for method in BRANCHES:
                    branch_image, H, status = images[method]
                    page_ocr = (
                        recognize(branch_image, config)
                        if branch_image is not None
                        else OCRResult(Status.OCR_ERROR)
                    )
                    page_rows.append(
                        {
                            "document_id": record["document_id"],
                            "view_id": view.metadata["view_id"],
                            "condition": condition,
                            "method": method,
                            "status": status,
                            "reference_text": page_reference,
                            "prediction_text": page_ocr.text,
                            **{
                                f"exploratory_page_{k}": v
                                for k, v in text_metrics(page_reference, page_ocr.text).items()
                            },
                            "runtime_ms": page_ocr.elapsed_ms,
                        }
                    )
                    composition = H @ view.H_page_to_view if H is not None else None
                    for entity in valid_entities:
                        box = entity["box"]
                        crop, bounds, detail = None, [], ""
                        if branch_image is not None and composition is not None:
                            try:
                                polygon = transform_points(entity_corners(box), composition)
                                crop, bounds = crop_region(branch_image, polygon, config)
                            except ValueError as exc:
                                detail = str(exc)
                        row = {
                            "dataset": "FUNSD-derived",
                            "document_id": record["document_id"],
                            "view_id": view.metadata["view_id"],
                            "condition": condition,
                            "entity_id": entity["entity_id"],
                            "method": method,
                            "reference_text": entity["text"],
                            "pipeline_status": status,
                            "bbox": json.dumps(bounds),
                            "source_bbox": json.dumps(box),
                            "source_box_height": box[3] - box[1],
                            "failure_detail": detail,
                        }
                        if crop is None:
                            tasks.append((row, None))
                        else:
                            key = hashlib.sha256(
                                crop.tobytes() + str(crop.shape).encode()
                            ).hexdigest()
                            if key not in pending:
                                pending[key] = executor.submit(recognize, crop, config, True)
                            tasks.append((row, key))
                for row, key in tasks:
                    if key is None:
                        output = OCRResult(
                            Status.OCR_ERROR, detail="Pipeline failure or empty/outside ROI"
                        )
                    else:
                        if key not in cache:
                            cache[key] = pending[key].result()
                        output = cache[key]
                    row.update(
                        prediction_text=output.text,
                        status=output.status.value,
                        runtime_ms=output.elapsed_ms,
                        engine_version=output.engine_version,
                        psm=config["ocr_roi_psm"],
                        failure_detail=row["failure_detail"] or output.detail,
                        **text_metrics(row["reference_text"], output.text),
                    )
                    regions.append(row)
                    if output.status != Status.OK:
                        failures.append(row)
                print(
                    f"OCR {index + 1}/{len(frame)} {condition}: "
                    f"{len(valid_entities)} entities x 5 branches",
                    flush=True,
                )
                # Checkpoints preserve auditable progress if a long CPU run is interrupted.
                write_rows(run / "ocr_regions.csv", regions)
                write_rows(run / "geometry_metrics.csv", geometry)
    if not regions:
        raise ValueError(f"No valid published entity annotations; see {run}")
    write_rows(run / "ocr_metrics.csv", regions)
    write_rows(run / "ocr_pages.csv", page_rows)
    write_rows(run / "predictions.csv", predictions)
    write_rows(
        run / "failures.csv",
        failures,
        ["document_id", "view_id", "entity_id", "method", "status", "failure_detail"],
    )
    write_rows(run / "derived_manifest.csv", derived)
    # Canonical derived metadata is cumulative and idempotent by source/view/split/seed.
    manifest_path = data_dir / "processed/funsd_derived_manifest.csv"
    new = pd.DataFrame(derived)
    if manifest_path.exists():
        new = pd.concat([pd.read_csv(manifest_path), new], ignore_index=True)
        new = new.drop_duplicates(["source_document_id", "condition", "seed", "split"], keep="last")
    new.to_csv(manifest_path, index=False)
    summary = summarize_ocr(pd.DataFrame(regions), run)
    finish(
        run,
        {
            "track": "Controlled FUNSD ROI-assisted OCR",
            "split": split,
            "variants": variants,
            **summary,
            "excluded_annotations": excluded,
            "source_documents": len(frame),
        },
    )
    return run
