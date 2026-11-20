"""Prediction-only pipeline: no GT corner arguments or annotation access."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .corners import refine_corners, validate_quad
from .detect import detect_document
from .geometry import output_size, rectify, transform_points
from .io import read_image, write_image, write_json
from .ocr import recognize
from .preprocess import preprocess
from .schemas import PipelineResult, Status
from .visualize import montage, overlay


def run_pipeline(
    image: np.ndarray,
    method: str,
    config: dict,
    size: tuple[int, int] | None = None,
    image_id: str = "image",
) -> tuple[PipelineResult, dict]:
    if method not in ("baseline", "refined"):
        raise ValueError(f"Unknown prediction method {method}")
    started = perf_counter()
    processed = preprocess(image, config)
    durations = {"preprocess_ms": (perf_counter() - started) * 1000}
    detection = detect_document(processed.detection_gray, config)
    durations["detect_ms"] = detection.elapsed_ms
    statuses = {"preprocess": "ok", "detect": detection.status.value}
    debug = {
        "processed": processed,
        "detection": detection,
        "rectification": None,
        "baseline_raw": None,
    }
    diagnostics = {
        "coordinate_system": "P_raw",
        "orientation": "upright_geometric",
        "T_raw_to_det": processed.T_raw_to_det,
        "refinement_rejected": False,
        "ratio_policy": "known_source_ratio" if size else "ratio_unknown",
    }
    if detection.status != Status.OK:
        return PipelineResult(
            image_id, method, detection.status, statuses, durations, diagnostics=diagnostics
        ), debug
    baseline = detection.candidate_quad
    debug["baseline_raw"] = transform_points(baseline, np.linalg.inv(processed.T_raw_to_det))
    selected = baseline
    if method == "refined":
        result = refine_corners(
            baseline, detection.diagnostics["edges"], processed.detection_gray, config
        )
        durations["corners_ms"] = result.elapsed_ms
        diagnostics.update(result.diagnostics)
        diagnostics["per_edge_support"] = result.per_edge_support
        if result.corners is None:
            statuses["corners"] = result.status.value
            return PipelineResult(
                image_id, method, result.status, statuses, durations, diagnostics=diagnostics
            ), debug
        selected = result.corners
    else:
        durations["corners_ms"] = 0.0
    statuses["corners"] = "ok"
    raw = transform_points(selected, np.linalg.inv(processed.T_raw_to_det))
    valid, reason = validate_quad(raw, (image.shape[1], image.shape[0]), config)
    if not valid:
        statuses["corners"] = "invalid_quad"
        diagnostics["reason"] = reason
        return PipelineResult(
            image_id, method, Status.INVALID_QUAD, statuses, durations, diagnostics=diagnostics
        ), debug
    size = size or output_size(raw, config["output_width"])
    rectification = rectify(image, raw, size, config)
    debug["rectification"] = rectification
    statuses["rectify"] = rectification.status.value
    durations["rectify_ms"] = rectification.elapsed_ms
    diagnostics["rectification"] = rectification.diagnostics
    return PipelineResult(
        image_id,
        method,
        rectification.status,
        statuses,
        durations,
        raw,
        rectification.H,
        diagnostics=diagnostics,
    ), debug


def save_debug(out: Path, image: np.ndarray, result: PipelineResult, debug: dict) -> None:
    processed, detection = debug["processed"], debug["detection"]
    candidate_image = cv2.cvtColor(processed.detection_gray, cv2.COLOR_GRAY2BGR)
    for candidate in detection.diagnostics["candidates"]:
        q = candidate["quad"]
        cv2.polylines(candidate_image, [q.astype(np.int32)], True, (0, 180, 255), 2)
        cv2.putText(
            candidate_image,
            f"{candidate['score']:.3f}",
            tuple(q[0].astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    quads = overlay(image, debug["baseline_raw"], (255, 100, 0), "baseline P_raw")
    quads = overlay(
        quads, result.predicted_quad, (0, 200, 0), "baseline orange / final green P_raw"
    )
    rectification = debug["rectification"]
    montage(
        out / "montage.jpg",
        [
            ("Original EXIF-oriented P_raw", image),
            ("Detection gray P_det", processed.detection_gray),
            ("Canny P_det", detection.diagnostics["edges"]),
            ("Candidates and scores P_det", candidate_image),
            ("Corner comparison P_raw", quads),
            (
                "Rectified P_rect: " + result.status.value,
                rectification.image if rectification else None,
            ),
        ],
    )
    write_image(out / "edges.png", detection.diagnostics["edges"])
    if detection.mask is not None:
        write_image(out / "mask.png", detection.mask)
    write_json(out / "candidates.json", detection.diagnostics["candidates"])
    write_json(
        out / "transform.json",
        {
            "predicted_quad": result.predicted_quad,
            "H": result.H,
            "diagnostics": result.diagnostics,
            "statuses": result.statuses,
            "durations": result.durations,
        },
    )
    if rectification is not None and rectification.image is not None:
        write_image(out / "rectified.png", rectification.image)


def infer(
    path: Path, method: str, out: Path, config: dict, save_stages: bool = False
) -> PipelineResult:
    try:
        image = read_image(path)
    except ValueError as exc:
        result = PipelineResult(
            path.name,
            method,
            Status.IMAGE_READ_ERROR,
            {"io": "image_read_error"},
            {},
            diagnostics={"reason": str(exc)},
        )
        write_json(out / "prediction.json", asdict(result))
        return result
    result, debug = run_pipeline(image, method, config, image_id=path.name)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    ocr = None
    if result.status == Status.OK:
        corrected = debug["rectification"].image
        write_image(out / "rectified.png", corrected)
        paths["rectified"] = str(out / "rectified.png")
        ocr = recognize(corrected, config)
        (out / "ocr.txt").write_text(ocr.text, encoding="utf-8")
    if save_stages:
        save_debug(out, image, result, debug)
    values = {**asdict(result), "output_paths": paths, "ocr": asdict(ocr) if ocr else None}
    if ocr and ocr.status != Status.OK:
        values["status"] = ocr.status
    write_json(out / "prediction.json", values)
    return PipelineResult(
        **{
            **result.__dict__,
            "output_paths": paths,
            "ocr": ocr,
            "status": ocr.status if ocr and ocr.status != Status.OK else result.status,
        }
    )
