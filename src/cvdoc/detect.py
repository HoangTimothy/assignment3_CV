"""Scored convex contour candidates with an explicit no-document outcome."""

from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np

from .corners import approximate_corners
from .schemas import DetectionResult, Status


def edge_support(quad: np.ndarray, edges: np.ndarray) -> float:
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    h, w = edges.shape
    values = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        samples = np.linspace(a, b, max(2, int(np.linalg.norm(b - a))))
        x = np.clip(np.rint(samples[:, 0]).astype(int), 0, w - 1)
        y = np.clip(np.rint(samples[:, 1]).astype(int), 0, h - 1)
        values.append(float(np.mean(distance[y, x] <= 2.5)))
    return float(np.mean(values))


def _detect_once(gray: np.ndarray, config: dict) -> DetectionResult:
    started = perf_counter()
    h, w = gray.shape
    edges = cv2.Canny(gray, config["canny_low"], config["canny_high"])
    kernel = np.ones((config["closing_kernel"],) * 2, np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    candidates = []
    mode_used = "external"
    for mode, label in ((cv2.RETR_EXTERNAL, "external"), (cv2.RETR_LIST, "all_fallback")):
        contours, _ = cv2.findContours(closed, mode, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour) / (w * h)
            if area < config["min_area_ratio"]:
                continue
            quad = approximate_corners(contour, (w, h), config)
            if quad is None:
                continue
            support = edge_support(quad, edges)
            convexity = min(
                1.0, cv2.contourArea(contour) / max(1.0, cv2.contourArea(cv2.convexHull(contour)))
            )
            border = float(
                np.mean(
                    (quad[:, 0] < 3)
                    | (quad[:, 1] < 3)
                    | (quad[:, 0] > w - 4)
                    | (quad[:, 1] > h - 4)
                )
            )
            whole_frame = max(0.0, (area - 0.85) / 0.15)
            score = (
                config["score_area"] * area
                + config["score_support"] * support
                + config["score_convexity"] * convexity
                - config["score_border_penalty"] * max(border, whole_frame)
            )
            candidates.append(
                {
                    "quad": quad,
                    "contour": contour,
                    "score": score,
                    "area_fraction": area,
                    "edge_support": support,
                    "convexity": convexity,
                    "border_penalty": max(border, whole_frame),
                }
            )
        if candidates:
            mode_used = label
            break
    diagnostics = {
        "edges": edges,
        "closed": closed,
        "retrieval": mode_used,
        "candidates": [{k: v for k, v in item.items() if k != "contour"} for item in candidates],
    }
    elapsed = (perf_counter() - started) * 1000
    if not candidates:
        return DetectionResult(Status.NO_CANDIDATE, elapsed_ms=elapsed, diagnostics=diagnostics)
    winner = max(candidates, key=lambda item: item["score"])
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, np.rint(winner["quad"]).astype(np.int32), 255)
    return DetectionResult(
        Status.OK, winner["quad"], winner["contour"], mask, winner["score"], elapsed, diagnostics
    )


def detect_document(gray: np.ndarray, config: dict) -> DetectionResult:
    """Try the fixed lower-threshold grid only when the primary detector fails."""
    start = perf_counter()
    attempts = []
    result = None
    for low, high in [(config["canny_low"], config["canny_high"])] + config["canny_fallbacks"]:
        settings = {**config, "canny_low": low, "canny_high": high}
        result = _detect_once(gray, settings)
        attempts.append({"low": low, "high": high, "status": result.status.value})
        if result.status == Status.OK:
            break
    return replace(
        result,
        elapsed_ms=(perf_counter() - start) * 1000,
        diagnostics={**result.diagnostics, "attempts": attempts},
    )
