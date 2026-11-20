"""Native-image convex IoU, semantic NCE, and pooled text edit counts."""

import re
import unicodedata
from collections.abc import Sequence

import cv2
import jiwer
import numpy as np


def polygon_iou(prediction: np.ndarray, reference: np.ndarray) -> float:
    first = np.asarray(prediction, np.float32)
    second = np.asarray(reference, np.float32)
    if (
        first.shape != (4, 2)
        or second.shape != (4, 2)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("IoU requires two finite quadrilaterals")
    if not cv2.isContourConvex(first) or not cv2.isContourConvex(second):
        raise ValueError("IoU requires convex polygons in cyclic order")
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = cv2.contourArea(first) + cv2.contourArea(second) - intersection
    return float(np.clip(intersection / union, 0, 1)) if union > 0 else 0.0


def normalized_corner_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    p, q = np.asarray(prediction, float), np.asarray(reference, float)
    if p.shape != (4, 2) or q.shape != (4, 2) or not np.isfinite([p, q]).all():
        raise ValueError("NCE requires finite semantic TL,TR,BR,BL corners")
    diagonal = float(np.linalg.norm(np.ptp(q, axis=0)))
    if diagonal <= 0:
        raise ValueError("Degenerate GT bounds")
    return float(np.linalg.norm(p - q, axis=1).mean() / diagonal)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def edit_counts(reference: str, prediction: str, words: bool = False) -> dict:
    reference, prediction = normalize_text(reference), normalize_text(prediction)
    process = jiwer.process_words if words else jiwer.process_characters
    result = process(reference, prediction)
    length = len(reference.split()) if words else len(reference)
    return {
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "errors": result.substitutions + result.deletions + result.insertions,
        "length": length,
        "rate": (
            (result.substitutions + result.deletions + result.insertions) / length
            if length
            else None
        ),
    }


def text_metrics(reference: str, prediction: str) -> dict:
    chars, words = edit_counts(reference, prediction), edit_counts(reference, prediction, True)
    return {
        "CER": chars["rate"],
        "WER": words["rate"],
        "char_errors": chars["errors"],
        "char_count": chars["length"],
        "word_errors": words["errors"],
        "word_count": words["length"],
        **{f"char_{key}": chars[key] for key in ("substitutions", "deletions", "insertions")},
        **{f"word_{key}": words[key] for key in ("substitutions", "deletions", "insertions")},
    }


def geometry_summary(rows: Sequence[dict]) -> dict:
    good = [r for r in rows if r["status"] == "ok"]
    values = np.array([r["NCE"] for r in good], float)
    return {
        "samples": len(rows),
        "valid": len(good),
        "failures": len(rows) - len(good),
        "valid_rate": len(good) / len(rows) if rows else None,
        "conditional_mean_iou": float(np.mean([r["polygon_iou"] for r in good])) if good else None,
        "complete_mean_iou": float(np.mean([r["polygon_iou"] for r in rows])) if rows else None,
        "conditional_mean_nce": float(values.mean()) if good else None,
        "conditional_median_nce": float(np.median(values)) if good else None,
        "conditional_p90_nce": float(np.quantile(values, 0.9)) if good else None,
        **{
            f"success_nce_{threshold}": sum(float(r["NCE"]) < threshold for r in good) / len(rows)
            if rows
            else None
            for threshold in (0.01, 0.02, 0.05)
        },
        "mean_runtime_ms": float(np.mean([r["runtime_ms"] for r in rows])) if rows else None,
        "refinement_fallback_rate": sum(bool(r.get("refinement_rejected")) for r in good)
        / len(good)
        if good
        else None,
    }
