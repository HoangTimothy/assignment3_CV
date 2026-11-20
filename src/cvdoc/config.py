"""Validated, serializable configuration loaded from packaged defaults."""

import math
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = yaml.safe_load(files("cvdoc").joinpath("defaults.yaml").read_text())
    if path:
        overrides = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(overrides, dict):
            raise ValueError("Configuration must be a YAML mapping")
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config.update(overrides)
    validate_config(config)
    return config


def validate_config(c: dict[str, Any]) -> None:

    numeric_keys = (
        "seed",
        "detect_max_side",
        "gaussian_kernel",
        "canny_low",
        "canny_high",
        "closing_kernel",
        "min_area_ratio",
        "max_area_ratio",
        "min_edge_ratio",
        "min_angle_degrees",
        "bounds_tolerance",
        "score_area",
        "score_support",
        "score_convexity",
        "score_border_penalty",
        "refine_corridor_ratio",
        "refine_corridor_min",
        "refine_corridor_max",
        "refine_min_support",
        "refine_max_shift_ratio",
        "refine_gradient_cos",
        "output_width",
        "ocr_page_psm",
        "ocr_roi_psm",
        "ocr_scale",
        "ocr_padding_ratio",
        "ocr_padding_min",
        "ocr_timeout",
        "save_successes",
        "ocr_workers",
    )
    for key in numeric_keys:
        value = c[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} requires a finite number")
    if not isinstance(c["clahe"], bool):
        raise ValueError("clahe must be a boolean")
    for key in ("data_dir", "results_dir", "ocr_language"):
        if not isinstance(c[key], str) or not c[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if c["tesseract_cmd"] is not None and not isinstance(c["tesseract_cmd"], str):
        raise ValueError("tesseract_cmd must be a string or null")
    if not isinstance(c["approx_epsilon"], list) or not all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in c["approx_epsilon"]
    ):
        raise ValueError("approx_epsilon must be a finite numeric list")
    if not isinstance(c["canny_fallbacks"], list) or not all(
        isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pair)
        for pair in c["canny_fallbacks"]
    ):
        raise ValueError("canny_fallbacks must contain finite threshold pairs")
    for key in ("detect_max_side", "output_width", "refine_min_support", "ocr_timeout"):
        if not isinstance(c[key], (int, float)) or c[key] < 2:
            raise ValueError(f"{key} must be at least 2")
    for key in ("gaussian_kernel", "closing_kernel"):
        if not isinstance(c[key], int) or c[key] < 1 or c[key] % 2 != 1:
            raise ValueError(f"{key} must be a positive odd integer")
    if not 0 <= c["canny_low"] < c["canny_high"] <= 255:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")
    if not 0 < c["min_area_ratio"] < c["max_area_ratio"] <= 1:
        raise ValueError("Invalid document area bounds")
    if not c["approx_epsilon"] or any(not 0 < x < 0.2 for x in c["approx_epsilon"]):
        raise ValueError("approx_epsilon must contain ratios in (0, .2)")
    if c["ocr_page_psm"] not in (3, 6) or c["ocr_roi_psm"] != 6:
        raise ValueError("Page PSM must be 3 or 6; ROI PSM must be 6")
    for key in ("ocr_scale", "min_edge_ratio", "refine_corridor_ratio"):
        if c[key] <= 0:
            raise ValueError(f"{key} must be positive")

    for key, value in c.items():
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    for key in (
        "detect_max_side",
        "output_width",
        "refine_min_support",
        "seed",
        "ocr_workers",
        "save_successes",
    ):
        if not isinstance(c[key], int) or isinstance(c[key], bool):
            raise ValueError(f"{key} must be an integer")
    if not 1 <= c["ocr_workers"] <= 16:
        raise ValueError("ocr_workers must be in [1,16]")
    if c["save_successes"] < 0 or c["seed"] < 0:
        raise ValueError("seed and save_successes must be nonnegative")
    if not 0 <= c["refine_gradient_cos"] <= 1:
        raise ValueError("refine_gradient_cos must be in [0,1]")
    if not 0 < c["min_angle_degrees"] < 90:
        raise ValueError("min_angle_degrees must be in (0,90)")
    if not 0 < c["refine_corridor_min"] <= c["refine_corridor_max"]:
        raise ValueError("Invalid refinement corridor bounds")
    for key in (
        "score_area",
        "score_support",
        "score_convexity",
        "score_border_penalty",
        "bounds_tolerance",
        "ocr_padding_min",
        "ocr_padding_ratio",
    ):
        if not isinstance(c[key], (int, float)) or c[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    for pair in c["canny_fallbacks"]:
        if len(pair) != 2 or not 0 <= pair[0] < pair[1] <= 255:
            raise ValueError("Invalid Canny fallback threshold pair")
