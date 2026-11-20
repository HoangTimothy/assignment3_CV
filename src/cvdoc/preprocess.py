"""Keep raw pixels intact; map points with the recorded anisotropic resize."""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Preprocessed:
    original_bgr: np.ndarray
    detection_gray: np.ndarray
    T_raw_to_det: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any]


def preprocess(image: np.ndarray, config: dict) -> Preprocessed:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected nonempty uint8 BGR image")
    h, w = image.shape[:2]
    scale = min(1.0, config["detect_max_side"] / max(w, h))
    dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
    detection = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(detection, cv2.COLOR_BGR2GRAY)
    if config["clahe"]:
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (config["gaussian_kernel"],) * 2, 0)
    transform = np.diag([dw / w, dh / h, 1.0])
    return Preprocessed(
        image,
        gray,
        transform,
        np.full((dh, dw), 255, np.uint8),
        {"sx": dw / w, "sy": dh / h, "offset_x": 0, "offset_y": 0},
    )
