"""Deterministic controlled scan views; these are not camera photographs."""

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DerivedView:
    image: np.ndarray
    mask: np.ndarray
    H_page_to_view: np.ndarray
    polygon: np.ndarray
    metadata: dict


def generate_view(
    page: np.ndarray, document_id: str, condition: str, seed: int = 42
) -> DerivedView:
    if condition not in ("frontal", "mild", "strong", "stress"):
        raise ValueError(f"Unknown perspective condition {condition}")
    h, w = page.shape[:2]
    if min(w, h) < 2:
        raise ValueError("Page is too small")
    key = f"{seed}/{document_id}/{condition}".encode()
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    rng = np.random.default_rng(local_seed)
    cw, ch = round(1.6 * w), round(1.6 * h)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    base = src.astype(float) + [(cw - w) / 2, (ch - h) / 2]
    low, high = (0.03, 0.10) if condition == "mild" else (0.10, 0.22)
    for attempt in range(100):
        quad = base.copy()
        if condition != "frontal":
            delta = rng.uniform(low, high, (4, 2)) * [w, h]
            quad += delta * rng.choice([-1, 1], (4, 2))
        convex = cv2.isContourConvex(quad.astype(np.float32))
        inside = (quad >= 0).all() and (quad[:, 0] < cw).all() and (quad[:, 1] < ch).all()
        area = cv2.contourArea(quad.astype(np.float32), oriented=True)
        if convex and inside and area > 0.25 * w * h:
            break
    else:
        raise ValueError("Unable to generate stable perspective quad")
    H = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    image = cv2.warpPerspective(page, H, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=(96, 96, 96))
    mask = cv2.warpPerspective(
        np.full((h, w), 255, np.uint8), H, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0
    )
    if condition == "stress":
        image = cv2.GaussianBlur(image, (3, 3), 0.7)
        image = np.clip(image.astype(float) * 0.85 + rng.normal(0, 3, image.shape), 0, 255).astype(
            np.uint8
        )
    metadata = {
        "source_document_id": document_id,
        "condition": condition,
        "seed": seed,
        "local_seed": local_seed,
        "view_id": f"{document_id}_{condition}",
        "source_size": [w, h],
        "image_size": [cw, ch],
        "polygon": quad.tolist(),
        "H_page_to_view": H.tolist(),
        "attempts": attempt + 1,
        "displacement_range": [0, 0] if condition == "frontal" else [low, high],
        "background_bgr": [96, 96, 96],
        "stress": condition == "stress",
    }
    return DerivedView(image, mask, H, quad, metadata)
