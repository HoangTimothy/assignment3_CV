"""Source-to-destination projective geometry with checked roundtrips."""

from time import perf_counter

import cv2
import numpy as np

from .corners import validate_quad
from .schemas import RectifyResult, Status


def transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(H, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError("Expected Nx2 points and 3x3 H")
    homogeneous = np.column_stack([p, np.ones(len(p))]) @ matrix.T
    if not np.isfinite(homogeneous).all() or (abs(homogeneous[:, 2]) < 1e-10).any():
        raise ValueError("Nonfinite projective mapping or point at infinity")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def destination_corners(size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if w < 2 or h < 2:
        raise ValueError("Output must be at least 2x2")
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)


def output_size(quad: np.ndarray | None, width: int, ratio: float | None = None) -> tuple[int, int]:
    if ratio is None:
        if quad is None:
            raise ValueError("Corners required when aspect ratio is unknown")
        sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        ratio = float((sides[0] + sides[2]) / (sides[1] + sides[3]))
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid physical aspect ratio")
    height = max(2, round(width / ratio))
    if width * height > 50_000_000:
        raise ValueError("Rectification exceeds 50 million pixel safety limit")
    return int(width), height


def rectify(
    image: np.ndarray, quad: np.ndarray, size: tuple[int, int], config: dict
) -> RectifyResult:
    started = perf_counter()
    h, w = image.shape[:2]
    valid, reason = validate_quad(quad, (w, h), config)
    if not valid:
        return RectifyResult(Status.INVALID_QUAD, diagnostics={"reason": reason})
    try:
        destination = destination_corners(size)
        H = cv2.getPerspectiveTransform(
            np.asarray(quad, np.float32), destination.astype(np.float32)
        )
        if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
            raise ValueError("Nonfinite or singular homography")
        inverse = np.linalg.inv(H)
        projected = transform_points(quad, H)
        error = float(np.max(np.linalg.norm(projected - destination, axis=1)))
        roundtrip = float(
            np.max(np.linalg.norm(transform_points(projected, inverse) - quad, axis=1))
        )
        if error > 0.01 or roundtrip > 0.01:
            raise ValueError("Homography reprojection failed")
        image_out = cv2.warpPerspective(
            image,
            H,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return RectifyResult(
            Status.OK,
            H,
            image_out,
            size,
            (perf_counter() - started) * 1000,
            {"H_inverse": inverse, "reprojection_error": error, "roundtrip_error": roundtrip},
        )
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        return RectifyResult(
            Status.INVALID_HOMOGRAPHY,
            elapsed_ms=(perf_counter() - started) * 1000,
            diagnostics={"reason": str(exc)},
        )
