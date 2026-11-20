"""Upright cyclic ordering, strict validity, and robust boundary-line refinement."""

from time import perf_counter

import cv2
import numpy as np

from .schemas import CornerResult, Status


def signed_area(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=float)
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)) / 2)


def order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Expected four finite corners")
    hull = cv2.convexHull(points.astype(np.float32), clockwise=False).reshape(-1, 2)
    if len(hull) != 4:
        raise ValueError("Four unique convex corners required")
    # Sorting cyclically avoids duplicate extrema on diamonds/severe skew.
    center = points.mean(axis=0)
    angle = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    cyclic = points[np.argsort(angle)]
    if signed_area(cyclic) < 0:
        cyclic = cyclic[::-1]
    midpoints = (cyclic + np.roll(cyclic, -1, axis=0)) / 2
    top_edges = np.argsort(midpoints[:, 1], kind="stable")
    first = int(top_edges[0])
    if abs(midpoints[top_edges[1], 1] - midpoints[first, 1]) < 0.01 * np.ptp(points[:, 1]):
        raise ValueError("Ambiguous upright top edge")
    ordered = np.roll(cyclic, -first, axis=0)
    edge = ordered[1] - ordered[0]
    if edge[0] <= 0 or abs(edge[1]) > 1.5 * edge[0]:
        raise ValueError("Unsupported page orientation")
    return ordered.copy()


def validate_quad(
    points: np.ndarray, image_size: tuple[int, int], config: dict, check_order: bool = True
) -> tuple[bool, str]:
    q = np.asarray(points, dtype=np.float64)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        return False, "nonfinite_or_shape"
    w, h = image_size
    sides = np.roll(q, -1, axis=0) - q
    lengths = np.linalg.norm(sides, axis=1)
    if min(lengths) < config["min_edge_ratio"] * np.hypot(w, h):
        return False, "short_edge"
    distances = np.linalg.norm(q[:, None] - q[None, :], axis=2)
    if np.min(distances + np.eye(4) * max(w, h) * 10) < 1:
        return False, "duplicate_corners"
    if not cv2.isContourConvex(q.astype(np.float32)) or signed_area(q) <= 0:
        return False, "nonconvex_or_reversed"
    ratio = signed_area(q) / (w * h)
    if not config["min_area_ratio"] <= ratio <= config["max_area_ratio"]:
        return False, "area_ratio"
    tolerance = config["bounds_tolerance"]
    if (
        (q < -tolerance).any()
        or (q[:, 0] > w - 1 + tolerance).any()
        or (q[:, 1] > h - 1 + tolerance).any()
    ):
        return False, "out_of_bounds"
    cosine = np.sum(-np.roll(sides, 1, axis=0) * sides, axis=1) / (np.roll(lengths, 1) * lengths)
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    if min(angles) < config["min_angle_degrees"] or max(angles) > 180 - config["min_angle_degrees"]:
        return False, "degenerate_angle"
    if check_order:
        try:
            ordered = order_corners(q)
        except ValueError as exc:
            return False, str(exc)
        if not np.allclose(q, ordered):
            return False, "wrong_corner_order"
    return True, "ok"


def approximate_corners(
    contour: np.ndarray, image_size: tuple[int, int], config: dict
) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    for alpha in config["approx_epsilon"]:
        approximation = cv2.approxPolyDP(contour, alpha * perimeter, True).reshape(-1, 2)
        if len(approximation) != 4:
            continue
        try:
            quad = order_corners(approximation)
        except ValueError:
            continue
        if validate_quad(quad, image_size, config)[0]:
            return quad
    return None


def intersect_lines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    matrix = np.array([first[:2], second[:2]], float)
    if abs(np.linalg.det(matrix)) < 1e-5:
        raise ValueError("Nearly parallel boundary lines")
    return np.linalg.solve(matrix, -np.array([first[2], second[2]], float))


def refine_corners(
    baseline: np.ndarray, edges: np.ndarray, gray: np.ndarray, config: dict
) -> CornerResult:
    started = perf_counter()
    q = np.asarray(baseline, dtype=np.float64)
    h, w = edges.shape
    support, lines = [], []
    diagnostics = {"refinement_rejected": False}

    def fallback(reason: str) -> CornerResult:
        return CornerResult(
            Status.OK,
            q.copy(),
            "refined",
            tuple(support),
            (perf_counter() - started) * 1000,
            {**diagnostics, "refinement_rejected": True, "reason": reason},
        )

    if not validate_quad(q, (w, h), config)[0]:
        return CornerResult(Status.INVALID_QUAD, method="refined")
    y, x = np.nonzero(edges)
    pixels = np.column_stack([x, y]).astype(float)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.column_stack([gx[y, x], gy[y, x]])
    magnitude = np.linalg.norm(gradient, axis=1)
    corridor = np.clip(
        config["refine_corridor_ratio"] * np.hypot(w, h),
        config["refine_corridor_min"],
        config["refine_corridor_max"],
    )
    diagnostics["corridor_pixels"] = float(corridor)
    for index in range(4):
        a, b = q[index], q[(index + 1) % 4]
        tangent = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-tangent[1], tangent[0]])
        projected = (pixels - a) @ tangent
        distance = abs((pixels - a) @ normal)
        compatible = abs(gradient @ normal) >= config["refine_gradient_cos"] * magnitude
        keep = (
            (distance <= corridor)
            & (projected >= 0.03 * np.linalg.norm(b - a))
            & (projected <= 0.97 * np.linalg.norm(b - a))
            & compatible
            & (magnitude > 10)
        )
        selected = pixels[keep]
        support.append(len(selected))
        if len(selected) < config["refine_min_support"]:
            return fallback("insufficient_edges")
        vx, vy, px, py = (
            cv2.fitLine(selected.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
            .ravel()
            .astype(float)
        )
        line = np.array([-vy, vx, vy * px - vx * py])
        line /= np.linalg.norm(line[:2])
        inliers = selected[abs(selected @ line[:2] + line[2]) <= max(1.5, corridor / 3)]
        if len(inliers) < config["refine_min_support"]:
            return fallback("insufficient_inliers")
        vx, vy, px, py = (
            cv2.fitLine(inliers.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
            .ravel()
            .astype(float)
        )
        line = np.array([-vy, vx, vy * px - vx * py])
        lines.append(line / np.linalg.norm(line[:2]))
    try:
        refined = np.array([intersect_lines(lines[(i - 1) % 4], lines[i]) for i in range(4)])
    except (ValueError, np.linalg.LinAlgError) as exc:
        return fallback(str(exc))
    displacement = np.linalg.norm(refined - q, axis=1)
    diagnostics["displacements"] = displacement.tolist()
    valid, reason = validate_quad(refined, (w, h), config)
    if not valid or max(displacement) > config["refine_max_shift_ratio"] * np.hypot(w, h):
        return fallback(reason if not valid else "excessive_displacement")
    return CornerResult(
        Status.OK,
        refined,
        "refined",
        tuple(support),
        (perf_counter() - started) * 1000,
        diagnostics,
    )
