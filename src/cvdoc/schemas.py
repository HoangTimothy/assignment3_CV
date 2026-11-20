"""Typed stage results. Coordinates are explicitly named in each record."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class Status(str, Enum):
    OK = "ok"
    NO_CANDIDATE = "no_candidate"
    INVALID_QUAD = "invalid_quad"
    INSUFFICIENT_EDGES = "insufficient_edges"
    INVALID_HOMOGRAPHY = "invalid_homography"
    IMAGE_READ_ERROR = "image_read_error"
    OCR_ERROR = "ocr_error"


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    source: str
    path: Path
    document_id: str
    split: str
    width: int
    height: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Quad:
    points: np.ndarray
    image_size: tuple[int, int]
    coordinate_system: str = "P_raw"
    orientation: str = "upright_geometric_TL_TR_BR_BL"

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=np.float64, copy=True)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Quad requires finite (4,2) points")
        points.setflags(write=False)
        object.__setattr__(self, "points", points)


@dataclass(frozen=True)
class DetectionResult:
    status: Status
    candidate_quad: np.ndarray | None = None
    contour: np.ndarray | None = None
    mask: np.ndarray | None = None
    score: float | None = None
    elapsed_ms: float = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CornerResult:
    status: Status
    corners: np.ndarray | None = None
    method: str = "baseline"
    per_edge_support: tuple[int, ...] = ()
    elapsed_ms: float = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RectifyResult:
    status: Status
    H: np.ndarray | None = None
    image: np.ndarray | None = None
    output_size: tuple[int, int] | None = None
    elapsed_ms: float = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRResult:
    status: Status
    text: str = ""
    engine_version: str = "unavailable"
    language: str = "eng"
    psm: int = 6
    elapsed_ms: float = 0
    detail: str = ""


@dataclass(frozen=True)
class PipelineResult:
    image_id: str
    method: str
    status: Status
    statuses: dict[str, str]
    durations: dict[str, float]
    predicted_quad: np.ndarray | None = None
    H: np.ndarray | None = None
    output_paths: dict[str, str] = field(default_factory=dict)
    ocr: OCRResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
