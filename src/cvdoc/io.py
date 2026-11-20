"""EXIF-aware image I/O and strict JSON serialization."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


def read_image(path: str | Path) -> np.ndarray:
    try:
        with Image.open(path) as source:
            if source.format not in ("PNG", "JPEG", "TIFF", "BMP", "WEBP"):
                raise ValueError("Unsupported image format")
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        if rgb.size == 0:
            raise ValueError("Empty image")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"image_read_error: {path}: {exc}") from exc


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None or image.size == 0:
        raise ValueError("Cannot save empty image")
    extension = path.suffix.lower()
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise OSError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False), encoding="utf-8"
    )
