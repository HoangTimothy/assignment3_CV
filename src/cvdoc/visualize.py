"""Auditable montages from real stage arrays."""

from pathlib import Path

import cv2
import numpy as np

from .io import write_image


def overlay(
    image: np.ndarray, quad: np.ndarray | None, color=(0, 255, 0), label: str = "prediction P_raw"
) -> np.ndarray:
    canvas = image.copy()
    if quad is not None:
        points = np.rint(quad).astype(np.int32)
        cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)
        for point, name in zip(points, ("TL", "TR", "BR", "BL")):
            cv2.circle(canvas, tuple(point), 5, color, -1)
            cv2.putText(
                canvas, name, tuple(point + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
    cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return canvas


def montage(path: Path, panels: list[tuple[str, np.ndarray | None]], tile_size=(480, 360)) -> None:
    tw, th = tile_size
    tiles = []
    for label, array in panels:
        tile = np.full((th, tw, 3), 235, np.uint8)
        if array is not None and array.size:
            image = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR) if array.ndim == 2 else array
            scale = min(tw / image.shape[1], (th - 30) / image.shape[0])
            resized = cv2.resize(
                image,
                (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            )
            y, x = 30 + (th - 30 - resized.shape[0]) // 2, (tw - resized.shape[1]) // 2
            tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(
            tile, label[:68], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA
        )
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.full((th, tw, 3), 235, np.uint8))
    grid = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
    write_image(path, grid)
