"""Generate a mathematical geometry fixture; this is not a research dataset image."""

from pathlib import Path

import cv2
import numpy as np

from cvdoc.io import write_image

if __name__ == "__main__":
    image = np.full((600, 800, 3), 96, np.uint8)
    points = np.array([[180, 80], [630, 100], [650, 530], [130, 510]], np.int32)
    cv2.fillConvexPoly(image, points, (245, 245, 245))
    cv2.putText(image, "TOP LEFT", (210, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
    cv2.putText(
        image, "DOCUMENT TEST 123", (200, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2
    )
    write_image(Path("tests/fixtures/sample_document.png"), image)
