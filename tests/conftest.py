import cv2
import numpy as np
import pytest

from cvdoc.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def page_fixture():
    # Mathematical unit-test geometry fixture, not a research dataset image.
    image = np.full((600, 800, 3), 96, np.uint8)
    quad = np.array([[180, 80], [630, 100], [650, 530], [130, 510]], dtype=float)
    cv2.fillConvexPoly(image, quad.astype(np.int32), (245, 245, 245))
    cv2.putText(image, "TOP LEFT", (210, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
    for y in range(200, 450, 35):
        cv2.line(image, (220, y), (550, y), (30, 30, 30), 2)
    return image, quad
