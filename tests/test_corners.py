import numpy as np
import pytest

from cvdoc.corners import intersect_lines, order_corners, refine_corners, validate_quad
from cvdoc.detect import detect_document
from cvdoc.preprocess import preprocess


def test_ordering_tl_tr_br_bl(page_fixture):
    _, q = page_fixture
    for permutation in ([2, 0, 3, 1], [1, 2, 3, 0], [0, 3, 2, 1]):
        np.testing.assert_allclose(order_corners(q[permutation]), q)


@pytest.mark.parametrize(
    "quad",
    [
        [[0, 0], [0, 0], [40, 40], [0, 40]],
        [[10, 10], [50, 50], [50, 10], [10, 50]],
        [[10, 10], [50, 10], [20, 20], [10, 50]],
        [[0, 0], [10, 0], [20, 0], [30, 0]],
        [[0, 0], [50, 0], [50, float("nan")], [0, 50]],
    ],
)
def test_invalid_quad_rejected(config, quad):
    assert not validate_quad(quad, (100, 100), config)[0]


def test_line_intersection():
    np.testing.assert_allclose(intersect_lines([1, 0, -3], [0, 1, -4]), [3, 4])
    with pytest.raises(ValueError):
        intersect_lines([1, 0, -3], [1, 0, -4])


def test_line_refinement_never_returns_nan(config, page_fixture):
    image, q = page_fixture
    result = refine_corners(
        q, np.zeros(image.shape[:2], np.uint8), np.zeros(image.shape[:2], np.uint8), config
    )
    np.testing.assert_array_equal(result.corners, q)
    assert result.diagnostics["refinement_rejected"]
    gray = preprocess(image, config).detection_gray
    detection = detect_document(gray, config)
    result = refine_corners(detection.candidate_quad, detection.diagnostics["edges"], gray, config)
    assert np.isfinite(result.corners).all()
    assert validate_quad(result.corners, (800, 600), config)[0]
