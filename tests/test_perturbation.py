import cv2
import numpy as np
import pytest

from cvdoc.datasets.perturb import generate_view


@pytest.mark.parametrize("condition", ["frontal", "mild", "strong", "stress"])
def test_derived_h_reproducible(condition):
    page = np.full((200, 150, 3), 255, np.uint8)
    a = generate_view(page, "page1", condition)
    b = generate_view(page, "page1", condition)
    np.testing.assert_array_equal(a.H_page_to_view, b.H_page_to_view)
    np.testing.assert_array_equal(a.image, b.image)


def test_derived_quad_is_convex():
    for seed in range(20):
        view = generate_view(np.full((200, 150, 3), 255, np.uint8), "page", "strong", seed)
        assert cv2.isContourConvex(view.polygon.astype(np.float32))
        assert (view.polygon >= 0).all()
        assert (view.polygon < [240, 320]).all()
        assert set(np.unique(view.mask)) <= {0, 255}
