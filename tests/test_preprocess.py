import numpy as np

from cvdoc.geometry import transform_points
from cvdoc.preprocess import preprocess


def test_downscale_point_mapping(config):
    image = np.zeros((1601, 2303, 3), np.uint8)
    result = preprocess(image, config)
    p = np.array([[400, 800], [2200, 1550]], float)
    restored = transform_points(
        transform_points(p, result.T_raw_to_det), np.linalg.inv(result.T_raw_to_det)
    )
    np.testing.assert_allclose(p, restored, atol=1e-8)
    assert max(result.detection_gray.shape) == 1400
    assert np.shares_memory(result.original_bgr, image)


def test_bgr_gray(config):
    image = np.full((10, 10, 3), (255, 0, 0), np.uint8)
    result = preprocess(image, config)
    assert result.detection_gray[0, 0] == 29
