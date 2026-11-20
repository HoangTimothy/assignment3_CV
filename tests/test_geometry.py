import numpy as np

from cvdoc.geometry import destination_corners, rectify, transform_points
from cvdoc.preprocess import preprocess
from cvdoc.schemas import Status


def test_homography_known_corners(config, page_fixture):
    image, quad = page_fixture
    result = rectify(image, quad, (400, 600), config)
    assert result.status == Status.OK
    np.testing.assert_allclose(
        transform_points(quad, result.H), destination_corners((400, 600)), atol=1e-3
    )


def test_homography_roundtrip(config, page_fixture):
    image, quad = page_fixture
    result = rectify(image, quad, (400, 600), config)
    points = np.random.default_rng(42).uniform([200, 150], [500, 450], size=(30, 2))
    np.testing.assert_allclose(
        transform_points(transform_points(points, result.H), np.linalg.inv(result.H)),
        points,
        atol=1e-7,
    )


def test_homography_resize_then_fullres(config, page_fixture):
    image, quad = page_fixture
    config["detect_max_side"] = 300
    processed = preprocess(image, config)
    det = transform_points(quad, processed.T_raw_to_det)
    raw = transform_points(det, np.linalg.inv(processed.T_raw_to_det))
    result = rectify(image, raw, (400, 600), config)
    assert result.status == Status.OK and result.image.shape == (600, 400, 3)


def test_border_is_white(config):
    image = np.zeros((100, 100, 3), np.uint8)
    quad = np.array([[-2, -2], [90, -2], [90, 90], [-2, 90]], float)
    result = rectify(image, quad, (100, 100), config)
    assert result.status == Status.OK
    np.testing.assert_array_equal(result.image[0, 0], [255, 255, 255])


def test_synthetic_no_flip(config):
    image = np.full((200, 200, 3), 255, np.uint8)
    image[30:60, 30:60] = [0, 0, 255]
    quad = np.array([[20, 20], [180, 20], [180, 180], [20, 180]], float)
    result = rectify(image, quad, (160, 160), config)
    assert result.image[20, 20, 2] > 250 and result.image[20, 20, 0] < 5
    assert result.image[140, 140, 0] > 250
    assert rectify(image, quad[[0, 3, 2, 1]], (160, 160), config).status != Status.OK
