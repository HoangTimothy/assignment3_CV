import cv2
import numpy as np

from cvdoc.corners import validate_quad
from cvdoc.detect import detect_document, edge_support
from cvdoc.metrics import polygon_iou
from cvdoc.preprocess import preprocess
from cvdoc.schemas import Status


def test_canny_border_on_clean_quad(config, page_fixture):
    image, quad = page_fixture
    result = detect_document(preprocess(image, config).detection_gray, config)
    assert edge_support(quad, result.diagnostics["edges"]) > 0.8


def test_contour_select_document(config, page_fixture):
    image, quad = page_fixture
    cv2.rectangle(image, (20, 20), (100, 120), (255, 255, 255), -1)
    result = detect_document(preprocess(image, config).detection_gray, config)
    assert result.status == Status.OK
    assert np.linalg.norm(result.candidate_quad - quad, axis=1).max() < 6
    assert result.mask[300, 400] == 255 and result.mask[30, 30] == 0


def test_no_document_does_not_fake_success(config):
    result = detect_document(np.full((200, 300), 100, np.uint8), config)
    assert result.status == Status.NO_CANDIDATE
    assert result.candidate_quad is None


def test_approx_poly_four_corners(config, page_fixture):
    image, _ = page_fixture
    result = detect_document(preprocess(image, config).detection_gray, config)
    assert result.candidate_quad.shape == (4, 2)


def test_broken_edge_and_small_page_fail_safely(config, page_fixture):
    image, reference = page_fixture
    cv2.line(image, (170, 70), (200, 90), (96, 96, 96), 15)
    result = detect_document(preprocess(image, config).detection_gray, config)
    if result.status == Status.OK:
        assert validate_quad(result.candidate_quad, (800, 600), config)[0]
        assert polygon_iou(result.candidate_quad, reference) > 0.9
    else:
        assert result.status == Status.NO_CANDIDATE
        assert result.candidate_quad is None and result.mask is None
    small = np.full((400, 400), 96, np.uint8)
    cv2.rectangle(small, (50, 50), (70, 80), 255, -1)
    assert detect_document(small, config).status == Status.NO_CANDIDATE
