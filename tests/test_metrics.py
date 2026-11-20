import numpy as np
import pytest

from cvdoc.metrics import (
    edit_counts,
    geometry_summary,
    normalize_text,
    normalized_corner_error,
    polygon_iou,
    text_metrics,
)

Q = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)


def test_polygon_iou_identity():
    assert polygon_iou(Q, Q) == pytest.approx(1)


def test_polygon_iou_disjoint():
    assert polygon_iou(Q, Q + 20) == 0


def test_polygon_iou_analytic():
    assert polygon_iou(Q, Q + [5, 0]) == pytest.approx(1 / 3)


def test_corner_error_zero():
    assert normalized_corner_error(Q, Q) == 0


def test_corner_error_known_offset():
    assert normalized_corner_error(Q + [3, 4], Q) == pytest.approx(5 / np.sqrt(200))


def test_cer_wer_examples():
    assert edit_counts("cat", "cut")["substitutions"] == 1
    assert edit_counts("cat", "cats")["insertions"] == 1
    assert edit_counts("cats", "cat")["deletions"] == 1
    assert text_metrics("one two", "one three")["WER"] == 0.5
    assert text_metrics("same", "same")["CER"] == 0


def test_text_normalization():
    assert normalize_text("  CAF\u00c9\n \uff21\uff22\uff23  12! \u00df ") == "caf\u00e9 abc 12! ss"


def test_empty_prediction_penalty():
    assert text_metrics("hello world", "")["CER"] == 1
    assert text_metrics("hello world", "")["WER"] == 1
    assert edit_counts("", "abc")["rate"] is None
    assert edit_counts("", "abc")["insertions"] == 3


def test_failure_denominator():
    summary = geometry_summary(
        [
            {"status": "ok", "NCE": 0, "polygon_iou": 1, "runtime_ms": 1},
            {"status": "no_candidate", "NCE": None, "polygon_iou": 0, "runtime_ms": 1},
        ]
    )
    assert summary["complete_mean_iou"] == 0.5
    assert summary["conditional_mean_iou"] == 1
    assert summary["success_nce_0.01"] == 0.5
