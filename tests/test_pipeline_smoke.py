import inspect

import numpy as np

from cvdoc.experiments import crop_region, prediction_branches
from cvdoc.geometry import transform_points
from cvdoc.pipeline import run_pipeline, save_debug
from cvdoc.schemas import Status


def test_pipeline_smoke(config, page_fixture, tmp_path):
    image, q = page_fixture
    result, debug = run_pipeline(image, "refined", config, (450, 600))
    assert result.status == Status.OK
    assert result.H is not None
    assert result.predicted_quad.shape == (4, 2)
    save_debug(tmp_path, image, result, debug)
    assert (tmp_path / "montage.jpg").is_file()
    assert (tmp_path / "candidates.json").is_file()


def test_no_oracle_leak(config, page_fixture, monkeypatch):
    from cvdoc import experiments

    image, _ = page_fixture

    def forbidden(*args, **kwargs):
        raise AssertionError("Oracle accessed in proposed inference")

    monkeypatch.setattr(experiments, "oracle_branch", forbidden)
    images, results = prediction_branches(image, (450, 600), config, "fixture")
    assert "E4_oracle" not in images
    assert all(r.status == Status.OK for r, _ in results.values())
    assert "reference" not in inspect.signature(run_pipeline).parameters
    assert "gt_quad" not in inspect.getsource(run_pipeline)


def test_region_crop_uses_composed_coordinates(config):
    image = np.full((200, 300, 3), 255, np.uint8)
    q = np.array([[10, 20], [50, 20], [50, 60], [10, 60]], float)
    H = np.array([[1, 0, 30], [0, 1, 40], [0, 0, 1]], float)
    crop, box = crop_region(image, transform_points(q, H), config)
    assert box == [38, 58, 82, 102]
    assert crop.shape == (44, 44, 3)


def test_infer_records_image_read_failure(config, tmp_path):
    from cvdoc.pipeline import infer

    bad = tmp_path / "bad.png"
    bad.write_text("broken")
    result = infer(bad, "baseline", tmp_path / "out", config)
    assert result.status == Status.IMAGE_READ_ERROR
    assert (tmp_path / "out/prediction.json").is_file()
