import pytest

from cvdoc.config import load_config


def test_config_defaults():
    assert load_config()["detect_max_side"] == 1400


def test_bad_config(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("gaussian_kernel: 4")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    "setting",
    [
        "canny_low: bad",
        "refine_corridor_min: .nan",
        "clahe: maybe",
        "ocr_workers: 0",
        "approx_epsilon: bad",
        "canny_fallbacks: bad",
    ],
)
def test_malformed_config_is_actionable(tmp_path, setting):
    path = tmp_path / "bad.yaml"
    path.write_text(setting)
    with pytest.raises(ValueError):
        load_config(path)
