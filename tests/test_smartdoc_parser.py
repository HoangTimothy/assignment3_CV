import numpy as np
import pandas as pd
import pytest

from cvdoc.datasets.smartdoc import gt_quad, parse_metadata, sample_frames


def metadata():
    rows = []
    for model in range(1, 6):
        for frame in range(1, 11):
            rows.append(
                dict(
                    bg_name="background01",
                    model_name=f"letter{model:03}",
                    modeltype_name="letter",
                    image_path=f"letter{model:03}/frame_{frame:04}.jpeg",
                    frame_index=frame,
                    model_width=2100,
                    model_height=2970,
                    tl_x=1,
                    tl_y=2,
                    bl_x=1,
                    bl_y=10,
                    br_x=8,
                    br_y=10,
                    tr_x=8,
                    tr_y=2,
                )
            )
    return pd.DataFrame(rows)


def test_smartdoc_gt_order_conversion():
    np.testing.assert_array_equal(gt_quad(metadata().iloc[0]), [[1, 2], [8, 2], [8, 10], [1, 10]])


def test_smartdoc_csv_parser(tmp_path):
    path = tmp_path / "metadata.csv.gz"
    metadata().to_csv(path, index=False)
    assert len(parse_metadata(path)) == 50


def test_smartdoc_split_group_disjoint():
    frame = sample_frames(metadata())
    assert frame.groupby("model_name")["split"].nunique().max() == 1
    assert frame["split"].value_counts().to_dict() == {"dev": 12, "val": 4, "test": 4}


def test_smartdoc_sampling_reproducible():
    pd.testing.assert_frame_equal(sample_frames(metadata()), sample_frames(metadata().iloc[::-1]))
    assert sample_frames(metadata()).groupby("model_name").size().eq(4).all()


def test_insufficient_frames():
    with pytest.raises(ValueError):
        sample_frames(metadata().iloc[:3])
