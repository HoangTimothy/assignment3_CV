import pandas as pd
import pytest

from cvdoc.report import build_report, validate_rows


def test_report_refuses_missing_run(tmp_path):
    with pytest.raises(ValueError):
        build_report(tmp_path)


def test_report_rejects_coverage_mismatch(tmp_path):
    pd.DataFrame([{"image_id": "one"}, {"image_id": "two"}]).to_csv(
        tmp_path / "selected_manifest.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "image_id": "one",
                "method": "baseline",
                "status": "ok",
                "polygon_iou": 1,
                "NCE": 0,
                "runtime_ms": 1,
            }
        ]
    ).to_csv(tmp_path / "geometry_metrics.csv", index=False)
    with pytest.raises(ValueError, match="manifest"):
        validate_rows(tmp_path, {"status": "complete", "kind": "geometry_baseline"})
