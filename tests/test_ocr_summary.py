import pandas as pd
import pytest

from cvdoc.experiments import BRANCHES, summarize_ocr
from cvdoc.metrics import text_metrics


def test_ocr_complete_and_conditional_denominators(tmp_path):
    rows = []
    for method in BRANCHES:
        for index, (reference, prediction, status) in enumerate(
            [("abc", "abc", "ok"), ("abcdef", "", "ocr_error")]
        ):
            rows.append(
                {
                    "document_id": f"d{index}",
                    "view_id": f"d{index}_mild",
                    "condition": "mild",
                    "entity_id": "e1",
                    "method": method,
                    "status": status,
                    "pipeline_status": status,
                    "source_box_height": 20,
                    **text_metrics(reference, prediction),
                }
            )
    summarize_ocr(pd.DataFrame(rows), tmp_path)
    summary = pd.read_csv(tmp_path / "ocr_summary.csv")
    selected = summary[(summary.method == "E3_refined") & (summary.condition == "mild")]
    complete = selected[selected.denominator == "complete"].iloc[0]
    conditional = selected[selected.denominator == "conditional_success"].iloc[0]
    assert complete.CER == pytest.approx(6 / 9)
    assert complete.entities == 2 and conditional.entities == 1
    assert conditional.CER == 0 and conditional.mean_document_CER == 0
    assert complete.mean_document_CER == 0.5
