import cv2
import numpy as np
import pytest

from cvdoc.environment import environment
from cvdoc.ocr import recognize
from cvdoc.schemas import Status


@pytest.mark.ocr
def test_tesseract_integration(config):
    if not environment(config)["ocr_ready"]:
        pytest.skip("Tesseract with English traineddata is unavailable")
    image = np.full((180, 650, 3), 255, np.uint8)
    cv2.putText(image, "DOCUMENT TEST 123", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 3)
    result = recognize(image, config, roi=True)
    assert result.status == Status.OK
    assert result.text.strip() and result.elapsed_ms > 0 and result.psm == 6
