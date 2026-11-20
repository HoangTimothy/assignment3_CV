"""Fixed Tesseract configuration with explicit empty/error predictions."""

from functools import lru_cache
from time import perf_counter

import cv2
import numpy as np
import pytesseract

from .environment import configure_tesseract
from .schemas import OCRResult, Status


@lru_cache(maxsize=4)
def engine_info(command: str) -> tuple[str, tuple[str, ...]]:
    pytesseract.pytesseract.tesseract_cmd = command
    return str(pytesseract.get_tesseract_version()), tuple(pytesseract.get_languages(config=""))


def require_ocr(config: dict) -> str:
    command = configure_tesseract(config)
    if not command:
        raise ValueError("Install Tesseract 5 and English traineddata; run cvdoc doctor")
    try:
        version, languages = engine_info(command)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot run Tesseract: {exc}") from exc
    if config["ocr_language"] not in languages:
        raise ValueError("English traineddata missing; install tesseract-ocr-eng")
    return version


def recognize(image: np.ndarray, config: dict, roi: bool = False) -> OCRResult:
    started = perf_counter()
    psm = config["ocr_roi_psm"] if roi else config["ocr_page_psm"]
    version = "unavailable"
    try:
        version = require_ocr(config)
        if image is None or image.size == 0:
            raise ValueError("Empty OCR crop")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        scale = config["ocr_scale"] if roi else 1.0
        if scale != 1:
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(
            rgb, lang=config["ocr_language"], config=f"--psm {psm}", timeout=config["ocr_timeout"]
        )
        return OCRResult(
            Status.OK, text, version, config["ocr_language"], psm, (perf_counter() - started) * 1000
        )
    except (ValueError, OSError, RuntimeError, pytesseract.TesseractError, cv2.error) as exc:
        return OCRResult(
            Status.OCR_ERROR,
            "",
            version,
            config["ocr_language"],
            psm,
            (perf_counter() - started) * 1000,
            str(exc),
        )
