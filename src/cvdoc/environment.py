"""Dependency and provenance diagnostics."""

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytesseract


def configure_tesseract(config: dict) -> str | None:
    command = config.get("tesseract_cmd") or shutil.which("tesseract")
    if not command and sys.platform == "win32":
        for candidate in (
            Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
            Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe",
        ):
            if candidate.is_file():
                command = str(candidate)
                break
    if command:
        pytesseract.pytesseract.tesseract_cmd = str(command)
    return command


def environment(config: dict) -> dict:
    command = configure_tesseract(config)
    version, languages, error = None, [], None
    if command:
        try:
            version = str(pytesseract.get_tesseract_version())
            languages = pytesseract.get_languages(config="")
        except (OSError, RuntimeError) as exc:
            error = str(exc)
    packages = {}
    for package in (
        "cvdoc",
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "pandas",
        "scipy",
        "matplotlib",
        "PyYAML",
        "pytesseract",
        "jiwer",
        "requests",
        "pytest",
        "ruff",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        git = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.TimeoutExpired):
        git = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "packages": packages,
        "git_commit": git,
        "tesseract_command": command,
        "tesseract_version": version,
        "tesseract_languages": languages,
        "tesseract_error": error,
        "ocr_ready": version is not None and "eng" in languages,
        "data_dir": str(Path(config["data_dir"]).resolve()),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
    }
