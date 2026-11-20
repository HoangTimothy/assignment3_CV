"""Secure archives, atomic downloads, provenance, and deterministic manifests."""

import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

import pandas as pd
import requests

from .. import __version__
from ..io import write_json


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, name: str) -> Path:
    # Reject Windows paths even when the adapter is running on Linux.
    normalized = name.replace("\\", "/")
    parts = PurePosixPath(normalized)
    if parts.is_absolute() or ".." in parts.parts or ":" in normalized:
        raise ValueError(f"Unsafe archive path: {name}")
    target = (root / normalized).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Archive path escapes destination: {name}")
    return target


def extract_archive(
    source: Path, destination: Path, select: Callable[[str], bool] | None = None
) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            # Validate every member before writing any member.
            for member in archive.infolist():
                safe_path(destination, member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are forbidden")
            needed = sum(
                m.file_size
                for m in archive.infolist()
                if not m.is_dir() and (select is None or select(m.filename))
            )
            if needed > shutil.disk_usage(destination).free:
                raise OSError(f"Insufficient disk: extraction needs {needed} bytes")
            for member in archive.infolist():
                if member.is_dir() or (select and not select(member.filename)):
                    continue
                target = safe_path(destination, member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
    else:
        with tarfile.open(source, "r:*") as archive:
            members = archive.getmembers()
            for member in members:
                safe_path(destination, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Archive links and special files are forbidden")
            needed = sum(
                m.size for m in members if m.isfile() and (select is None or select(m.name))
            )
            if needed > shutil.disk_usage(destination).free:
                raise OSError(f"Insufficient disk: extraction needs {needed} bytes")
            for member in members:
                if not member.isfile() or (select and not select(member.name)):
                    continue
                target = safe_path(destination, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
    return count


def download(
    url: str, destination: Path, expected_size: int | None = None, expected_hash: str | None = None
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (not expected_hash or sha256(destination) == expected_hash):
        if expected_size is None or destination.stat().st_size == expected_size:
            return destination
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        size = expected_size or int(response.headers.get("Content-Length", 0))
        print(f"Downloading {url}: {size or 'unknown'} bytes", flush=True)
        if size and shutil.disk_usage(destination.parent).free < size * 2:
            raise OSError("Insufficient disk for download and working space")
        partial = destination.with_suffix(destination.suffix + ".part")
        with partial.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
        if size and partial.stat().st_size != size:
            raise ValueError("Download length mismatch; partial file retained")
        if expected_hash and sha256(partial) != expected_hash:
            raise ValueError("Download SHA-256 mismatch")
        partial.replace(destination)
    return destination


def provenance(data_dir: Path, dataset: str, values: dict) -> None:
    path = data_dir / "processed/dataset_provenance.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current[dataset] = {**current.get(dataset, {}), **values, "software_version": __version__}
    write_json(path, current)


def save_manifest(records: list[dict], path: Path) -> None:
    if not records:
        raise ValueError("Cannot write an empty manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def load_manifest(data_dir: Path, dataset: str, split: str | None = None) -> pd.DataFrame:
    path = data_dir / f"processed/{dataset}_manifest.csv"
    if not path.is_file():
        raise ValueError(f"Missing {path}. Run data prepare-{dataset}, then data make-splits")
    frame = pd.read_csv(path, keep_default_na=False, dtype={"image_id": str, "document_id": str})
    if split:
        frame = frame[frame["split"] == split]
    if frame.empty:
        raise ValueError(f"No {dataset} records for split {split}")
    missing = [p for p in frame["path"] if not Path(p).is_file()]
    if missing:
        raise ValueError(
            f"{len(missing)} missing images; first: {missing[0]}. Prepare actual frames"
        )
    return frame
