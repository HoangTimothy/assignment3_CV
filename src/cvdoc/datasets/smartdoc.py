"""SmartDoc v2 named-column parser and document-disjoint quantile sampling."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

from .manifest import download, extract_archive, provenance, safe_path, save_manifest, sha256

REPOSITORY = "https://github.com/jchazalon/smartdoc15-ch1-dataset"
TAG = "v2.0.0"
COLUMNS = [
    "bg_name",
    "model_name",
    "modeltype_name",
    "image_path",
    "frame_index",
    "model_width",
    "model_height",
    "tl_x",
    "tl_y",
    "bl_x",
    "bl_y",
    "br_x",
    "br_y",
    "tr_x",
    "tr_y",
]


def parse_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"SmartDoc missing named columns: {sorted(missing)}")
    for name in [
        n
        for n in COLUMNS
        if n.endswith(("_x", "_y")) or n in ("frame_index", "model_width", "model_height")
    ]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name]).all():
            raise ValueError(f"Nonfinite SmartDoc column: {name}")
    if (frame[["model_width", "model_height"]] <= 0).any().any():
        raise ValueError("SmartDoc model dimensions must be positive")
    if frame["image_path"].duplicated().any():
        raise ValueError("Duplicate SmartDoc image paths")
    for value in frame["image_path"]:
        safe_path(path.parent, value)
    return frame


def gt_quad(row: dict | pd.Series) -> np.ndarray:
    # Preserve semantic labels; never choose the best GT permutation.
    return np.array(
        [[row[f"{p}_x"], row[f"{p}_y"]] for p in ("tl", "tr", "br", "bl")], dtype=np.float64
    )


def split_for_model(model: str) -> str:
    match = re.fullmatch(r"([a-z]+)(00[1-5])", model)
    if not match:
        raise ValueError(f"Unexpected SmartDoc model identity: {model}")
    return {1: "dev", 2: "dev", 3: "dev", 4: "val", 5: "test"}[int(match[2])]


def sample_frames(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    selected = []
    for (_, model), group in frame.groupby(["bg_name", "model_name"], sort=True):
        group = group.sort_values(["frame_index", "image_path"])
        if len(group) < 4 or group["frame_index"].duplicated().any():
            raise ValueError(f"Need at least four unique frames for {model}")
        used = set()
        for quantile in (0.2, 0.4, 0.6, 0.8):
            target = quantile * (len(group) - 1)
            index = min(
                (i for i in range(len(group)) if i not in used), key=lambda i: (abs(i - target), i)
            )
            used.add(index)
            record = group.iloc[index].to_dict()
            record.update(split=split_for_model(model), seed=seed)
            selected.append(record)
    return pd.DataFrame(selected)


def find_metadata(root: Path) -> Path:
    paths = sorted(root.rglob("metadata.csv.gz")) + sorted(root.rglob("metadata.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one SmartDoc metadata file under {root}, found {len(paths)}")
    return paths[0]


def prepare(
    data_dir: Path,
    source: Path | None = None,
    fetch: bool = False,
    metadata_only: Path | None = None,
) -> Path:
    root = data_dir / "raw/smartdoc"
    if metadata_only:
        metadata = metadata_only.resolve()
        frame = parse_metadata(metadata)
        provenance(
            data_dir,
            "smartdoc",
            {
                "status": "metadata_only",
                "rows": len(frame),
                "metadata_path": str(metadata),
                "license": "CC BY 4.0",
                "url": REPOSITORY,
                "metadata_sha256": sha256(metadata),
                "warnings": ["Images not validated"],
            },
        )
        return metadata
    if fetch:
        response = requests.get(
            f"https://api.github.com/repos/jchazalon/smartdoc15-ch1-dataset/releases/tags/{TAG}",
            timeout=30,
        )
        response.raise_for_status()
        assets = {a["name"]: a for a in response.json()["assets"]}
        if "frames.tar.gz" not in assets or "sha256.chksum" not in assets:
            raise ValueError(f"Release {TAG} lacks expected assets")
        asset = assets["frames.tar.gz"]
        checksums = requests.get(assets["sha256.chksum"]["browser_download_url"], timeout=30)
        checksums.raise_for_status()
        expected = next(
            (
                line.split()[0]
                for line in checksums.text.splitlines()
                if line.split()[-1].lstrip("*") == "frames.tar.gz"
            ),
            None,
        )
        if expected is None:
            raise ValueError("No advertised frames.tar.gz checksum")
        source = download(
            asset["browser_download_url"], data_dir / "raw/frames.tar.gz", asset["size"], expected
        )
    if source is None or not source.is_file():
        raise ValueError("Use prepare-smartdoc --source frames.tar.gz or --download")
    extract_archive(source, root, lambda name: Path(name).name == "metadata.csv.gz")
    metadata = find_metadata(root)
    sampled = sample_frames(parse_metadata(metadata))
    wanted = set(sampled["image_path"])
    # Extract selected frames only, plus source credit and version files.
    prefix = metadata.parent.relative_to(root).as_posix()

    def select(name: str) -> bool:
        relative = name.removeprefix("./")
        if prefix != "." and relative.startswith(prefix + "/"):
            relative = relative[len(prefix) + 1 :]
        return relative in wanted or Path(name).name in ("README.md", "LICENCE", "VERSION")

    extract_archive(source, root, select)
    provenance(
        data_dir,
        "smartdoc",
        {
            "url": REPOSITORY,
            "license": "CC BY 4.0",
            "archive": source.name,
            "archive_sha256": sha256(source),
            "release": TAG,
            "status": "selected_frames_extracted",
            "source_rows": len(parse_metadata(metadata)),
            "selected_rows": len(sampled),
            "seed": 42,
            "warnings": [],
        },
    )
    return metadata


def make_manifest(data_dir: Path, seed: int) -> list[dict]:
    metadata = find_metadata(data_dir / "raw/smartdoc")
    sampled = sample_frames(parse_metadata(metadata), seed)
    records = []
    for _, row in sampled.iterrows():
        path = safe_path(metadata.parent, row["image_path"])
        if not path.is_file():
            raise ValueError(f"Missing selected SmartDoc frame: {path}")
        with Image.open(path) as image:
            w, h = image.size
        quad = gt_quad(row)
        if not ((quad >= -2).all() and (quad[:, 0] <= w + 2).all() and (quad[:, 1] <= h + 2).all()):
            raise ValueError(f"GT dimensions disagree with actual image {path}")
        records.append(
            {
                "image_id": row["image_path"],
                "document_id": row["model_name"],
                "source": "smartdoc",
                "path": str(path),
                "width": w,
                "height": h,
                "split": row["split"],
                "background": row["bg_name"],
                "class": row["modeltype_name"],
                "frame_index": row["frame_index"],
                "model_width": row["model_width"],
                "model_height": row["model_height"],
                "gt_quad": json.dumps(quad.tolist()),
                "seed": seed,
            }
        )
    save_manifest(records, data_dir / "processed/smartdoc_manifest.csv")
    counts = pd.DataFrame(records)["split"].value_counts().to_dict()
    provenance(
        data_dir,
        "smartdoc",
        {
            "manifest_counts": counts,
            "seed": seed,
            "warnings": [] if len(records) == 600 else ["Incomplete source combinations"],
        },
    )
    return records
