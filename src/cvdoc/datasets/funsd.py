"""FUNSD published entity annotations and untouched official test split."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .manifest import download, extract_archive, provenance, save_manifest, sha256

URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
TERMS = "https://guillaumejaume.github.io/FUNSD/work/"


def parse_annotation(path: Path) -> tuple[list[dict], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("form"), list):
        raise ValueError(f"Invalid FUNSD form annotation: {path}")
    entities, excluded = [], {"empty_text": 0, "invalid_box": 0}
    ids = set()
    for index, item in enumerate(payload["form"]):
        text = item.get("text", "")
        if not isinstance(text, str) or not text.strip():
            excluded["empty_text"] += 1
            continue
        box = item.get("box")
        try:
            box = np.asarray(box, dtype=float)
            valid = (
                box.shape == (4,)
                and np.isfinite(box).all()
                and box[0] >= 0
                and box[1] >= 0
                and box[2] > box[0]
                and box[3] > box[1]
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            excluded["invalid_box"] += 1
            continue
        identity = str(item.get("id", index))
        if identity in ids:
            raise ValueError(f"Duplicate entity ID {identity} in {path}")
        ids.add(identity)
        entities.append({"entity_id": identity, "text": text, "box": box.tolist()})
    return entities, excluded


def assign_splits(train_ids: list[str], test_ids: list[str], seed: int = 42) -> dict[str, str]:
    if set(train_ids) & set(test_ids) or len(set(train_ids)) != len(train_ids):
        raise ValueError("Overlapping or duplicate FUNSD IDs")
    order = np.random.default_rng(seed).permutation(sorted(train_ids)).tolist()
    n_val = 30 if len(order) == 149 else round(len(order) * 30 / 149)
    return {
        **{identity: ("val" if i < n_val else "dev") for i, identity in enumerate(order)},
        **{identity: "test" for identity in sorted(test_ids)},
    }


def prepare(
    data_dir: Path, accept_terms: bool, source: Path | None = None, fetch: bool = False
) -> Path:
    print(f"FUNSD: noncommercial research/education only; age 18+. Terms: {TERMS}")
    if not accept_terms:
        raise ValueError("Personal review/acceptance required: re-run with --accept-terms")
    if fetch or source is None:
        source = download(URL, data_dir / "raw/funsd.zip")
    if not source.is_file():
        raise ValueError(f"FUNSD archive not found: {source}")
    root = data_dir / "raw/funsd"
    count = extract_archive(source, root, lambda name: "__MACOSX" not in Path(name).parts)
    provenance(
        data_dir,
        "funsd",
        {
            "url": URL,
            "terms_url": TERMS,
            "license": "Noncommercial research/education; age 18+",
            "terms_accepted": True,
            "archive_sha256": sha256(source),
            "archive": source.name,
            "extracted_files": count,
            "status": "extracted",
            "warnings": [],
        },
    )
    return root


def make_manifest(data_dir: Path, seed: int) -> list[dict]:
    root = data_dir / "raw/funsd"
    examples = []
    for original in ("training_data", "testing_data"):
        matches = sorted(
            p for p in root.rglob(f"{original}/annotations") if "__MACOSX" not in p.parts
        )
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous FUNSD {original}/annotations under {root}")
        annotation_root = matches[0]
        paths = sorted(annotation_root.glob("*.json"))
        if not paths:
            raise ValueError(f"No annotations in {annotation_root}")
        for annotation in paths:
            images = sorted((annotation_root.parent / "images").glob(annotation.stem + ".*"))
            if len(images) != 1:
                raise ValueError(f"Missing or ambiguous image for {annotation}")
            entities, excluded = parse_annotation(annotation)
            with Image.open(images[0]) as image:
                w, h = image.size
            examples.append(
                {
                    "image_id": annotation.stem,
                    "document_id": annotation.stem,
                    "source": "funsd",
                    "path": str(images[0].resolve()),
                    "annotation_path": str(annotation.resolve()),
                    "original_split": original,
                    "width": w,
                    "height": h,
                    "entities": len(entities),
                    "excluded": json.dumps(excluded),
                    "image_sha256": sha256(images[0]),
                    "annotation_sha256": sha256(annotation),
                    "seed": seed,
                }
            )
    train = [r["image_id"] for r in examples if r["original_split"] == "training_data"]
    test = [r["image_id"] for r in examples if r["original_split"] == "testing_data"]
    if len(train) != 149 or len(test) != 50:
        raise ValueError(f"Expected official 149/50 FUNSD pages, found {len(train)}/{len(test)}")
    splits = assign_splits(train, test, seed)
    for record in examples:
        record["split"] = splits[record["image_id"]]
    save_manifest(examples, data_dir / "processed/funsd_manifest.csv")
    provenance(
        data_dir,
        "funsd",
        {
            "official_train": len(train),
            "official_test": len(test),
            "manifest_counts": {
                s: sum(r["split"] == s for r in examples) for s in ("dev", "val", "test")
            },
            "seed": seed,
        },
    )
    return examples
