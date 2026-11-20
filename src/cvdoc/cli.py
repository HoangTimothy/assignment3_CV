"""Command-line orchestration; vision algorithms live in independent modules."""

import argparse
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests

from .config import load_config
from .datasets import funsd, smartdoc
from .datasets.manifest import load_manifest
from .environment import environment
from .io import read_image
from .visualize import montage, overlay


def common(command: argparse.ArgumentParser, limit: bool = True) -> None:
    command.add_argument("--config", default=argparse.SUPPRESS, help="YAML overrides")
    if limit:
        command.add_argument("--max-items", type=int, help="Bound a CPU smoke run")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="cvdoc", description="CPU document geometry and OCR")
    result.add_argument("--config", default=None)
    commands = result.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect CPU dependencies and OCR")
    common(doctor, False)
    doctor.add_argument("--json", action="store_true")
    data = commands.add_parser("data", help="Prepare licensed public datasets")
    common(data, False)
    sub = data.add_subparsers(dest="action", required=True)
    prepare_funsd = sub.add_parser("prepare-funsd", help="Requires personal terms acceptance")
    common(prepare_funsd)
    prepare_funsd.add_argument("--accept-terms", action="store_true")
    prepare_funsd.add_argument("--source", type=Path)
    prepare_funsd.add_argument("--download", action="store_true")
    prepare_smartdoc = sub.add_parser("prepare-smartdoc", help="SmartDoc v2 selected frames")
    common(prepare_smartdoc)
    prepare_smartdoc.add_argument("--source", type=Path)
    prepare_smartdoc.add_argument("--download", action="store_true")
    prepare_smartdoc.add_argument("--metadata-only", type=Path)
    splits = sub.add_parser("make-splits", help="Create deterministic split manifests")
    common(splits)
    splits.add_argument("--seed", type=int, default=None)
    inspect = sub.add_parser("inspect", help="Save deterministic real-image/GT contact sheets")
    common(inspect)
    inspect.add_argument("--dataset", choices=("smartdoc", "funsd"), required=True)
    inspect.add_argument("--n", type=int, default=12)
    infer = commands.add_parser("infer", help="Detect, rectify, and OCR one input image")
    common(infer)
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--method", choices=("baseline", "refined"), default="refined")
    infer.add_argument("--out", type=Path, required=True)
    infer.add_argument("--save-stages", action="store_true")
    benchmark = commands.add_parser("benchmark", help="Separate geometry and controlled OCR tracks")
    common(benchmark, False)
    sub = benchmark.add_subparsers(dest="track", required=True)
    for name in ("geometry", "ocr", "all"):
        command = sub.add_parser(name, help=f"Run {name} evaluation")
        common(command)
        command.add_argument("--split", choices=("dev", "val", "test"), default="val")
        if name == "geometry":
            command.add_argument("--method", choices=("baseline", "refined"), default="refined")
        if name in ("ocr", "all"):
            command.add_argument("--variants", default="frontal,mild,strong")
    report = commands.add_parser("report", help="Build verified aggregate tables and plots")
    common(report, False)
    build = report.add_subparsers(dest="action", required=True).add_parser("build")
    common(build)
    build.add_argument("--run-dir", type=Path, required=True)
    return result


def inspect_dataset(data_dir: Path, dataset: str, n: int, config: dict) -> Path:
    frame = load_manifest(data_dir, dataset)
    selected = frame.sample(n=min(n, len(frame)), random_state=config["seed"])
    panels = []
    for row in selected.to_dict("records"):
        image = read_image(row["path"])
        if dataset == "smartdoc":
            image = overlay(
                image, np.array(json.loads(row["gt_quad"])), (0, 0, 255), row["document_id"]
            )
        else:
            entities, _ = funsd.parse_annotation(Path(row["annotation_path"]))
            for entity in entities:
                x0, y0, x1, y1 = np.rint(entity["box"]).astype(int)
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1)
        panels.append((f"{row['image_id']} | {row['split']}", image))
    path = data_dir / "processed" / f"{dataset}_inspection.jpg"
    montage(path, panels)
    selected.to_csv(path.with_suffix(".csv"), index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        maximum = getattr(args, "max_items", None)
        if maximum is not None and maximum < 1:
            raise ValueError("--max-items must be positive")
        if args.command == "doctor":
            info = environment(config)
            print(json.dumps(info, indent=2))
            return 0 if info["ocr_ready"] else 1
        if args.command == "data":
            root = Path(config["data_dir"])
            if args.action == "prepare-funsd":
                print(funsd.prepare(root, args.accept_terms, args.source, args.download))
            elif args.action == "prepare-smartdoc":
                print(smartdoc.prepare(root, args.source, args.download, args.metadata_only))
            elif args.action == "make-splits":
                seed = config["seed"] if args.seed is None else args.seed
                found = False
                for name, module in (("smartdoc", smartdoc), ("funsd", funsd)):
                    if (root / "raw" / name).is_dir():
                        records = module.make_manifest(root, seed)
                        print(f"{name}: {len(records)} samples")
                        found = True
                if not found:
                    raise ValueError(
                        "No datasets prepared. Run data prepare-smartdoc or prepare-funsd"
                    )
            elif args.action == "inspect":
                if args.n < 1:
                    raise ValueError("--n must be positive")
                print(inspect_dataset(root, args.dataset, min(args.n, maximum or args.n), config))
        elif args.command == "infer":
            from .pipeline import infer

            result = infer(args.input, args.method, args.out, config, args.save_stages)
            print(f"{result.status.value}: {args.out}")
            return 0 if result.status.value == "ok" else 1
        elif args.command == "benchmark":
            from .experiments import benchmark_geometry, benchmark_ocr

            runs = []
            if args.track in ("geometry", "all"):
                methods = ("baseline", "refined") if args.track == "all" else (args.method,)
                for method in methods:
                    runs.append(benchmark_geometry(config, args.split, method, maximum))
            if args.track in ("ocr", "all"):
                runs.append(benchmark_ocr(config, args.split, args.variants.split(","), maximum))
            from .report import build_report

            for run in runs:
                build_report(run)
                print(run)
        elif args.command == "report":
            from .report import build_report

            print(build_report(args.run_dir))
        return 0
    except (
        ValueError,
        OSError,
        RuntimeError,
        requests.RequestException,
        cv2.error,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"cvdoc: {exc}", file=sys.stderr)
        return 2
