"""Export aggregate artifacts only; never include dataset images or transcripts."""

import argparse
import zipfile
from pathlib import Path

ALLOWED = {
    "summary.json",
    "ocr_summary.csv",
    "ocr_paired.csv",
    "geometry_summary.csv",
    "runtime_summary.csv",
    "failure_summary.csv",
    "config.yaml",
    "environment.json",
    "run.json",
}


def export(run: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.iterdir()):
            if path.name in ALLOWED and path.is_file():
                archive.write(path, path.name)
        for path in sorted((run / "figures").glob("*.png")):
            archive.write(path, "figures/" + path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.run, args.destination)
