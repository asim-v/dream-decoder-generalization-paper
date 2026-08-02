#!/usr/bin/env python3
"""Safely extract checksum-verified DREAM Figure 5 source archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "dream-consciousness" / "raw" / "fig5-original"
EXTRACT_ROOT = ROOT / "data" / "dream-consciousness" / "extracted" / "fig5-original"
DATASETS = ("degen_ma", "oudiette", "siclari", "turku")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def extract_archive(dataset: str, archive: Path) -> str:
    target = (EXTRACT_ROOT / dataset).resolve()
    marker_dir = target / ".markers"
    marker = marker_dir / f"{archive.name}.json"
    archive_md5 = md5(archive)
    if marker.exists():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("md5") == archive_md5:
            return f"already extracted {dataset}/{archive.name}"

    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zipped:
        for info in zipped.infolist():
            destination = (target / info.filename).resolve()
            if destination != target and target not in destination.parents:
                raise RuntimeError(f"Unsafe archive member: {info.filename}")
            zipped.extract(info, target)

    marker_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"archive": archive.name, "md5": archive_md5}, indent=2) + "\n",
        encoding="utf-8",
    )
    return f"extracted {dataset}/{archive.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", metavar="DATASET")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    selected = args.datasets or list(DATASETS)
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        parser.error(
            f"unknown dataset(s): {', '.join(unknown)}; "
            f"choose from {', '.join(DATASETS)}"
        )

    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for dataset in selected:
            archives = sorted((RAW_ROOT / dataset).glob("*.zip"))
            if not archives:
                raise FileNotFoundError(f"No ZIP archives found for {dataset}")
            for archive in archives:
                jobs.append(executor.submit(extract_archive, dataset, archive))
        for completed, future in enumerate(as_completed(jobs), start=1):
            print(f"{completed}/{len(jobs)} {future.result()}", flush=True)


if __name__ == "__main__":
    main()
