#!/usr/bin/env python3
"""Download and checksum the public datasets used by DREAM Figure 5.

Large files are written to ``.part`` paths and atomically renamed only after
their byte count and MD5 match the versioned Figshare record. Interrupted
downloads resume when the server honors HTTP Range requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "dream-consciousness" / "raw" / "fig5-original"
MANIFEST_PATH = ROOT / "data" / "dream-consciousness" / "fig5-source-files.json"
ARTICLES = {
    "degen_ma": {"article_id": 22086266, "version": 2},
    "oudiette": {"article_id": 22210684, "version": 1},
    "siclari": {"article_id": 23306054, "version": 3},
    "turku": {"article_id": 23274596, "version": 2},
}
PRINT_LOCK = threading.Lock()


def say(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def md5(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def article_manifest(name: str) -> dict[str, object]:
    expected = ARTICLES[name]
    response = requests.get(
        f"https://api.figshare.com/v2/articles/{expected['article_id']}", timeout=60
    )
    response.raise_for_status()
    article = response.json()
    if article["version"] != expected["version"]:
        raise RuntimeError(
            f"{name}: expected article version {expected['version']}, "
            f"received {article['version']}"
        )
    return {
        "dataset": name,
        "article_id": article["id"],
        "version": article["version"],
        "doi": article["doi"],
        "license": article["license"]["name"],
        "files": [
            {
                "id": item["id"],
                "name": item["name"],
                "size": item["size"],
                "md5": item["computed_md5"],
                "download_url": item["download_url"],
            }
            for item in article["files"]
        ],
    }


def download(dataset: str, item: dict[str, object]) -> Path:
    target_dir = RAW_ROOT / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / str(item["name"])
    partial = target.with_name(target.name + ".part")
    expected_size = int(item["size"])
    expected_md5 = str(item["md5"])

    if target.exists():
        if target.stat().st_size == expected_size and md5(target) == expected_md5:
            say(f"verified existing {dataset}/{target.name}")
            return target
        raise RuntimeError(f"Existing final file failed verification: {target}")

    resume_at = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
    with requests.get(
        str(item["download_url"]), headers=headers, stream=True, timeout=(30, 120)
    ) as response:
        response.raise_for_status()
        resumed = resume_at > 0 and response.status_code == 206
        if resume_at and not resumed:
            say(f"server did not resume {target.name}; restarting")
            resume_at = 0
        mode = "ab" if resumed else "wb"
        downloaded = resume_at
        next_report = ((downloaded // (512 * 1024 * 1024)) + 1) * 512 * 1024 * 1024
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    say(
                        f"{dataset}/{target.name}: "
                        f"{downloaded / 2**30:.1f}/{expected_size / 2**30:.1f} GiB"
                    )
                    next_report += 512 * 1024 * 1024

    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {partial}: {partial.stat().st_size} != {expected_size}"
        )
    actual_md5 = md5(partial)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {partial}: {actual_md5} != {expected_md5}")
    os.replace(partial, target)
    say(f"complete {dataset}/{target.name}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="*",
        default=None,
        metavar="DATASET",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    started = time.time()
    selected_datasets = args.datasets or sorted(ARTICLES)
    unknown = sorted(set(selected_datasets) - set(ARTICLES))
    if unknown:
        parser.error(
            f"unknown dataset(s): {', '.join(unknown)}; "
            f"choose from {', '.join(sorted(ARTICLES))}"
        )
    manifests = [article_manifest(name) for name in selected_datasets]
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, object]] = {}
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(encoding="utf-8") as handle:
            existing = {
                str(item["dataset"]): item
                for item in json.load(handle).get("articles", [])
            }
    existing.update({str(item["dataset"]): item for item in manifests})
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "retrieved": "2026-08-02",
                "articles": [existing[key] for key in sorted(existing)],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for manifest in manifests:
            for item in manifest["files"]:
                jobs.append(executor.submit(download, str(manifest["dataset"]), item))
        for future in as_completed(jobs):
            future.result()

    say(f"Verified {len(jobs)} files in {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
