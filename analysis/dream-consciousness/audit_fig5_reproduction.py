#!/usr/bin/env python3
"""Audit whether the public DREAM Figure 5 code runs on its public inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import mne


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "dream-consciousness"
OUT = ROOT / "outputs" / "dream-consciousness"

DATASETS = {
    "Zhang_Wamsley_2019": {
        "root": DATA / "extracted" / "zhang-wamsley" / "Zhang & Wamsley 2019",
        "expected_channels": ["F4-REF", "C4-REF", "O2-REF", "A1-REF", "A2-REF"],
        "code_delimiter": ",",
    },
    "DeGenearoMA": {
        "root": DATA / "extracted" / "fig5-original" / "degen_ma" / "Multiple awakenings",
        "expected_channels": ["F4", "C4", "O2"],
        "code_delimiter": ";",
    },
    "DeGenaro_YoungAdults": {
        "root": DATA / "extracted" / "young-adults" / "Dream_YoungAdults",
        "expected_channels": ["F4", "C4", "O2"],
        "code_delimiter": ",",
    },
    "SiclariMA": {
        "root": DATA / "extracted" / "fig5-original" / "siclari" / "Tononi Serial Awakenings",
        "expected_channels": ["F4-M1", "C4-M1", "O2-M1"],
        "code_delimiter": ",",
    },
    "rem_Turku": {
        "root": DATA / "extracted" / "fig5-original" / "turku" / "REM_Turku",
        "expected_channels": ["F4", "C4", "O2"],
        "code_delimiter": ",",
    },
}


def read_records(path: Path) -> tuple[list[dict[str, str]], str]:
    first = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
    actual = ";" if first.count(";") > first.count(",") else ","
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=actual)), actual


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalize_record_filename(dataset: str, value: str) -> str:
    name = Path(value.replace("\\", "/")).name
    if dataset == "rem_Turku" and not name.lower().endswith(".edf"):
        name += ".edf"
    return name


def main() -> None:
    exclusions = {
        (row["dataset"], row["filename"])
        for row in read_records(DATA / "exclusions.csv")[0]
    }
    rows: list[dict[str, object]] = []
    for dataset, config in DATASETS.items():
        root = config["root"]
        records, actual_delimiter = read_records(root / "Records.csv")
        edfs = sorted((root / "Data" / "PSG").glob("*.edf"))
        names = {path.name for path in edfs}
        expected_names = {
            normalize_record_filename(dataset, row["Filename"]) for row in records
        }
        missing = sorted(expected_names - names)
        extra = sorted(names - expected_names)
        sample = edfs[0]
        raw = mne.io.read_raw_edf(sample, preload=False, verbose="ERROR")
        channels = set(raw.ch_names)
        strict = [row for row in records if row["Experience"] in {"0", "2"}]
        usable_strict = [
            row
            for row in strict
            if normalize_record_filename(dataset, row["Filename"]) in names
            and (dataset, normalize_record_filename(dataset, row["Filename"]))
            not in exclusions
        ]
        label_counts = Counter(row["Experience"] for row in usable_strict)
        stage_counts = Counter(row["Last sleep stage"] for row in usable_strict)
        formatted_count = sum(path.name.startswith("formatted_") for path in edfs)
        rows.append(
            {
                "dataset": dataset,
                "records": len(records),
                "edf_files": len(edfs),
                "strict_usable": len(usable_strict),
                "no_experience": label_counts["0"],
                "experience": label_counts["2"],
                "subjects": len({row["Subject ID"] for row in usable_strict}),
                "nrem_strict": sum(stage_counts[str(code)] for code in (1, 2, 3, 4)),
                "rem_strict": (
                    len(usable_strict)
                    if dataset == "rem_Turku"
                    else stage_counts["5"]
                ),
                "actual_delimiter": actual_delimiter,
                "code_delimiter": config["code_delimiter"],
                "delimiter_matches": actual_delimiter == config["code_delimiter"],
                "expected_channels": "|".join(config["expected_channels"]),
                "channels_found": all(ch in channels for ch in config["expected_channels"]),
                "formatted_edfs_expected_by_code": dataset == "SiclariMA",
                "formatted_edfs_found": formatted_count,
                "missing_edfs": "|".join(missing),
                "extra_edfs": "|".join(extra),
                "sample_sfreq": float(raw.info["sfreq"]),
                "sample_channels": len(raw.ch_names),
            }
        )

    code_root = DATA / "dream-fig5-code"
    manifest = json.loads((DATA / "fig5-source-files.json").read_text(encoding="utf-8"))
    oudiette_root = DATA / "extracted" / "fig5-original" / "oudiette"
    oudiette_records, _ = read_records(oudiette_root / "Records.csv")
    oudiette_sleep = [
        row
        for row in oudiette_records
        if row["Last sleep stage"] in {"1", "2", "3", "4", "5"}
        and row["Experience"] in {"0", "2"}
    ]
    summary = {
        "audit_date": "2026-08-02",
        "article_dataset_count_in_prose": 6,
        "public_code_dataset_count": len(DATASETS),
        "public_code_datasets": list(DATASETS),
        "public_code_md5": {
            path.name: md5(path) for path in sorted(code_root.glob("*")) if path.is_file()
        },
        "source_articles_in_manifest": len(manifest["articles"]),
        "source_files_in_manifest": sum(len(x["files"]) for x in manifest["articles"]),
        "article_only_oudiette": {
            "records": len(oudiette_records),
            "strict_sleep_records": len(oudiette_sleep),
            "strict_sleep_no_experience": sum(
                row["Experience"] == "0" for row in oudiette_sleep
            ),
            "strict_sleep_experience": sum(
                row["Experience"] == "2" for row in oudiette_sleep
            ),
            "channels": ["Fp1-A2", "C3-A2", "O1-A2"],
            "interpretation": (
                "Named among six datasets used for EEG analyses in article prose, "
                "but not enumerated by the Figure 5 classifier code; it lacks the "
                "code's common F4/C4/O2 montage and has one strict sleeping "
                "No-experience observation."
            ),
        },
        "literal_execution_blockers": [
            "feat_ext_funcs is imported and used but absent from the code deposit",
            "DeGenearoMA code delimiter is semicolon while public Records.csv is comma-separated",
            "SiclariMA code requires formatted_*.edf with F4-M1/C4-M1/O2-M1; public versions 1-3 contain Chan 1..257 files",
        ],
        "scope_ambiguity": (
            "Article prose names six datasets for EEG analyses, whereas the Figure 5 "
            "classifier code enumerates five; Oudiette's omission is technically "
            "plausible but not explained in the deposited classifier workflow."
        ),
        "dataset_rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "fig5-source-audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = OUT / "fig5-source-audit.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    for row in rows:
        print(
            row["dataset"],
            f"strict={row['strict_usable']}",
            f"NREM={row['nrem_strict']}",
            f"REM={row['rem_strict']}",
            f"channels={row['channels_found']}",
            f"missing={row['missing_edfs'] or 'none'}",
        )


if __name__ == "__main__":
    main()
