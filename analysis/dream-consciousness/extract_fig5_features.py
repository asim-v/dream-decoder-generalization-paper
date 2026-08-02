#!/usr/bin/env python3
"""Extract DREAM Figure 5 replication and exploratory nonlinear features.

The replication features follow the deposited workflow where it is complete:
the final 30 seconds, three common right-hemisphere channels, 0.4--35 Hz
multitaper relative PSD, 128 Hz broadband Catch22, and band-filtered Catch22.
The missing ``feat_ext_funcs`` module is replaced by direct pycatch22 calls.

For the public 256-channel Siclari files, the requested F4/C4/O2-M1 signals
are reconstructed as E224/E164/E150 minus E93. These are the nearest HydroCel
GSN-256 sensors to F4/C4/O2/M1 in MNE's standard montage. Eight s37 files lack
E93, so the nearest available M1 candidate (E92, then E103/E94/E102) is used.
This adaptation is explicitly tagged in every output row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
import pycatch22
from scipy.signal import butter, resample_poly, sosfiltfilt


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "dream-consciousness"
DERIVED = DATA / "derived" / "fig5"
DEFAULT_OUTPUT = DERIVED / "fig5-features.csv"
TARGET_SFREQ = 128.0
WINDOW_SECONDS = 30.0
STAGES = {1: "NREM", 2: "NREM", 3: "NREM", 4: "NREM", 5: "REM"}
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.1, 8.0),
    "alpha": (8.1, 11.0),
    "sigma": (11.1, 15.0),
    "beta": (15.1, 20.0),
    "gamma": (20.1, 35.0),
}
DATASETS = {
    "Zhang_Wamsley_2019": {
        "root": DATA / "extracted" / "zhang-wamsley" / "Zhang & Wamsley 2019",
        "channels": ["F4-REF", "C4-REF", "O2-REF", "A1-REF", "A2-REF"],
        "mode": "average_mastoids",
    },
    "DeGenearoMA": {
        "root": DATA / "extracted" / "fig5-original" / "degen_ma" / "Multiple awakenings",
        "channels": ["F4", "C4", "O2"],
        "mode": "direct",
    },
    "DeGenaro_YoungAdults": {
        "root": DATA / "extracted" / "young-adults" / "Dream_YoungAdults",
        "channels": ["F4", "C4", "O2"],
        "mode": "direct",
    },
    "SiclariMA": {
        "root": DATA / "extracted" / "fig5-original" / "siclari" / "Tononi Serial Awakenings",
        "channels": ["Chan 224", "Chan 164", "Chan 150", "Chan 93"],
        "mode": "hydrocel_to_m1",
    },
    "rem_Turku": {
        "root": DATA / "extracted" / "fig5-original" / "turku" / "REM_Turku",
        "channels": ["F4", "C4", "O2"],
        "mode": "direct",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def edf_name(dataset: str, value: str) -> str:
    name = Path(value.replace("\\", "/")).name
    if dataset == "rem_Turku" and not name.lower().endswith(".edf"):
        name += ".edf"
    return name


def build_jobs() -> list[dict[str, object]]:
    exclusions = {
        (row["dataset"], row["filename"]) for row in read_csv(DATA / "exclusions.csv")
    }
    jobs: list[dict[str, object]] = []
    for dataset, config in DATASETS.items():
        root = Path(config["root"])
        for record in read_csv(root / "Records.csv"):
            if record["Experience"] not in {"0", "2"}:
                continue
            filename = edf_name(dataset, record["Filename"])
            if (dataset, filename) in exclusions:
                continue
            path = root / "Data" / "PSG" / filename
            if not path.exists():
                continue
            stage_code = 5 if dataset == "rem_Turku" else int(record["Last sleep stage"])
            if stage_code not in STAGES:
                continue
            jobs.append(
                {
                    "dataset": dataset,
                    "path": path,
                    "filename": filename,
                    "case_id": record["Case ID"],
                    "subject_id": f"{dataset}:{record['Subject ID']}",
                    "label": 1 if record["Experience"] == "2" else 0,
                    "stage": STAGES[stage_code],
                    "stage_code": stage_code,
                    "channels": config["channels"],
                    "mode": config["mode"],
                }
            )
    return jobs


def cache_path(job: dict[str, object], include_filtered: bool) -> Path:
    key = hashlib.sha1(
        f"v2|filtered={include_filtered}|{job['dataset']}|{job['filename']}".encode("utf-8")
    ).hexdigest()
    return DERIVED / "record-cache" / f"{key}.json"


def load_final_window(job: dict[str, object]) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(job["path"], preload=False, verbose="ERROR")
    if job["mode"] == "hydrocel_to_m1":
        sensor_to_index: dict[int, int] = {}
        for index, name in enumerate(raw.ch_names):
            normalized = name.lower().replace("chan", "").replace("e", "").strip()
            if normalized.isdigit():
                sensor_to_index[int(normalized)] = index
        reference = next(
            (sensor for sensor in (93, 92, 103, 94, 102) if sensor in sensor_to_index),
            None,
        )
        if reference is None:
            raise RuntimeError("No usable left-mastoid HydroCel sensor")
        picks = [sensor_to_index[sensor] for sensor in (224, 164, 150, reference)]
    else:
        picks = [raw.ch_names.index(name) for name in job["channels"]]
    sfreq = float(raw.info["sfreq"])
    sample_count = round(WINDOW_SECONDS * sfreq) + 1
    start = max(0, raw.n_times - sample_count)
    data = raw.get_data(picks=picks, start=start, stop=raw.n_times)
    mode = job["mode"]
    if mode == "average_mastoids":
        data = data[:3] - data[3:5].mean(axis=0, keepdims=True)
    elif mode == "hydrocel_to_m1":
        data = data[:3] - data[3]
    else:
        data = data[:3]
    data -= np.median(data, axis=1, keepdims=True)
    return data, sfreq


def downsample(data: np.ndarray, sfreq: float) -> np.ndarray:
    ratio = Fraction(TARGET_SFREQ / sfreq).limit_denominator(2000)
    return resample_poly(data, ratio.numerator, ratio.denominator, axis=1)


def bandpass(data: np.ndarray, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=TARGET_SFREQ, output="sos")
    return sosfiltfilt(sos, data, axis=1)


def finite(value: object) -> float:
    result = float(value)
    return result if math.isfinite(result) else float("nan")


def catch22_features(data: np.ndarray, prefix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for channel_index, signal in enumerate(data):
        output = pycatch22.catch22_all(signal.tolist(), catch24=False)
        for name, value in zip(output["names"], output["values"]):
            result[f"{prefix}_ch{channel_index + 1}_{name}"] = finite(value)
    return result


def psd_features(data: np.ndarray, sfreq: float) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        psd, freqs = mne.time_frequency.psd_array_multitaper(
            data,
            sfreq=sfreq,
            fmin=0.4,
            fmax=35.0,
            adaptive=False,
            normalization="length",
            verbose="ERROR",
        )
    psd = psd / np.maximum(psd.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    result: dict[str, float] = {}
    for band_name, (low, high) in BANDS.items():
        lo, hi = np.searchsorted(freqs, [low, high])
        values = psd[:, lo:hi].sum(axis=1)
        for channel_index, value in enumerate(values):
            result[f"psd_ch{channel_index + 1}_{band_name}"] = finite(value)
    return result


def spd_log_features(data: np.ndarray, prefix: str) -> dict[str, float]:
    covariance = np.cov(data)
    covariance /= max(float(np.trace(covariance)), np.finfo(float).tiny)
    covariance = 0.99 * covariance + 0.01 * np.eye(3) / 3
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    log_covariance = (eigenvectors * np.log(np.maximum(eigenvalues, 1e-12))) @ eigenvectors.T
    result: dict[str, float] = {}
    for row in range(3):
        for column in range(row, 3):
            scale = math.sqrt(2) if row != column else 1.0
            result[f"{prefix}_r{row + 1}c{column + 1}"] = finite(
                scale * log_covariance[row, column]
            )
    return result


def ordinal_pattern_ids(signal: np.ndarray, lag: int) -> np.ndarray:
    triples = np.column_stack((signal[: -2 * lag], signal[lag:-lag], signal[2 * lag :]))
    order = np.argsort(triples, axis=1, kind="stable")
    # Lehmer-like base-3 code; only six values occur and bincount is stable.
    return order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]


def irreversibility_features(data: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for channel_index, signal in enumerate(data):
        signal = (signal - signal.mean()) / max(signal.std(), np.finfo(float).tiny)
        for lag in (1, 2, 4, 8):
            forward = np.bincount(ordinal_pattern_ids(signal, lag), minlength=27).astype(float)
            reverse = np.bincount(ordinal_pattern_ids(signal[::-1], lag), minlength=27).astype(float)
            forward = (forward + 0.5) / (forward.sum() + 13.5)
            reverse = (reverse + 0.5) / (reverse.sum() + 13.5)
            midpoint = 0.5 * (forward + reverse)
            js = 0.5 * np.sum(forward * np.log(forward / midpoint)) + 0.5 * np.sum(
                reverse * np.log(reverse / midpoint)
            )
            x0, x1 = signal[:-lag], signal[lag:]
            cubic = np.mean(x0 * x0 * x1 - x0 * x1 * x1)
            result[f"irreversibility_ch{channel_index + 1}_lag{lag}_ordinal_js"] = finite(js)
            result[f"irreversibility_ch{channel_index + 1}_lag{lag}_cubic"] = finite(cubic)
    return result


def extract(job: dict[str, object], include_filtered: bool) -> dict[str, object]:
    cached = cache_path(job, include_filtered)
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    data, sfreq = load_final_window(job)
    sampled = downsample(data, sfreq)
    broadband = bandpass(sampled, 0.4, 35.0)
    features: dict[str, object] = {
        "dataset": job["dataset"],
        "filename": job["filename"],
        "case_id": job["case_id"],
        "subject_id": job["subject_id"],
        "label": job["label"],
        "stage": job["stage"],
        "stage_code": job["stage_code"],
        "sfreq_original": sfreq,
        "samples_original": data.shape[1],
        "channel_adaptation": job["mode"],
    }
    features.update(psd_features(data, sfreq))
    features.update(catch22_features(sampled, "catch22_broadband"))
    features.update(spd_log_features(broadband, "riemann_broadband"))
    features.update(irreversibility_features(broadband))
    for band_name, (low, high) in BANDS.items():
        filtered = bandpass(sampled, low, high)
        features.update(spd_log_features(filtered, f"riemann_{band_name}"))
        if include_filtered:
            features.update(catch22_features(filtered, f"catch22_filtered_{band_name}"))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(features, allow_nan=True) + "\n", encoding="utf-8")
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--skip-filtered-catch22",
        action="store_true",
        help="omit the expensive filtered Catch22 replication family",
    )
    args = parser.parse_args()
    jobs = build_jobs()
    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(f"Extracting {len(jobs)} strict NREM/REM records", flush=True)
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(extract, job, not args.skip_filtered_catch22): job
            for job in jobs
        }
        for number, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                raise RuntimeError(f"Failed {job['dataset']}/{job['filename']}") from error
            if number % 25 == 0 or number == len(jobs):
                print(f"processed {number}/{len(jobs)}", flush=True)
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["filename"])))
    fieldnames = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows x {len(fieldnames)} columns to {args.output}")


if __name__ == "__main__":
    main()
