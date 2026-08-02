#!/usr/bin/env python3
"""Normalize temporal-irreversibility features against phase surrogates.

Phase randomization preserves each channel's power spectrum and amplitude while
destroying nonlinear temporal ordering. The resulting z-scores therefore ask
whether an epoch's arrow-of-time statistic exceeds what its spectrum alone
would produce.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from extract_fig5_features import (
    DATA,
    DERIVED,
    bandpass,
    build_jobs,
    downsample,
    load_final_window,
    ordinal_pattern_ids,
)


DEFAULT_OUTPUT = DERIVED / "fig5-irreversibility-surrogates.csv"


def statistics(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ordinal-JS and signed cubic asymmetry for signals x time."""
    n_signals = signals.shape[0]
    ordinal = np.empty((n_signals, 4), dtype=float)
    cubic = np.empty((n_signals, 4), dtype=float)
    for row, signal in enumerate(signals):
        signal = (signal - signal.mean()) / max(signal.std(), np.finfo(float).tiny)
        for column, lag in enumerate((1, 2, 4, 8)):
            forward = np.bincount(ordinal_pattern_ids(signal, lag), minlength=27).astype(float)
            reverse = np.bincount(ordinal_pattern_ids(signal[::-1], lag), minlength=27).astype(float)
            forward = (forward + 0.5) / (forward.sum() + 13.5)
            reverse = (reverse + 0.5) / (reverse.sum() + 13.5)
            midpoint = 0.5 * (forward + reverse)
            ordinal[row, column] = 0.5 * np.sum(forward * np.log(forward / midpoint)) + 0.5 * np.sum(
                reverse * np.log(reverse / midpoint)
            )
            x0, x1 = signal[:-lag], signal[lag:]
            cubic[row, column] = np.mean(x0 * x0 * x1 - x0 * x1 * x1)
    return ordinal, cubic


def phase_surrogates(signal: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(signal)
    phase = rng.uniform(0.0, 2 * np.pi, size=(count, len(spectrum)))
    phase[:, 0] = 0.0
    if signal.size % 2 == 0:
        phase[:, -1] = 0.0
    randomized = np.abs(spectrum)[None, :] * np.exp(1j * (np.angle(spectrum)[None, :] + phase))
    randomized[:, 0] = spectrum[0]
    if signal.size % 2 == 0:
        randomized[:, -1] = spectrum[-1]
    return np.fft.irfft(randomized, n=signal.size, axis=1)


def cache_path(job: dict[str, object], count: int) -> Path:
    key = hashlib.sha1(
        f"v1|n={count}|{job['dataset']}|{job['filename']}".encode("utf-8")
    ).hexdigest()
    return DERIVED / "surrogate-cache" / f"{key}.json"


def extract(job: dict[str, object], count: int) -> dict[str, object]:
    cached = cache_path(job, count)
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    raw, sfreq = load_final_window(job)
    data = bandpass(downsample(raw, sfreq), 0.4, 35.0)
    seed_material = f"20260802|{job['dataset']}|{job['filename']}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    output: dict[str, object] = {
        "dataset": job["dataset"],
        "filename": job["filename"],
        "surrogate_count": count,
    }
    for channel, signal in enumerate(data, start=1):
        observed_ordinal, observed_cubic = statistics(signal[None, :])
        surrogates = phase_surrogates(signal, count, rng)
        null_ordinal, null_cubic = statistics(surrogates)
        for lag_index, lag in enumerate((1, 2, 4, 8)):
            for name, observed, null in (
                ("ordinal", observed_ordinal[0, lag_index], null_ordinal[:, lag_index]),
                ("cubic", observed_cubic[0, lag_index], null_cubic[:, lag_index]),
            ):
                sd = float(null.std(ddof=1))
                z = (float(observed) - float(null.mean())) / max(sd, np.finfo(float).tiny)
                if not math.isfinite(z):
                    z = float("nan")
                output[f"irrsurr_ch{channel}_lag{lag}_{name}_z"] = z
                output[f"irrexs_ch{channel}_lag{lag}_{name}"] = float(observed - null.mean())
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(output, allow_nan=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogates", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    jobs = build_jobs()
    if args.limit is not None:
        jobs = jobs[: args.limit]
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract, job, args.surrogates): job for job in jobs}
        for number, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                raise RuntimeError(f"Failed {job['dataset']}/{job['filename']}") from error
            if number % 25 == 0 or number == len(jobs):
                print(f"processed {number}/{len(jobs)}", flush=True)
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["filename"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
