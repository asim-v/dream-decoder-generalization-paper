#!/usr/bin/env python3
"""Add subject-equal estimates and cluster-bootstrap CIs to lab-held-out tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from run_fig5_validation import (
    FEATURE_FAMILIES,
    INPUT,
    OUT,
    SEED,
    SURROGATE_INPUT,
    feature_columns,
    train_predict,
)


def subject_weights(groups: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    del unique
    return 1.0 / counts[inverse]


def cluster_bootstrap(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    estimates: list[float] = []
    attempts = 0
    while len(estimates) < repetitions and attempts < repetitions * 10:
        attempts += 1
        sampled = rng.choice(unique, size=len(unique), replace=True)
        chosen = np.concatenate([indices[group] for group in sampled])
        sampled_y = y[chosen]
        if len(np.unique(sampled_y)) < 2:
            continue
        weights = np.concatenate(
            [np.full(len(indices[group]), 1.0 / len(indices[group])) for group in sampled]
        )
        estimates.append(roc_auc_score(sampled_y, scores[chosen], sample_weight=weights))
    if not estimates:
        return float("nan"), float("nan"), 0
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high), len(estimates)


def load_frame(input_path: Path, surrogate_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path, low_memory=False)
    if surrogate_path.exists():
        surrogate = pd.read_csv(surrogate_path, low_memory=False)
        frame = frame.merge(
            surrogate.drop(columns=["surrogate_count"]),
            on=["dataset", "filename"],
            how="left",
            validate="one_to_one",
        )
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    riemann = feature_columns(frame, ("riemann_",))
    aligned = frame.groupby(["dataset", "stage"])[riemann].transform(
        lambda values: values - values.mean()
    )
    aligned.columns = [f"aligned_{column}" for column in riemann]
    return pd.concat([frame, aligned], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--surrogate-input", type=Path, default=SURROGATE_INPUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    frame = load_frame(args.input, args.surrogate_input)
    rows: list[dict[str, object]] = []
    for stage in ("NREM", "REM"):
        stage_frame = frame.loc[frame["stage"] == stage].reset_index(drop=True)
        for family, prefixes in FEATURE_FAMILIES.items():
            columns = feature_columns(stage_frame, prefixes)
            if not columns:
                continue
            x = stage_frame[columns].to_numpy(dtype=float)
            y = stage_frame["label"].to_numpy(dtype=int)
            groups = stage_frame["subject_id"].to_numpy()
            datasets = stage_frame["dataset"].to_numpy()
            print(f"{stage} / {family}", flush=True)
            family_rows: list[dict[str, object]] = []
            for fold, held_out in enumerate(sorted(set(datasets))):
                test = np.flatnonzero(datasets == held_out)
                train = np.flatnonzero(datasets != held_out)
                prediction = train_predict(x[train], y[train], x[test], SEED + fold)[:, 1]
                weights = subject_weights(groups[test])
                auc = float(roc_auc_score(y[test], prediction, sample_weight=weights))
                low, high, valid = cluster_bootstrap(
                    y[test], prediction, groups[test], args.bootstrap, SEED + fold
                )
                result = {
                    "stage": stage,
                    "feature_family": family,
                    "held_out_dataset": held_out,
                    "n_records": len(test),
                    "n_subjects": len(set(groups[test])),
                    "no_experience": int((y[test] == 0).sum()),
                    "experience": int((y[test] == 1).sum()),
                    "subject_equal_auroc": auc,
                    "cluster_ci_2.5": low,
                    "cluster_ci_97.5": high,
                    "valid_bootstraps": valid,
                    "balanced_accuracy_at_0.5": float(
                        balanced_accuracy_score(
                            y[test], prediction >= 0.5, sample_weight=weights
                        )
                    ),
                }
                family_rows.append(result)
                rows.append(result)
            rows.append(
                {
                    "stage": stage,
                    "feature_family": family,
                    "held_out_dataset": "MACRO_MEAN",
                    "n_records": sum(int(row["n_records"]) for row in family_rows),
                    "n_subjects": sum(int(row["n_subjects"]) for row in family_rows),
                    "no_experience": sum(int(row["no_experience"]) for row in family_rows),
                    "experience": sum(int(row["experience"]) for row in family_rows),
                    "subject_equal_auroc": float(
                        np.mean([float(row["subject_equal_auroc"]) for row in family_rows])
                    ),
                    "cluster_ci_2.5": "",
                    "cluster_ci_97.5": "",
                    "valid_bootstraps": "",
                    "balanced_accuracy_at_0.5": float(
                        np.mean(
                            [float(row["balanced_accuracy_at_0.5"]) for row in family_rows]
                        )
                    ),
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig5-lodo-subject-inference.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
