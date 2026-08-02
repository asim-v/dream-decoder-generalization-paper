#!/usr/bin/env python3
"""Validate the release contracts for the targeted DREAM revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_targeted_revision as revision


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "dream-consciousness" / "targeted-revision"


def validate(out: Path) -> dict[str, object]:
    frame = revision.load_features(revision.DEFAULT_FEATURES, revision.DEFAULT_SURROGATES)
    predictions = pd.read_csv(out / "targeted-oof-predictions.csv", low_memory=False)
    baselines = pd.read_csv(out / "site-only-oof-predictions.csv", low_memory=False)
    folds = pd.read_csv(out / "targeted-fold-definitions.csv", low_memory=False)
    repeats = int(predictions["repetition"].nunique())
    revision.validate_prediction_contract(frame, predictions, baselines, folds, repeats)

    checks: dict[str, object] = {
        "records_total": int(len(frame)),
        "records_nrem": int((frame["stage"] == "NREM").sum()),
        "records_rem": int((frame["stage"] == "REM").sum()),
        "repetitions": repeats,
        "feature_families": int(predictions["feature_family"].nunique()),
    }

    # Every participant is assigned to exactly one test fold per repetition.
    fold_membership = folds.groupby(["stage", "repetition", "subject_id"])["fold"].nunique()
    if int(fold_membership.max()) != 1:
        raise AssertionError("A participant appears in multiple test folds")
    checks["participant_leakage"] = False

    # Stored folds exactly match a fresh deterministic construction.
    for stage in ("NREM", "REM"):
        metadata = revision.record_metadata(frame, stage)
        expected = revision.deterministic_splits(metadata, repeats)
        stored = revision.fold_assignment_matrix(folds, metadata, stage)
        for repetition, fold, _, test in expected:
            if not np.all(stored[repetition, test] == fold):
                raise AssertionError(f"Stored fold mismatch: {stage} repetition {repetition}")
    checks["deterministic_fold_rerun"] = True

    # Recompute every site-only score from training records only.
    for stage in ("NREM", "REM"):
        metadata = revision.record_metadata(frame, stage)
        y = metadata["label"].to_numpy(dtype=int)
        datasets = metadata["dataset"].astype(str).to_numpy()
        subjects = metadata["subject_id"].astype(str).to_numpy()
        stored_folds = revision.fold_assignment_matrix(folds, metadata, stage)
        baseline = baselines[baselines["stage"] == stage].set_index(
            ["repetition", "record_index"]
        )
        for repetition in range(repeats):
            for fold in range(5):
                test = stored_folds[repetition] == fold
                train = ~test
                training = pd.DataFrame(
                    {
                        "dataset": datasets[train],
                        "subject_id": subjects[train],
                        "label": y[train],
                    }
                )
                record_prevalence = training.groupby("dataset")["label"].mean()
                participant_prevalence = (
                    training.groupby(["dataset", "subject_id"])["label"]
                    .mean()
                    .groupby("dataset")
                    .mean()
                )
                for position in np.flatnonzero(test):
                    row = baseline.loc[(repetition, int(metadata.iloc[position]["record_index"]))]
                    dataset = datasets[position]
                    if not np.isclose(
                        row["record_prevalence_probability"], record_prevalence[dataset]
                    ):
                        raise AssertionError("Record prevalence used non-training information")
                    if not np.isclose(
                        row["participant_prevalence_probability"],
                        participant_prevalence[dataset],
                    ):
                        raise AssertionError(
                            "Participant prevalence used non-training information"
                        )
    checks["site_baseline_uses_test_labels"] = False

    decomposition = pd.read_csv(out / "site-pair-summary.csv")
    error = np.max(
        np.abs(
            decomposition["reconstructed_pooled_auroc"]
            - decomposition["direct_pooled_auroc"]
        )
    )
    if error > 1e-12:
        raise AssertionError("AUROC site-pair identity failed")
    checks["maximum_pair_decomposition_error"] = float(error)

    # One representative XGBoost fold must reproduce exactly in a fresh rerun.
    metadata = revision.record_metadata(frame, "NREM")
    split = revision.deterministic_splits(metadata, 1)[0]
    _, fold, train, test = split
    columns = revision.feature_columns(metadata, revision.FEATURE_FAMILIES["PSD (published)"])
    x = metadata[columns].to_numpy(dtype=float)
    y = metadata["label"].to_numpy(dtype=int)
    first = revision.train_predict(x[train], y[train], x[test], revision.SEED + fold)
    second = revision.train_predict(x[train], y[train], x[test], revision.SEED + fold)
    if not np.array_equal(first, second):
        raise AssertionError("Representative model rerun is not deterministic")
    checks["deterministic_model_rerun"] = True

    if checks["records_total"] != revision.EXPECTED_COUNTS["ALL"]:
        raise AssertionError("Unexpected total record count")
    if checks["records_nrem"] != revision.EXPECTED_COUNTS["NREM"]:
        raise AssertionError("Unexpected NREM record count")
    if checks["records_rem"] != revision.EXPECTED_COUNTS["REM"]:
        raise AssertionError("Unexpected REM record count")
    checks["status"] = "pass"
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = validate(args.out)
    path = args.out / "validation-report.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
