#!/usr/bin/env python3
"""Compare DREAM Figure 5 performance across a validation ladder.

All experience models use the deposited XGBoost hyperparameters and its
implicit default of ten boosting rounds. The only change between validation
regimes is which observations are allowed to cross the train/test boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "dream-consciousness" / "derived" / "fig5" / "fig5-features.csv"
SURROGATE_INPUT = (
    ROOT
    / "data"
    / "dream-consciousness"
    / "derived"
    / "fig5"
    / "fig5-irreversibility-surrogates.csv"
)
OUT = ROOT / "outputs" / "dream-consciousness"
SEED = 20260802
FEATURE_FAMILIES = {
    "PSD (published)": ("psd_",),
    "Catch22 broadband (published)": ("catch22_broadband_",),
    "Catch22 filtered (adapted)": ("catch22_filtered_",),
    "Riemannian covariance": ("riemann_",),
    "Log-Euclidean domain alignment": ("aligned_riemann_",),
    "Temporal irreversibility": ("irreversibility_",),
    "Surrogate-normalized irreversibility": ("irrsurr_",),
    "Geometry + irreversibility": ("riemann_", "irreversibility_"),
}


def feature_columns(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [column for column in frame.columns if column.startswith(prefixes)]


def train_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    num_class: int = 2,
) -> np.ndarray:
    if num_class == 2:
        params = {
            "objective": "multi:softprob",
            "eval_metric": "logloss",
            "num_class": 2,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": seed,
            "nthread": 1,
        }
    else:
        params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": num_class,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": seed,
            "nthread": 1,
        }
    model = xgb.train(params, xgb.DMatrix(x_train, label=y_train))
    return model.predict(xgb.DMatrix(x_test))


def binary_metrics(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(y, probability)),
        float(balanced_accuracy_score(y, probability >= 0.5)),
    )


def experience_splits(
    frame: pd.DataFrame,
    stage: str,
    family: str,
    prefixes: tuple[str, ...],
    repeats: int,
) -> list[dict[str, object]]:
    subset = frame.loc[frame["stage"] == stage].reset_index(drop=True)
    columns = feature_columns(subset, prefixes)
    x = subset[columns].to_numpy(dtype=float)
    y = subset["label"].to_numpy(dtype=int)
    groups = subset["subject_id"].to_numpy()
    datasets = subset["dataset"].to_numpy()
    results: list[dict[str, object]] = []

    for regime in ("random_record", "subject_held_out"):
        for repetition in range(repeats):
            split_seed = SEED + repetition
            if regime == "random_record":
                splitter = StratifiedKFold(5, shuffle=True, random_state=split_seed)
                splits = splitter.split(x, y)
            else:
                splitter = StratifiedGroupKFold(5, shuffle=True, random_state=split_seed)
                splits = splitter.split(x, y, groups)
            for fold, (train, test) in enumerate(splits):
                prediction = train_predict(x[train], y[train], x[test], split_seed + fold)
                auc, balanced = binary_metrics(y[test], prediction[:, 1])
                results.append(
                    {
                        "task": "experience",
                        "stage": stage,
                        "feature_family": family,
                        "regime": regime,
                        "repetition": repetition,
                        "fold": fold,
                        "held_out_dataset": "",
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_train_subjects": len(set(groups[train])),
                        "n_test_subjects": len(set(groups[test])),
                        "auroc": auc,
                        "balanced_accuracy": balanced,
                    }
                )

    for fold, held_out in enumerate(sorted(set(datasets))):
        test = np.flatnonzero(datasets == held_out)
        train = np.flatnonzero(datasets != held_out)
        if len(np.unique(y[test])) < 2 or len(np.unique(y[train])) < 2:
            continue
        prediction = train_predict(x[train], y[train], x[test], SEED + fold)
        auc, balanced = binary_metrics(y[test], prediction[:, 1])
        results.append(
            {
                "task": "experience",
                "stage": stage,
                "feature_family": family,
                "regime": "laboratory_held_out",
                "repetition": 0,
                "fold": fold,
                "held_out_dataset": held_out,
                "n_train": len(train),
                "n_test": len(test),
                "n_train_subjects": len(set(groups[train])),
                "n_test_subjects": len(set(groups[test])),
                "auroc": auc,
                "balanced_accuracy": balanced,
            }
        )
    return results


def fingerprint_splits(
    frame: pd.DataFrame,
    stage: str,
    family: str,
    prefixes: tuple[str, ...],
) -> list[dict[str, object]]:
    subset = frame.loc[frame["stage"] == stage].reset_index(drop=True)
    columns = feature_columns(subset, prefixes)
    x = subset[columns].to_numpy(dtype=float)
    names = sorted(subset["dataset"].unique())
    name_to_label = {name: index for index, name in enumerate(names)}
    y = subset["dataset"].map(name_to_label).to_numpy(dtype=int)
    groups = subset["subject_id"].to_numpy()
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    results: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
        prediction = train_predict(x[train], y[train], x[test], SEED + fold, len(names))
        auc = float(roc_auc_score(y[test], prediction, multi_class="ovr", average="macro"))
        accuracy = float(np.mean(np.argmax(prediction, axis=1) == y[test]))
        results.append(
            {
                "task": "dataset_identity",
                "stage": stage,
                "feature_family": family,
                "regime": "subject_held_out",
                "repetition": 0,
                "fold": fold,
                "held_out_dataset": "",
                "n_train": len(train),
                "n_test": len(test),
                "n_train_subjects": len(set(groups[train])),
                "n_test_subjects": len(set(groups[test])),
                "auroc": auc,
                "balanced_accuracy": accuracy,
            }
        )
    return results


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["stage"], row["feature_family"], row["regime"])].append(row)
    output: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        # Random/group CV is summarized per repetition, matching the published
        # mean-over-folds then mean-over-iterations convention. LODO is macro-lab.
        repetition_values: dict[int, list[float]] = defaultdict(list)
        repetition_balanced: dict[int, list[float]] = defaultdict(list)
        for row in values:
            repetition_values[int(row["repetition"])].append(float(row["auroc"]))
            repetition_balanced[int(row["repetition"])].append(float(row["balanced_accuracy"]))
        aucs = np.array([np.mean(x) for x in repetition_values.values()])
        balanced = np.array([np.mean(x) for x in repetition_balanced.values()])
        if key[3] == "laboratory_held_out":
            aucs = np.array([float(row["auroc"]) for row in values])
            balanced = np.array([float(row["balanced_accuracy"]) for row in values])
        output.append(
            {
                "task": key[0],
                "stage": key[1],
                "feature_family": key[2],
                "regime": key[3],
                "mean_auroc": float(aucs.mean()),
                "sd_auroc": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
                "p2.5_auroc": float(np.percentile(aucs, 2.5)),
                "p97.5_auroc": float(np.percentile(aucs, 97.5)),
                "mean_balanced_accuracy": float(balanced.mean()),
                "n_folds": len(values),
                "n_repetitions": len(repetition_values),
            }
        )
    return output


def plot(summary: pd.DataFrame, path: Path, families: list[str]) -> None:
    experience = summary.loc[summary["task"] == "experience"].copy()
    regimes = ["random_record", "subject_held_out", "laboratory_held_out"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(families)))
    width = 0.12
    for axis, stage in zip(axes, ("NREM", "REM")):
        stage_data = experience.loc[experience["stage"] == stage]
        xbase = np.arange(len(regimes))
        for index, (family, color) in enumerate(zip(families, colors)):
            values = []
            for regime in regimes:
                match = stage_data.loc[
                    (stage_data["feature_family"] == family)
                    & (stage_data["regime"] == regime),
                    "mean_auroc",
                ]
                values.append(float(match.iloc[0]))
            offset = (index - (len(families) - 1) / 2) * width
            axis.bar(xbase + offset, values, width, label=family, color=color)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_title(stage)
        axis.set_xticks(xbase, ["Random\nrecord", "New\nsubject", "New\nlaboratory"])
        axis.set_ylim(0.3, 1.0)
        axis.set_ylabel("AUROC")
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Dream-experience prediction across increasingly strict validation")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--surrogate-input", type=Path, default=SURROGATE_INPUT)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    if args.surrogate_input.exists():
        surrogate = pd.read_csv(args.surrogate_input, low_memory=False)
        frame = frame.merge(
            surrogate.drop(columns=["surrogate_count"]),
            on=["dataset", "filename"],
            how="left",
            validate="one_to_one",
        )
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    # Label-free, transductive batch alignment. Each lab/stage covariance is
    # translated to its log-Euclidean mean before any labels are inspected.
    riemann_columns = feature_columns(frame, ("riemann_",))
    aligned = frame.groupby(["dataset", "stage"])[riemann_columns].transform(
        lambda values: values - values.mean()
    )
    aligned.columns = [f"aligned_{column}" for column in aligned.columns]
    frame = pd.concat([frame, aligned], axis=1)
    rows: list[dict[str, object]] = []
    active_families = {
        family: prefixes
        for family, prefixes in FEATURE_FAMILIES.items()
        if feature_columns(frame, prefixes)
    }
    for stage in ("NREM", "REM"):
        for family, prefixes in active_families.items():
            print(f"experience: {stage} / {family}", flush=True)
            rows.extend(experience_splits(frame, stage, family, prefixes, args.repeats))
            print(f"dataset identity: {stage} / {family}", flush=True)
            rows.extend(fingerprint_splits(frame, stage, family, prefixes))
    summary = summarize(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    folds_path = OUT / "fig5-validation-folds.csv"
    summary_path = OUT / "fig5-validation-summary.csv"
    pd.DataFrame(rows).to_csv(folds_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(summary_path, index=False)
    json_path = OUT / "fig5-validation-summary.json"
    json_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "boosting_rounds": 10,
                "repetitions": args.repeats,
                "rows": len(frame),
                "feature_counts": {
                    family: len(feature_columns(frame, prefixes))
                    for family, prefixes in active_families.items()
                },
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot(summary_frame, OUT / "fig5-validation-ladder.png", list(active_families))
    print(f"Wrote {folds_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
