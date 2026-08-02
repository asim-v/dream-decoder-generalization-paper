#!/usr/bin/env python3
"""Run the prespecified DREAM site-conditioned validation diagnostics.

The script reuses the frozen feature matrix, regenerates deterministic
participant-held-out out-of-fold predictions, and writes machine-readable
diagnostics plus publication figures. It does not read raw EEG.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = (
    ROOT / "data" / "dream-consciousness" / "derived" / "fig5" / "fig5-features.csv"
)
DEFAULT_SURROGATES = (
    ROOT
    / "data"
    / "dream-consciousness"
    / "derived"
    / "fig5"
    / "fig5-irreversibility-surrogates.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "dream-consciousness" / "targeted-revision"
DEFAULT_LODO = ROOT / "outputs" / "dream-consciousness" / "fig5-lodo-subject-inference.csv"
DEFAULT_FOLDS = ROOT / "outputs" / "dream-consciousness" / "fig5-validation-folds.csv"

SEED = 20260802
EXPECTED_COUNTS = {"ALL": 1065, "NREM": 730, "REM": 335}
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "PSD (published)": ("psd_",),
    "Catch22 broadband (published)": ("catch22_broadband_",),
    "Catch22 filtered (adapted)": ("catch22_filtered_",),
    "Riemannian covariance": ("riemann_",),
    "Log-Euclidean domain alignment": ("aligned_riemann_",),
    "Temporal irreversibility": ("irreversibility_",),
    "Surrogate-normalized irreversibility": ("irrsurr_",),
}
PRIMARY = {
    "NREM": "PSD (published)",
    "REM": "Catch22 filtered (adapted)",
}
DISPLAY = {
    "DeGenaro_YoungAdults": "Young Adults",
    "DeGenearoMA": "Multiple Awakenings",
    "SiclariMA": "Siclari",
    "Zhang_Wamsley_2019": "Zhang",
    "rem_Turku": "Turku",
}


def feature_columns(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [column for column in frame.columns if column.startswith(prefixes)]


def load_features(features: Path, surrogates: Path) -> pd.DataFrame:
    frame = pd.read_csv(features, low_memory=False)
    if surrogates.exists():
        surrogate = pd.read_csv(surrogates, low_memory=False)
        frame = frame.merge(
            surrogate.drop(columns=["surrogate_count"], errors="ignore"),
            on=["dataset", "filename"],
            how="left",
            validate="one_to_one",
        )
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    riemann = feature_columns(frame, ("riemann_",))
    aligned = frame.groupby(["dataset", "stage"], sort=False)[riemann].transform(
        lambda values: values - values.mean()
    )
    aligned.columns = [f"aligned_{column}" for column in aligned.columns]
    frame = pd.concat([frame, aligned], axis=1)
    frame.insert(0, "record_index", np.arange(len(frame), dtype=int))
    if len(frame) != EXPECTED_COUNTS["ALL"]:
        raise ValueError(f"Expected 1,065 records, found {len(frame):,}")
    for stage in ("NREM", "REM"):
        observed = int((frame["stage"] == stage).sum())
        if observed != EXPECTED_COUNTS[stage]:
            raise ValueError(f"Expected {EXPECTED_COUNTS[stage]} {stage} records, found {observed}")
    expected_subject = frame["dataset"].astype(str) + ":"
    if not np.all(
        [str(subject).startswith(prefix) for subject, prefix in zip(frame["subject_id"], expected_subject)]
    ):
        raise ValueError("Participant identifiers are not consistently dataset-scoped")
    if frame.duplicated(["dataset", "filename"]).any():
        raise ValueError("dataset + filename must uniquely identify a record")
    return frame


def train_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
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
    model = xgb.train(params, xgb.DMatrix(x_train, label=y_train))
    return model.predict(xgb.DMatrix(x_test))[:, 1]


def deterministic_splits(
    subset: pd.DataFrame, repeats: int
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    y = subset["label"].to_numpy(dtype=int)
    groups = subset["subject_id"].to_numpy()
    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for repetition in range(repeats):
        splitter = StratifiedGroupKFold(
            5, shuffle=True, random_state=SEED + repetition
        )
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            if set(groups[train]) & set(groups[test]):
                raise AssertionError("Participant leakage detected")
            splits.append((repetition, fold, train, test))
    return splits


def site_prevalence_predictions(
    subset: pd.DataFrame,
    splits: list[tuple[int, int, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    y = subset["label"].to_numpy(dtype=int)
    datasets = subset["dataset"].astype(str).to_numpy()
    subjects = subset["subject_id"].astype(str).to_numpy()
    rows: list[pd.DataFrame] = []
    for repetition, fold, train, test in splits:
        train_frame = pd.DataFrame(
            {"dataset": datasets[train], "subject_id": subjects[train], "label": y[train]}
        )
        record_prevalence = train_frame.groupby("dataset")["label"].mean().to_dict()
        participant_means = (
            train_frame.groupby(["dataset", "subject_id"], as_index=False)["label"].mean()
        )
        participant_prevalence = (
            participant_means.groupby("dataset")["label"].mean().to_dict()
        )
        global_record = float(train_frame["label"].mean())
        global_participant = float(participant_means["label"].mean())
        test_sites = datasets[test]
        out = subset.iloc[test][
            ["record_index", "dataset", "filename", "subject_id", "label", "stage"]
        ].copy()
        out["repetition"] = repetition
        out["fold"] = fold
        out["record_prevalence_probability"] = [
            float(record_prevalence.get(site, global_record)) for site in test_sites
        ]
        out["participant_prevalence_probability"] = [
            float(participant_prevalence.get(site, global_participant)) for site in test_sites
        ]
        out["record_global_fallback"] = [site not in record_prevalence for site in test_sites]
        out["participant_global_fallback"] = [
            site not in participant_prevalence for site in test_sites
        ]
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def generate_predictions(
    frame: pd.DataFrame,
    out: Path,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    baselines: list[pd.DataFrame] = []
    fold_rows: list[pd.DataFrame] = []
    for stage in ("NREM", "REM"):
        subset = frame.loc[frame["stage"] == stage].copy().reset_index(drop=True)
        splits = deterministic_splits(subset, repeats)
        baselines.append(site_prevalence_predictions(subset, splits))
        for repetition, fold, train, test in splits:
            fold_frame = subset.iloc[test][
                ["record_index", "dataset", "filename", "subject_id", "label", "stage"]
            ].copy()
            fold_frame["repetition"] = repetition
            fold_frame["fold"] = fold
            fold_frame["n_train"] = len(train)
            fold_frame["n_test"] = len(test)
            fold_rows.append(fold_frame)
        for family, prefixes in FEATURE_FAMILIES.items():
            columns = feature_columns(subset, prefixes)
            if not columns:
                continue
            print(f"OOF {stage}: {family} ({len(columns)} features)", flush=True)
            x = subset[columns].to_numpy(dtype=float)
            y = subset["label"].to_numpy(dtype=int)
            for repetition, fold, train, test in splits:
                probability = train_predict(
                    x[train], y[train], x[test], SEED + repetition + fold
                )
                result = subset.iloc[test][
                    ["record_index", "dataset", "filename", "subject_id", "label", "stage"]
                ].copy()
                result["feature_family"] = family
                result["repetition"] = repetition
                result["fold"] = fold
                result["probability"] = probability
                predictions.append(result)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    baseline_frame = pd.concat(baselines, ignore_index=True)
    folds = pd.concat(fold_rows, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(out / "targeted-oof-predictions.csv", index=False)
    baseline_frame.to_csv(out / "site-only-oof-predictions.csv", index=False)
    folds.to_csv(out / "targeted-fold-definitions.csv", index=False)
    return prediction_frame, baseline_frame, folds


def validate_prediction_contract(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    baselines: pd.DataFrame,
    folds: pd.DataFrame,
    repeats: int,
) -> None:
    active_families = predictions["feature_family"].nunique()
    expected_predictions = len(frame) * repeats * active_families
    if len(predictions) != expected_predictions:
        raise AssertionError(
            f"Expected {expected_predictions:,} prediction rows, found {len(predictions):,}"
        )
    expected_baselines = len(frame) * repeats
    if len(baselines) != expected_baselines or len(folds) != expected_baselines:
        raise AssertionError("Incomplete baseline predictions or fold definitions")
    keys = ["record_index", "feature_family", "repetition"]
    if predictions.duplicated(keys).any():
        raise AssertionError("Duplicate model out-of-fold predictions")
    if baselines.duplicated(["record_index", "repetition"]).any():
        raise AssertionError("Duplicate baseline out-of-fold predictions")
    if predictions["probability"].isna().any():
        raise AssertionError("Missing model probabilities")
    if baselines[
        ["record_prevalence_probability", "participant_prevalence_probability"]
    ].isna().any().any():
        raise AssertionError("Missing site-only baseline probabilities")
    if baselines[["record_global_fallback", "participant_global_fallback"]].any().any():
        raise AssertionError("Unexpected global-prior fallback")
    for _, fold in folds.groupby(["stage", "repetition", "fold"], sort=False):
        if fold["subject_id"].duplicated().any():
            # Repeated records from a test participant are valid. The actual
            # leakage check is performed when constructing every split.
            pass


def record_metadata(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    return (
        frame.loc[frame["stage"] == stage]
        .sort_values("record_index")
        .reset_index(drop=True)
    )


def score_matrix(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    stage: str,
    family: str,
    score_column: str = "probability",
) -> np.ndarray:
    subset = predictions.loc[
        (predictions["stage"] == stage)
        & ((predictions["feature_family"] == family) if "feature_family" in predictions else True)
    ]
    matrix = subset.pivot(index="repetition", columns="record_index", values=score_column)
    matrix = matrix.reindex(columns=metadata["record_index"].to_numpy())
    if matrix.isna().any().any():
        raise AssertionError(f"Incomplete score matrix for {stage} / {family}")
    return matrix.to_numpy(dtype=float)


def baseline_score_matrix(
    baselines: pd.DataFrame,
    metadata: pd.DataFrame,
    stage: str,
    score_column: str,
) -> np.ndarray:
    subset = baselines.loc[baselines["stage"] == stage]
    matrix = subset.pivot(index="repetition", columns="record_index", values=score_column)
    matrix = matrix.reindex(columns=metadata["record_index"].to_numpy())
    if matrix.isna().any().any():
        raise AssertionError(f"Incomplete baseline score matrix for {stage}")
    return matrix.to_numpy(dtype=float)


def fold_assignment_matrix(
    folds: pd.DataFrame, metadata: pd.DataFrame, stage: str
) -> np.ndarray:
    subset = folds.loc[folds["stage"] == stage]
    matrix = subset.pivot(index="repetition", columns="record_index", values="fold")
    matrix = matrix.reindex(columns=metadata["record_index"].to_numpy())
    if matrix.isna().any().any():
        raise AssertionError(f"Incomplete fold matrix for {stage}")
    return matrix.to_numpy(dtype=int)


def auc_components_many(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return concordant weight and positive-negative pair weight per row.

    ``weights`` has shape (bootstrap samples, observations). Scores and labels
    are one-dimensional. Ties receive half credit.
    """
    if weights.ndim == 1:
        weights = weights[None, :]
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]
    sorted_weights = weights[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    positive = np.add.reduceat(sorted_weights * (sorted_y == 1), starts, axis=1)
    negative = np.add.reduceat(sorted_weights * (sorted_y == 0), starts, axis=1)
    negative_before = np.cumsum(negative, axis=1) - negative
    numerator = np.sum(positive * (negative_before + 0.5 * negative), axis=1)
    denominator = np.sum(positive, axis=1) * np.sum(negative, axis=1)
    return numerator, denominator


def auc_many(y: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    numerator, denominator = auc_components_many(y, scores, weights)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def finite_mean(values: Iterable[np.ndarray]) -> np.ndarray:
    stack = np.vstack(list(values))
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)
    return np.divide(
        np.nansum(stack, axis=0),
        counts,
        out=np.full(stack.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )


def participant_equal_base(subjects: np.ndarray) -> np.ndarray:
    values, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    del values
    return 1.0 / counts[inverse]


def cluster_bootstrap_weights(
    metadata: pd.DataFrame, bootstraps: int, seed: int
) -> np.ndarray:
    datasets = metadata["dataset"].astype(str).to_numpy()
    subjects = metadata["subject_id"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    weights = np.zeros((bootstraps, len(metadata)), dtype=np.float32)
    for dataset in sorted(np.unique(datasets)):
        record_index = np.flatnonzero(datasets == dataset)
        site_subjects = np.unique(subjects[record_index])
        draws = rng.integers(0, len(site_subjects), size=(bootstraps, len(site_subjects)))
        subject_counts = np.zeros((bootstraps, len(site_subjects)), dtype=np.float32)
        for row in range(bootstraps):
            subject_counts[row] = np.bincount(draws[row], minlength=len(site_subjects))
        subject_to_position = {subject: i for i, subject in enumerate(site_subjects)}
        positions = np.array([subject_to_position[subject] for subject in subjects[record_index]])
        weights[:, record_index] = subject_counts[:, positions]
    return weights


def evaluate_scores(
    scores: np.ndarray,
    metadata: pd.DataFrame,
    cluster_weights: np.ndarray,
    base_weights: np.ndarray,
    folds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    y = metadata["label"].to_numpy(dtype=int)
    datasets = metadata["dataset"].astype(str).to_numpy()
    weights = cluster_weights * base_weights[None, :]
    pooled: list[np.ndarray] = []
    macro: list[np.ndarray] = []
    within_num: list[np.ndarray] = []
    within_den: list[np.ndarray] = []
    fold_macro: list[np.ndarray] = []
    fold_within_num: list[np.ndarray] = []
    fold_within_den: list[np.ndarray] = []
    site_values: dict[str, list[np.ndarray]] = {
        dataset: [] for dataset in sorted(np.unique(datasets))
    }
    for repetition_index, repetition_scores in enumerate(scores):
        pooled.append(auc_many(y, repetition_scores, weights))
        rep_sites: list[np.ndarray] = []
        rep_num = np.zeros(len(weights), dtype=float)
        rep_den = np.zeros(len(weights), dtype=float)
        for dataset in sorted(site_values):
            mask = datasets == dataset
            num, den = auc_components_many(y[mask], repetition_scores[mask], weights[:, mask])
            values = np.divide(
                num,
                den,
                out=np.full_like(num, np.nan, dtype=float),
                where=den > 0,
            )
            site_values[dataset].append(values)
            rep_sites.append(values)
            rep_num += num
            rep_den += den
        macro.append(finite_mean(rep_sites))
        within_num.append(rep_num)
        within_den.append(rep_den)
        if folds is not None:
            rep_fold_sites: list[np.ndarray] = []
            rep_fold_num = np.zeros(len(weights), dtype=float)
            rep_fold_den = np.zeros(len(weights), dtype=float)
            for dataset in sorted(site_values):
                site_num = np.zeros(len(weights), dtype=float)
                site_den = np.zeros(len(weights), dtype=float)
                for fold in sorted(np.unique(folds[repetition_index])):
                    mask = (datasets == dataset) & (folds[repetition_index] == fold)
                    if not mask.any():
                        continue
                    num, den = auc_components_many(
                        y[mask], repetition_scores[mask], weights[:, mask]
                    )
                    site_num += num
                    site_den += den
                site_auc = np.divide(
                    site_num,
                    site_den,
                    out=np.full_like(site_num, np.nan, dtype=float),
                    where=site_den > 0,
                )
                rep_fold_sites.append(site_auc)
                rep_fold_num += site_num
                rep_fold_den += site_den
            fold_macro.append(finite_mean(rep_fold_sites))
            fold_within_num.append(rep_fold_num)
            fold_within_den.append(rep_fold_den)
    total_num = np.sum(np.vstack(within_num), axis=0)
    total_den = np.sum(np.vstack(within_den), axis=0)
    result = {
        "pooled": finite_mean(pooled),
        "site_macro": finite_mean(macro),
        "within_pair_weighted": np.divide(
            total_num,
            total_den,
            out=np.full_like(total_num, np.nan, dtype=float),
            where=total_den > 0,
        ),
    }
    if folds is not None:
        fold_num = np.sum(np.vstack(fold_within_num), axis=0)
        fold_den = np.sum(np.vstack(fold_within_den), axis=0)
        result["site_macro_fold_conditioned"] = finite_mean(fold_macro)
        result["within_pair_weighted_fold_conditioned"] = np.divide(
            fold_num,
            fold_den,
            out=np.full_like(fold_num, np.nan, dtype=float),
            where=fold_den > 0,
        )
    for dataset, values in site_values.items():
        result[f"site::{dataset}"] = finite_mean(values)
    return result


def interval(values: np.ndarray) -> tuple[float, float, int]:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return np.nan, np.nan, 0
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high), int(len(valid))


def composition_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (stage, dataset), group in frame.groupby(["stage", "dataset"], sort=True):
        participant_counts = group.groupby("subject_id").size()
        experience = int(group["label"].sum())
        no_experience = int(len(group) - experience)
        minority = min(experience, no_experience)
        if minority == 0:
            flag = "single_class; within_site_auroc_undefined"
        elif minority < 5:
            flag = "minority_class_lt_5"
        else:
            flag = ""
        rows.append(
            {
                "stage": stage,
                "dataset": dataset,
                "display_name": DISPLAY.get(dataset, dataset),
                "participants": int(group["subject_id"].nunique()),
                "records": int(len(group)),
                "experience": experience,
                "no_experience": no_experience,
                "experience_prevalence": experience / len(group),
                "records_per_participant_median": float(participant_counts.median()),
                "records_per_participant_min": int(participant_counts.min()),
                "records_per_participant_max": int(participant_counts.max()),
                "within_site_flag": flag,
            }
        )
    return pd.DataFrame(rows)


def count_context(metadata: pd.DataFrame) -> dict[str, int]:
    positive = int(metadata["label"].sum())
    negative = int(len(metadata) - positive)
    pairs = 0
    for _, group in metadata.groupby("dataset"):
        pos = int(group["label"].sum())
        pairs += pos * (len(group) - pos)
    return {
        "n_labs": int(sum(group["label"].nunique() == 2 for _, group in metadata.groupby("dataset"))),
        "n_participants": int(metadata["subject_id"].nunique()),
        "n_positive": positive,
        "n_negative": negative,
        "n_within_pairs": int(pairs),
    }


def append_estimates(
    rows: list[dict[str, object]],
    analysis: str,
    stage: str,
    family: str,
    weighting: str,
    point: dict[str, np.ndarray],
    bootstrap: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    estimands: Iterable[str] = (
        "pooled",
        "site_macro",
        "within_pair_weighted",
        "site_macro_fold_conditioned",
        "within_pair_weighted_fold_conditioned",
    ),
) -> None:
    context = count_context(metadata)
    for estimand in estimands:
        if estimand not in point:
            continue
        low, high, valid = interval(bootstrap[estimand])
        rows.append(
            {
                "analysis": analysis,
                "stage": stage,
                "feature_family": family,
                "weighting": weighting,
                "estimand": estimand,
                "estimate": float(point[estimand][0]),
                "ci_low": low,
                "ci_high": high,
                "valid_bootstraps": valid,
                **context,
            }
        )


def site_specific_rows(
    stage: str,
    family: str,
    weighting: str,
    point: dict[str, np.ndarray],
    bootstrap: dict[str, np.ndarray],
    metadata: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, group in metadata.groupby("dataset", sort=True):
        key = f"site::{dataset}"
        if key not in point:
            continue
        low, high, valid = interval(bootstrap[key])
        positive = int(group["label"].sum())
        negative = int(len(group) - positive)
        rows.append(
            {
                "stage": stage,
                "feature_family": family,
                "weighting": weighting,
                "dataset": dataset,
                "display_name": DISPLAY.get(dataset, dataset),
                "estimate": float(point[key][0]),
                "ci_low": low,
                "ci_high": high,
                "valid_bootstraps": valid,
                "participants": int(group["subject_id"].nunique()),
                "experience": positive,
                "no_experience": negative,
                "pairs": positive * negative,
            }
        )
    return rows


def site_pair_decomposition(
    scores: np.ndarray,
    metadata: pd.DataFrame,
    stage: str,
    family: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    y = metadata["label"].to_numpy(dtype=int)
    datasets = metadata["dataset"].astype(str).to_numpy()
    sites = sorted(np.unique(datasets))
    rows: list[dict[str, object]] = []
    total_pairs = int(np.sum(y == 1) * np.sum(y == 0))
    for positive_site in sites:
        pos = (datasets == positive_site) & (y == 1)
        for negative_site in sites:
            neg = (datasets == negative_site) & (y == 0)
            pair_count = int(pos.sum() * neg.sum())
            values: list[float] = []
            if pair_count:
                pair_y = np.r_[np.ones(pos.sum(), dtype=int), np.zeros(neg.sum(), dtype=int)]
                weights = np.ones((1, len(pair_y)))
                for repetition_scores in scores:
                    pair_scores = np.r_[repetition_scores[pos], repetition_scores[neg]]
                    values.append(float(auc_many(pair_y, pair_scores, weights)[0]))
            concordance = float(np.mean(values)) if values else np.nan
            rows.append(
                {
                    "stage": stage,
                    "feature_family": family,
                    "positive_site": positive_site,
                    "negative_site": negative_site,
                    "positive_site_display": DISPLAY.get(positive_site, positive_site),
                    "negative_site_display": DISPLAY.get(negative_site, negative_site),
                    "positive_records": int(pos.sum()),
                    "negative_records": int(neg.sum()),
                    "pair_count": pair_count,
                    "pair_fraction": pair_count / total_pairs if total_pairs else np.nan,
                    "concordance": concordance,
                    "pooled_auroc_contribution": (
                        pair_count / total_pairs * concordance if pair_count else 0.0
                    ),
                }
            )
    result = pd.DataFrame(rows)
    diagonal = result["positive_site"] == result["negative_site"]
    within_pairs = int(result.loc[diagonal, "pair_count"].sum())
    cross_pairs = int(result.loc[~diagonal, "pair_count"].sum())
    within_contribution = float(result.loc[diagonal, "pooled_auroc_contribution"].sum())
    cross_contribution = float(result.loc[~diagonal, "pooled_auroc_contribution"].sum())
    within_concordance = (
        float(
            np.average(
                result.loc[diagonal, "concordance"],
                weights=result.loc[diagonal, "pair_count"],
            )
        )
        if within_pairs
        else np.nan
    )
    cross_concordance = (
        float(
            np.average(
                result.loc[~diagonal, "concordance"],
                weights=result.loc[~diagonal, "pair_count"],
            )
        )
        if cross_pairs
        else np.nan
    )
    summary = {
        "stage": stage,
        "feature_family": family,
        "total_pairs": total_pairs,
        "within_pairs": within_pairs,
        "cross_pairs": cross_pairs,
        "within_pair_fraction": within_pairs / total_pairs,
        "cross_pair_fraction": cross_pairs / total_pairs,
        "within_concordance": within_concordance,
        "cross_concordance": cross_concordance,
        "within_auroc_contribution": within_contribution,
        "cross_auroc_contribution": cross_contribution,
        "reconstructed_pooled_auroc": within_contribution + cross_contribution,
    }
    direct = np.mean(
        [auc_many(y, repetition_scores, np.ones((1, len(y))))[0] for repetition_scores in scores]
    )
    if not np.isclose(summary["reconstructed_pooled_auroc"], direct, atol=1e-12):
        raise AssertionError("Site-pair decomposition does not reconstruct pooled AUROC")
    summary["direct_pooled_auroc"] = float(direct)
    return result, summary


def evaluation_jackknife(
    scores: np.ndarray,
    metadata: pd.DataFrame,
    stage: str,
    family: str,
) -> pd.DataFrame:
    y = metadata["label"].to_numpy(dtype=int)
    datasets = metadata["dataset"].astype(str).to_numpy()
    subjects = metadata["subject_id"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for removed in sorted(np.unique(datasets)):
        keep = datasets != removed
        record_values = [
            float(auc_many(y[keep], score[keep], np.ones((1, keep.sum())))[0])
            for score in scores
        ]
        base = participant_equal_base(subjects[keep])
        participant_values = [
            float(auc_many(y[keep], score[keep], base[None, :])[0]) for score in scores
        ]
        rows.extend(
            [
                {
                    "stage": stage,
                    "feature_family": family,
                    "removed_evaluation_dataset": removed,
                    "weighting": "record",
                    "estimate": float(np.mean(record_values)),
                    "n_records": int(keep.sum()),
                },
                {
                    "stage": stage,
                    "feature_family": family,
                    "removed_evaluation_dataset": removed,
                    "weighting": "participant_equal",
                    "estimate": float(np.mean(participant_values)),
                    "n_records": int(keep.sum()),
                },
            ]
        )
    return pd.DataFrame(rows)


def plot_site_pairs(pair_frame: pd.DataFrame, path: Path) -> None:
    stages = [stage for stage in ("NREM", "REM") if stage in set(pair_frame["stage"])]
    fig, axes = plt.subplots(1, len(stages), figsize=(12, 5.2), squeeze=False)
    for axis, stage in zip(axes[0], stages):
        subset = pair_frame[pair_frame["stage"] == stage]
        sites = sorted(subset["positive_site"].unique())
        matrix = subset.pivot(
            index="positive_site", columns="negative_site", values="concordance"
        ).reindex(index=sites, columns=sites)
        counts = subset.pivot(
            index="positive_site", columns="negative_site", values="pair_count"
        ).reindex(index=sites, columns=sites)
        image = axis.imshow(matrix.to_numpy(), vmin=0.3, vmax=0.7, cmap="RdBu_r")
        for row in range(len(sites)):
            for column in range(len(sites)):
                value = matrix.iloc[row, column]
                count = int(counts.iloc[row, column])
                axis.text(
                    column,
                    row,
                    f"{value:.2f}\n{count:,}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value - 0.5) > 0.12 else "black",
                )
        labels = [DISPLAY.get(site, site) for site in sites]
        axis.set_xticks(range(len(sites)), labels, rotation=35, ha="right")
        axis.set_yticks(range(len(sites)), labels)
        axis.set_xlabel("No-experience site")
        axis.set_ylabel("Experience site" if stage == "REM" else "")
        axis.set_title(f"{stage}: {PRIMARY[stage]}")
    fig.subplots_adjust(left=0.11, right=0.88, top=0.87, bottom=0.20, wspace=0.30)
    color_axis = fig.add_axes([0.91, 0.22, 0.015, 0.58])
    cbar = fig.colorbar(image, cax=color_axis)
    cbar.set_label("Concordance")
    fig.suptitle("Site-pair decomposition of participant-held-out AUROC")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_target_laboratories(lodo_path: Path, folds_path: Path, path: Path) -> None:
    inference = pd.read_csv(lodo_path)
    folds = pd.read_csv(folds_path)
    panels = [
        ("NREM", ["PSD (published)"], "NREM PSD"),
        ("REM", ["Catch22 filtered (adapted)"], "REM filtered Catch22"),
        (
            "REM",
            ["Temporal irreversibility", "Surrogate-normalized irreversibility"],
            "REM temporal asymmetry controls",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.6), sharex=True)
    colors = ["#2b6cb0", "#c05621"]
    for axis, (stage, families, title) in zip(axes, panels):
        panel = inference[
            (inference["stage"] == stage)
            & (inference["feature_family"].isin(families))
            & (inference["held_out_dataset"] != "MACRO_MEAN")
        ].copy()
        sites = sorted(panel["held_out_dataset"].unique())
        y_positions = np.arange(len(sites))
        for family_index, family in enumerate(families):
            current = panel[panel["feature_family"] == family].set_index("held_out_dataset")
            offset = (family_index - (len(families) - 1) / 2) * 0.16
            values = current.reindex(sites)["subject_equal_auroc"].to_numpy(float)
            lows = current.reindex(sites)["cluster_ci_2.5"].to_numpy(float)
            highs = current.reindex(sites)["cluster_ci_97.5"].to_numpy(float)
            axis.errorbar(
                values,
                y_positions + offset,
                xerr=np.vstack([values - lows, highs - values]),
                fmt="o",
                color=colors[family_index],
                capsize=2,
                label=family,
            )
            record_targets = folds[
                (folds["stage"] == stage)
                & (folds["feature_family"] == family)
                & (folds["regime"] == "laboratory_held_out")
            ]
            record_macro = float(record_targets["auroc"].mean())
            subject_macro = float(current["subject_equal_auroc"].mean())
            axis.axvline(record_macro, color=colors[family_index], linestyle=":", alpha=0.55)
            axis.axvline(subject_macro, color=colors[family_index], linestyle="-", alpha=0.35)
        labels = []
        first_family = panel[panel["feature_family"] == families[0]].set_index(
            "held_out_dataset"
        )
        for site in sites:
            row = first_family.loc[site]
            labels.append(
                f"{DISPLAY.get(site, site)}\n"
                f"n={int(row['n_subjects'])}; E={int(row['experience'])}; NE={int(row['no_experience'])}"
            )
        axis.set_yticks(y_positions, labels)
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_xlim(0.15, 0.95)
        axis.set_xlabel("Participant-equal AUROC (95% cluster CI)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        if len(families) > 1:
            handles, labels_for_legend = axis.get_legend_handles_labels()
    fig.suptitle(
        "Laboratory-held-out performance by target collection\n"
        "Solid and dotted vertical lines mark participant-equal and record-weighted macro means"
    )
    fig.legend(
        handles,
        labels_for_legend,
        fontsize=8,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.82, 0.015),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_diagnostics(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    baselines: pd.DataFrame,
    folds: pd.DataFrame,
    out: Path,
    bootstraps: int,
) -> dict[str, object]:
    composition = composition_table(frame)
    composition.to_csv(out / "stage-specific-composition.csv", index=False)
    estimate_rows: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []
    pair_frames: list[pd.DataFrame] = []
    pair_summaries: list[dict[str, float]] = []
    jackknives: list[pd.DataFrame] = []
    delta_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "seed": SEED,
        "bootstraps": bootstraps,
        "primary": {},
    }
    for stage_index, stage in enumerate(("NREM", "REM")):
        metadata = record_metadata(frame, stage)
        y = metadata["label"].to_numpy(dtype=int)
        datasets = metadata["dataset"].astype(str).to_numpy()
        subjects = metadata["subject_id"].astype(str).to_numpy()
        fold_matrix = fold_assignment_matrix(folds, metadata, stage)
        point_clusters = np.ones((1, len(metadata)), dtype=float)
        bootstrap_clusters = cluster_bootstrap_weights(
            metadata, bootstraps, SEED + 1000 * (stage_index + 1)
        )
        bases = {
            "record": np.ones(len(metadata), dtype=float),
            "participant_equal": participant_equal_base(subjects),
        }
        family_cache: dict[tuple[str, str], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
        for family in sorted(predictions.loc[predictions["stage"] == stage, "feature_family"].unique()):
            scores = score_matrix(predictions, metadata, stage, family)
            for weighting, base in bases.items():
                point = evaluate_scores(
                    scores, metadata, point_clusters, base, fold_matrix
                )
                bootstrap = evaluate_scores(
                    scores, metadata, bootstrap_clusters, base, fold_matrix
                )
                family_cache[(family, weighting)] = (point, bootstrap)
                append_estimates(
                    estimate_rows,
                    "eeg_model",
                    stage,
                    family,
                    weighting,
                    point,
                    bootstrap,
                    metadata,
                )
                if family == PRIMARY[stage]:
                    site_rows.extend(
                        site_specific_rows(
                            stage, family, weighting, point, bootstrap, metadata
                        )
                    )
        baseline_matrices = {
            "record_prevalence": baseline_score_matrix(
                baselines, metadata, stage, "record_prevalence_probability"
            ),
            "participant_prevalence": baseline_score_matrix(
                baselines, metadata, stage, "participant_prevalence_probability"
            ),
        }
        baseline_cache: dict[
            tuple[str, str], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        ] = {}
        for baseline_name, scores in baseline_matrices.items():
            for weighting, base in bases.items():
                point = evaluate_scores(
                    scores, metadata, point_clusters, base, fold_matrix
                )
                bootstrap = evaluate_scores(
                    scores, metadata, bootstrap_clusters, base, fold_matrix
                )
                baseline_cache[(baseline_name, weighting)] = (point, bootstrap)
                append_estimates(
                    estimate_rows,
                    f"site_only_{baseline_name}",
                    stage,
                    "No EEG features",
                    weighting,
                    point,
                    bootstrap,
                    metadata,
                )
        primary_scores = score_matrix(predictions, metadata, stage, PRIMARY[stage])
        pairs, pair_summary = site_pair_decomposition(
            primary_scores, metadata, stage, PRIMARY[stage]
        )
        pair_frames.append(pairs)
        pair_summaries.append(pair_summary)
        jackknives.append(
            evaluation_jackknife(primary_scores, metadata, stage, PRIMARY[stage])
        )
        for weighting in bases:
            model_point, model_boot = family_cache[(PRIMARY[stage], weighting)]
            for baseline_name in baseline_matrices:
                baseline_point, baseline_boot = baseline_cache[(baseline_name, weighting)]
                for estimand in (
                    "pooled",
                    "site_macro",
                    "within_pair_weighted",
                    "site_macro_fold_conditioned",
                    "within_pair_weighted_fold_conditioned",
                ):
                    delta_boot = model_boot[estimand] - baseline_boot[estimand]
                    low, high, valid = interval(delta_boot)
                    delta_rows.append(
                        {
                            "stage": stage,
                            "feature_family": PRIMARY[stage],
                            "baseline": baseline_name,
                            "weighting": weighting,
                            "estimand": estimand,
                            "estimate": float(
                                model_point[estimand][0] - baseline_point[estimand][0]
                            ),
                            "ci_low": low,
                            "ci_high": high,
                            "valid_bootstraps": valid,
                        }
                    )
        # Prespecified REM sensitivity excluding Turku. The point estimate and
        # cluster interval reuse stored OOF predictions; no model is retrained.
        if stage == "REM":
            keep = datasets != "rem_Turku"
            rem_metadata = metadata.loc[keep].reset_index(drop=True)
            rem_scores = primary_scores[:, keep]
            rem_folds = fold_matrix[:, keep]
            rem_point_clusters = np.ones((1, len(rem_metadata)), dtype=float)
            rem_boot = cluster_bootstrap_weights(rem_metadata, bootstraps, SEED + 9000)
            for weighting, base in {
                "record": np.ones(len(rem_metadata)),
                "participant_equal": participant_equal_base(
                    rem_metadata["subject_id"].astype(str).to_numpy()
                ),
            }.items():
                point = evaluate_scores(
                    rem_scores, rem_metadata, rem_point_clusters, base, rem_folds
                )
                bootstrap = evaluate_scores(
                    rem_scores, rem_metadata, rem_boot, base, rem_folds
                )
                append_estimates(
                    estimate_rows,
                    "sensitivity_rem_excluding_turku",
                    stage,
                    PRIMARY[stage],
                    weighting,
                    point,
                    bootstrap,
                    rem_metadata,
                )
        primary_record, _ = family_cache[(PRIMARY[stage], "record")]
        site_record, _ = baseline_cache[("record_prevalence", "record")]
        summary["primary"][stage] = {
            "feature_family": PRIMARY[stage],
            "pooled_participant_held_out_auroc": float(primary_record["pooled"][0]),
            "site_macro_auroc": float(primary_record["site_macro"][0]),
            "within_pair_weighted_auroc": float(primary_record["within_pair_weighted"][0]),
            "site_only_pooled_auroc": float(site_record["pooled"][0]),
            "site_only_within_pair_weighted_auroc": float(
                site_record["within_pair_weighted"][0]
            ),
            "fold_conditioned_within_site_auroc": float(
                primary_record["within_pair_weighted_fold_conditioned"][0]
            ),
            "site_only_fold_conditioned_within_site_auroc": float(
                site_record["within_pair_weighted_fold_conditioned"][0]
            ),
        }
    estimates = pd.DataFrame(estimate_rows)
    estimates.to_csv(out / "targeted-estimates.csv", index=False)
    pd.DataFrame(site_rows).to_csv(out / "primary-site-estimates.csv", index=False)
    pairs = pd.concat(pair_frames, ignore_index=True)
    pairs.to_csv(out / "site-pair-concordance.csv", index=False)
    pd.DataFrame(pair_summaries).to_csv(out / "site-pair-summary.csv", index=False)
    pd.concat(jackknives, ignore_index=True).to_csv(
        out / "evaluation-composition-jackknife.csv", index=False
    )
    pd.DataFrame(delta_rows).to_csv(out / "paired-model-baseline-differences.csv", index=False)
    plot_site_pairs(pairs, out / "site-pair-heatmaps.png")
    plot_target_laboratories(
        DEFAULT_LODO, DEFAULT_FOLDS, out / "target-laboratory-forest.png"
    )
    (out / "targeted-revision-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--surrogates", type=Path, default=DEFAULT_SURROGATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--force-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_features(args.features, args.surrogates)
    prediction_path = args.out / "targeted-oof-predictions.csv"
    baseline_path = args.out / "site-only-oof-predictions.csv"
    folds_path = args.out / "targeted-fold-definitions.csv"
    if (
        not args.force_predictions
        and prediction_path.exists()
        and baseline_path.exists()
        and folds_path.exists()
    ):
        predictions = pd.read_csv(prediction_path, low_memory=False)
        baselines = pd.read_csv(baseline_path, low_memory=False)
        folds = pd.read_csv(folds_path, low_memory=False)
    else:
        predictions, baselines, folds = generate_predictions(frame, args.out, args.repeats)
    validate_prediction_contract(frame, predictions, baselines, folds, args.repeats)
    summary = run_diagnostics(
        frame, predictions, baselines, folds, args.out, args.bootstraps
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
