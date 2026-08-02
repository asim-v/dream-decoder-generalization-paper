#!/usr/bin/env python3
"""Unit and contract tests for the targeted DREAM revision."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_targeted_revision as revision


class MetricTests(unittest.TestCase):
    def test_weighted_auc_matches_sklearn(self) -> None:
        y = np.array([0, 1, 0, 1, 1, 0])
        score = np.array([0.1, 0.9, 0.4, 0.4, 0.7, 0.2])
        weight = np.array([1.0, 2.0, 0.5, 1.5, 1.0, 3.0])
        observed = revision.auc_many(y, score, weight[None, :])[0]
        expected = roc_auc_score(y, score, sample_weight=weight)
        self.assertAlmostEqual(observed, expected, places=14)

    def test_site_pair_decomposition_identity(self) -> None:
        metadata = pd.DataFrame(
            {
                "label": [1, 0, 1, 0, 1, 0],
                "dataset": ["a", "a", "b", "b", "a", "b"],
                "subject_id": ["a:1", "a:2", "b:1", "b:2", "a:3", "b:3"],
            }
        )
        scores = np.array(
            [[0.8, 0.2, 0.7, 0.3, 0.5, 0.5], [0.7, 0.1, 0.6, 0.4, 0.4, 0.4]]
        )
        _, summary = revision.site_pair_decomposition(scores, metadata, "NREM", "test")
        self.assertAlmostEqual(
            summary["reconstructed_pooled_auroc"],
            summary["direct_pooled_auroc"],
            places=14,
        )


class SplitAndBaselineTests(unittest.TestCase):
    @staticmethod
    def fixture() -> pd.DataFrame:
        rows = []
        for site in ("a", "b"):
            for subject in range(10):
                for record in range(2):
                    rows.append(
                        {
                            "record_index": len(rows),
                            "dataset": site,
                            "filename": f"{site}-{subject}-{record}",
                            "subject_id": f"{site}:{subject}",
                            "label": (subject + record) % 2,
                            "stage": "NREM",
                        }
                    )
        return pd.DataFrame(rows)

    def test_splits_are_deterministic_and_leak_free(self) -> None:
        frame = self.fixture()
        first = revision.deterministic_splits(frame, 2)
        second = revision.deterministic_splits(frame, 2)
        groups = frame["subject_id"].to_numpy()
        self.assertEqual(len(first), 10)
        for left, right in zip(first, second):
            self.assertEqual(left[:2], right[:2])
            np.testing.assert_array_equal(left[2], right[2])
            np.testing.assert_array_equal(left[3], right[3])
            self.assertFalse(set(groups[left[2]]) & set(groups[left[3]]))

    def test_site_only_baseline_ignores_test_labels(self) -> None:
        frame = self.fixture()
        splits = revision.deterministic_splits(frame, 1)
        original = revision.site_prevalence_predictions(frame, splits).sort_values(
            ["repetition", "fold", "record_index"]
        )
        changed = frame.copy()
        # Change labels only for the first test fold and recompute that fold.
        repetition, fold, train, test = splits[0]
        changed.loc[test, "label"] = 1 - changed.loc[test, "label"]
        modified = revision.site_prevalence_predictions(
            changed, [(repetition, fold, train, test)]
        ).sort_values(["repetition", "fold", "record_index"])
        reference = original[(original["repetition"] == repetition) & (original["fold"] == fold)]
        np.testing.assert_allclose(
            reference["record_prevalence_probability"],
            modified["record_prevalence_probability"],
        )
        np.testing.assert_allclose(
            reference["participant_prevalence_probability"],
            modified["participant_prevalence_probability"],
        )

    def test_fold_conditioned_site_baseline_is_chance(self) -> None:
        frame = self.fixture()
        splits = revision.deterministic_splits(frame, 2)
        baseline = revision.site_prevalence_predictions(frame, splits)
        fold_rows = []
        for repetition, fold, _, test in splits:
            out = frame.iloc[test][["record_index", "stage"]].copy()
            out["repetition"] = repetition
            out["fold"] = fold
            fold_rows.append(out)
        folds = pd.concat(fold_rows, ignore_index=True)
        metadata = frame.sort_values("record_index").reset_index(drop=True)
        scores = revision.baseline_score_matrix(
            baseline, metadata, "NREM", "record_prevalence_probability"
        )
        fold_matrix = revision.fold_assignment_matrix(folds, metadata, "NREM")
        result = revision.evaluate_scores(
            scores,
            metadata,
            np.ones((1, len(metadata))),
            np.ones(len(metadata)),
            fold_matrix,
        )
        self.assertAlmostEqual(
            result["within_pair_weighted_fold_conditioned"][0], 0.5, places=14
        )

    def test_subject_ids_are_dataset_scoped(self) -> None:
        frame = self.fixture()
        self.assertTrue(
            all(
                subject.startswith(f"{dataset}:")
                for dataset, subject in zip(frame["dataset"], frame["subject_id"])
            )
        )


if __name__ == "__main__":
    unittest.main()
