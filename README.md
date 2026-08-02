# Cross-Laboratory Validation of EEG Classification of Reported Dream Experience

This repository reproduces the analyses and manuscript for a validation audit
of the DREAM Figure 5 experience-classification workflow. The central question
is whether participant-held-out EEG discrimination remains when ranking is
restricted to records collected in the same laboratory, and whether the fitted
rule transfers to an unseen laboratory.

The release contains no raw EEG. It includes the frozen derived feature
matrices, deterministic folds, complete out-of-fold probabilities,
machine-readable result tables, analysis and validation code, generated LaTeX
assets, figures, and the compiled paper.

## Reproduce

Create the tested environment:

```bash
conda env create -f environment.yml
conda activate dream-validation
```

Then run the single reproduction command from the repository root:

```bash
python reproduce.py
```

This command regenerates all participant-held-out model predictions from the
included feature matrices, recomputes 2,000 participant-cluster bootstrap
samples, runs the contract tests, validates the release outputs, generates the
LaTeX numbers and tables, refreshes both figures, and builds `paper/paper.pdf`
with Tectonic.

For a fast integrity check that reuses the published out-of-fold predictions:

```bash
python reproduce.py --reuse-predictions
```

The full run is deterministic but may take several minutes on a laptop. Model
training uses one CPU thread to make reruns stable across machines.

## Expected primary results

The full run should recover these participant-held-out record-weighted AUROCs:

| Stage and primary family | Pooled | Same-laboratory pairs | Fold-conditioned same-laboratory pairs | Laboratory held out |
|---|---:|---:|---:|---:|
| NREM PSD | 0.534 | 0.498 | 0.509 | 0.493 |
| REM filtered Catch22 | 0.629 | 0.498 | 0.552 | 0.470 |

The precise estimates, cluster-bootstrap intervals, site-pair decomposition,
site-only baselines, target-laboratory estimates, and sensitivities are stored
under `outputs/dream-consciousness/targeted-revision/`.

## Repository structure

- `reproduce.py`: one-command analysis, validation, asset generation, and PDF build.
- `environment.yml` and `requirements-lock.txt`: tested software environment.
- `analysis/dream-consciousness/`: acquisition, preprocessing, modeling,
  bootstrap, table, figure, test, and validation code.
- `data/dream-consciousness/fig5-source-files.json`: versioned source URLs,
  persistent identifiers, expected byte counts, licenses, and MD5 hashes.
- `data/dream-consciousness/SOURCE.md`: source and integrity documentation.
- `data/dream-consciousness/exclusions.csv`: frozen file-level exclusions.
- `data/dream-consciousness/derived/fig5/`: redistributed derived feature
  matrices used by the models.
- `outputs/dream-consciousness/`: deterministic folds, OOF probabilities,
  machine-readable estimates, diagnostics, and validation report.
- `paper/`: generated tables, figures, LaTeX source, bibliography, and PDF.
- `REPRODUCTION_DEVIATIONS.md`: exact boundary between literal and conceptual
  reproduction of the deposited workflow.

## What cannot be redistributed or reconstructed exactly

Raw and expanded EEG are intentionally excluded. Source archives are available
from the DOI-pinned public records listed in
`data/dream-consciousness/fig5-source-files.json`; together they are much larger
than this release.

Exact numerical reproduction of the upstream Figure 5 feature matrix remains
blocked by two absent upstream artifacts: the imported `feat_ext_funcs` module
and the preformatted Siclari EDF files (or their generation script) expected by
the deposit. The public pipeline therefore uses documented minimum adaptations:
direct `pycatch22` extraction and a stated HydroCel channel/reference mapping.
See `REPRODUCTION_DEVIATIONS.md` before comparing the adapted benchmark with the
published point estimates.

## Independent checks

The release validator checks:

- 1,065 expected records (730 NREM; 335 REM);
- no participant leakage across folds;
- dataset-scoped participant identifiers;
- no test-label use in the site-prevalence baseline;
- exact reconstruction of pooled AUROC from within- and cross-site pairs;
- deterministic fold and model reruns;
- the complete 20-repetition, seven-feature-family prediction contract.

Run the test and validator separately with:

```bash
python analysis/dream-consciousness/test_targeted_revision.py
python analysis/dream-consciousness/validate_targeted_revision.py
```

## Source study

- Wong et al., “A dream EEG and mentation database,” *Nature Communications*
  (2025), DOI: [10.1038/s41467-025-61945-1](https://doi.org/10.1038/s41467-025-61945-1)
- DREAM Figure 5 deposit, Zenodo DOI:
  [10.5281/zenodo.15234845](https://doi.org/10.5281/zenodo.15234845)

Author: Javier Emilio Bazán Sánchez (`bazan@ciencias.unam.mx`).

## License

Manuscript text, analysis code, and original figure composition are released
under [CC BY 4.0](LICENSE). Source datasets retain the licenses stated by their
respective records.
