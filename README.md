# When Dream Decoders Learn the Laboratory

Paper-only companion repository for the two-column preprint:

> *When Dream Decoders Learn the Laboratory: Cross-Dataset Validation and Reproducibility Audit of DREAM Experience Classification*

The manuscript audits the released DREAM Figure 5 workflow and asks whether experience classifiers generalize from pooled or participant-held-out observations to an entirely unseen laboratory. It covers 1,065 binary reports from five public collections (730 NREM; 335 REM), a validation ladder, dataset-identity negative controls, Riemannian covariance features, log-Euclidean alignment, temporal irreversibility, and phase-randomized surrogate controls.

## Contents

- `main.tex` — two-column manuscript source
- `references.bib` — bibliography
- `figures/validation-ladder.png` — principal validation figure
- `paper.pdf` — compiled manuscript snapshot

This repository intentionally excludes raw or expanded EEG, caches, exploratory notebooks, and analysis intermediates. Those materials belong to the separate analysis workspace.

## Build

Install [Tectonic](https://tectonic-typesetting.github.io/) and run:

```powershell
tectonic --keep-logs --keep-intermediates main.tex
```

The author field is currently an anonymous placeholder and should be replaced before submission.

## Source study

- Wong et al., “A dream EEG and mentation database,” *Nature Communications* (2025), DOI: [10.1038/s41467-025-61945-1](https://doi.org/10.1038/s41467-025-61945-1)
- DREAM Figure 5 deposit, Zenodo DOI: [10.5281/zenodo.15234845](https://doi.org/10.5281/zenodo.15234845)

## License

Manuscript text and original figure composition are available under [CC BY 4.0](LICENSE).
