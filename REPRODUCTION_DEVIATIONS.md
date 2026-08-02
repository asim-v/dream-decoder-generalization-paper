# Reproduction Deviations

Audit date: 2026-08-02.

This document distinguishes literal numerical reproduction of the deposited DREAM Figure 5 workflow from a conceptual reproduction of its stated analysis. The official Nature Communications article links the Figure 5 example-code deposit at <https://doi.org/10.5281/zenodo.15234845>. That concept DOI currently resolves to Zenodo record 15234846, published 2025-04-17, containing only `Readme.txt` and `workflow_exp_classification.py` under CC BY 4.0.

The source archives used here are pinned by version, file identifier, byte count, and MD5 in `data/dream-consciousness/fig5-source-files.json`.

| Issue | Deposited workflow expects | Publicly available material | Minimum adaptation | Expected effect or uncertainty | Reproduction scope affected |
|---|---|---|---|---|---|
| Missing feature module | `import feat_ext_funcs as fext` and `fext.get_22_plus_mvit(...)` for band-filtered features | The Zenodo record has two files and does not contain `feat_ext_funcs`, a package specification, or a link to it | Compute the 22 canonical Catch22 features directly with `pycatch22` after each band-pass filter | The REM filtered-Catch22 benchmark cannot be reproduced numerically; feature ordering or any unreported additional statistic may differ | Exact numerical reproduction; conceptual feature-family reproduction retained |
| Siclari preprocessing | Files named `formatted_*.edf` exposing `F4-M1`, `C4-M1`, and `O2-M1` | Figshare versions 1-3 contain high-density EDFs with numerical channel names and no formatted EDFs | Reconstruct F4/C4/O2-M1 as HydroCel E224/E164/E150 minus E93; eight files lacking E93 use the nearest available left-mastoid candidate | Montage and reference differences can change every downstream feature and classifier score | Exact and conceptual reproduction for the Siclari contribution |
| De Gennaro delimiter | Semicolon delimiter for Multiple Awakenings | The current public `Records.csv` is comma-separated | Detect/use the comma delimiter | Necessary for execution; no expected numerical change after correct parsing | Literal execution only |
| Dataset scope | Article prose describes six collections used for EEG analyses; Figure 5 code enumerates five | Oudiette N1Data is the plausible sixth collection, but has Fp1/C3/O1 rather than F4/C4/O2 and only one strict sleeping No-experience observation | Audit Oudiette but do not insert it post hoc into the five-collection classifier | The exact intended Figure 5 scope remains ambiguous | Conceptual scope and exact sample definition |
| Current archive versions | A fixed analysis-time archive state, not identified in the code deposit | Current API history exposes De Gennaro Multiple Awakenings v1-v2, Siclari v1-v3, Turku v1-v2, and Oudiette v1 | Pin the exact retrieved versions and every file checksum | Later amendments may alter files or metadata relative to the authors' environment | Exact reproduction and sample composition |
| Missing or corrupt public EDFs | One usable EDF per referenced record | Three Young Adults EDF members fail CRC while matching the publisher ZIP MD5; one Turku EDF named by metadata is absent | Exclude `DG05.edf`, `DG41.edf`, `DM02.edf`, and `case133_s27.edf`; record every exclusion | Changes sample size, class balance, and fitted decision boundaries | Exact and conceptual reproduction |
| Siclari mastoid availability | A uniform M1 reference for every formatted file | Eight public high-density files lack E93 | Use the nearest documented left-mastoid candidate and record the adaptation per epoch | Small, directionally unknown changes in covariance, power, and time-series features | Exact reproduction; limited conceptual impact |
| Randomness | Repeated random splits and label shuffles; the script draws an unrecorded Python random seed and does not set an XGBoost seed | No seed file or saved folds | Use seed 20260802, deterministic five-fold splits, one thread, and publish every fold and OOF probability | Reproduced distributions need not match the authors' particular random draws | Exact numerical reproduction; improves auditability |
| Software environment | Python 3.12.3 and an unversioned package list | No lockfile or exact package versions | Pin the tested dependency versions in the public environment file and add CI | Library defaults, filters, and XGBoost behavior may differ across releases | Exact numerical reproduction |
| Boosting rounds | `xgb.train(params, dtrain)` without an explicit round count | XGBoost's API default is ten rounds in the tested release | Set and report ten rounds explicitly | Protects against future default changes; matches the present implicit behavior | Future exact reproduction |
| Label and stage exclusions | Strict Experience/No-experience classification after removal of Wake | Public registries also contain ambiguous labels, unknown labels, Wake, and corrupted/unavailable files | Retain only strict binary labels and NREM/REM records, and publish the exclusion log | Produces 1,065 records rather than every registry row | Exact sample definition and conceptual interpretation |
| Participant identity | Grouping must distinguish equal local identifiers from different collections | Some identifiers are only unique inside their source dataset | Prefix every subject identifier with the dataset key | Prevents false grouping or leakage across collections | All participant-level validation and uncertainty |

## Inspected official states

- Zenodo concept DOI 10.5281/zenodo.15234845 and resolved record 15234846: two files, no tagged software release, branch, environment, or auxiliary feature module.
- Figshare version histories: De Gennaro Multiple Awakenings v1-v2; Oudiette v1; Siclari v1-v3; REM Turku v1-v2.
- DREAM database registry and current non-revoked amendments.
- Nature article, Methods, supplementary links, Data availability, and Code availability statements.
- The deposited script and README, including channel lists, filtering, PSD normalization, cross-validation, XGBoost parameters, and file-name assumptions.

No public branch, tag, supplement, or linked archive inspected on 2026-08-02 supplied `feat_ext_funcs`, the Siclari `formatted_*.edf` files or generation script, a complete environment lockfile, or fixed random seeds.
