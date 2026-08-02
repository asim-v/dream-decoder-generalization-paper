# DREAM data sources and integrity record

All files are public under CC BY 4.0. Retrieved 2026-08-02. Raw archives and
expanded EDF files are git-ignored; registry metadata and analysis outputs are
kept in the repository.

## Registry

- Landing page and DOI: <https://doi.org/10.26180/22133105>
- `People.csv`: file 49774977, MD5 `5846dc5022a5c36619557d409bcbb3cb`
- `Data records.csv`: file 59170370, MD5 `6f9be5b4abb0b603ee802fffb015155f`
- `Datasets.csv`: file 59170373, MD5 `6382c9feafcf396acf85ab09e5a3298b`

The registry retains obsolete amendments. `inventory_dream_registry.py` keeps
only rows marked `Latest amendment = TRUE` and not revoked, then matches records
to that amendment.

## Pilot archives

| Dataset | Landing page | Figshare file | Bytes | MD5 |
|---|---|---:|---:|---|
| LODE data | <https://doi.org/10.6084/m9.figshare.22147085.v1> | 39373088 | 133,574,623 | `36ddc1ffe4e2867acbeb51c6382601f2` |
| Dream_YoungAdults | <https://doi.org/10.6084/m9.figshare.14899506.v1> | 39673915 | 36,782,182 | `f4fba4f3b78ce13d54ad424ed2c09aef` |
| Zhang & Wamsley 2019 | <https://doi.org/10.6084/m9.figshare.22226692.v1> | 39504757 | 1,049,159,479 | `5854cfea4925f57d4d0a440518f4b72a` |

LODE metadata are separate files in the same deposit:

- `ExperimentalDescription.txt`, file 39373079, MD5 `6af3a62ddaa0681cc355cb96ca0eac85`;
- `README.txt`, file 39373082, MD5 `6ea141322b9d22ec1db7dc49143d921f`;
- `Records.csv`, file 39373085, MD5 `1e534fb7627bf772a1f97e04179d0f0f`.

## Publisher-side archive defect

The local Dream_YoungAdults ZIP matches Figshare's published MD5, but CRC checks
fail for `DG05.edf`, `DG41.edf`, and `DM02.edf`. The first partial extraction was
removed. All three files are excluded in `data/dream-consciousness/exclusions.csv`.
The remaining 62 EDF files parse successfully.

## DREAM Figure 5 code and source collections

The article is <https://doi.org/10.1038/s41467-025-61945-1>. Its Figure 5 code
was retrieved from <https://doi.org/10.5281/zenodo.15234845> and preserved
verbatim under `data/dream-consciousness/dream-fig5-code/`:

| File | MD5 |
|---|---|
| `Readme.txt` | `e5d7f2559a7b0591b693cd344a29bf2c` |
| `workflow_exp_classification.py` | `e05b9265e4b6be835dbb9270f56c102a` |

Additional public collections retrieved and checksum-verified on 2026-08-02:

| Local key | Versioned DOI | Files | Total bytes |
|---|---|---:|---:|
| `degen_ma` | <https://doi.org/10.6084/m9.figshare.22086266.v2> | 1 | 6,395,662,397 |
| `oudiette` | <https://doi.org/10.6084/m9.figshare.22210684.v1> | 1 | 43,261,557 |
| `siclari` | <https://doi.org/10.26180/23306054.v3> | 38 | 8,409,097,333 |
| `turku` | <https://doi.org/10.6084/m9.figshare.23274596.v2> | 1 | 363,436,164 |

All are marked CC BY 4.0 by their Figshare records. Per-file IDs, byte counts,
MD5 hashes, and download URLs are pinned in `fig5-source-files.json`.

Oudiette is one of six collections named for the article's broader EEG
analyses, but the deposited Figure 5 classifier enumerates five. Its sleeping
subset has only one strict No-experience record and its Fp1/C3/O1 montage is not
the classifier's common F4/C4/O2 montage, so it is audited but not inserted into
the classifier post hoc.
