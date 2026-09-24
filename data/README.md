# Data

The raw LP-STEM acquisitions are **not stored in this repository**. Each Velox `.emd`
HAADF stack is >100 MB, above GitHub's hard per-file limit, so they are archived on
Zenodo instead.

## Download

**Zenodo record:** <!-- TODO: replace with the DOI once the record is published -->
`https://doi.org/10.5281/zenodo.XXXXXXX`

Download the `.emd` files into this folder:

```
data/
├── 20250110_HAADF_STEM_150_kx_1051.emd
├── 20250110_HAADF_STEM_210_kx_1053.emd
└── 20250110_HAADF_STEM_150_kx_1106.emd
```

Or from the command line:

```bash
cd data
curl -L -O "https://zenodo.org/records/XXXXXXX/files/20250110_HAADF_STEM_210_kx_1053.emd?download=1"
```

The engine scripts in `tracking_algo/` expect to find the stacks here. Adjust the path
constant at the top of each script if you keep them elsewhere.

## Contents of the archive

| file | movie | magnification | pixel size | frames |
|---|---|---|---|---|
| `…_150_kx_1051.emd` | 1051 | 150 kx | 0.651 nm | 69 |
| `…_210_kx_1053.emd` | 1053 | 210 kx | 0.460 nm | 72 |
| `…_150_kx_1106.emd` | 1106 | 150 kx | 0.651 nm | — |

All frames of each stack are analysed; the engine does not drop or mask any.

Also archived: `PP_tracks_20250110-1053-PtDum.csv` and the `HAADF011–070` display crops —
the hand-annotated dumbbell trajectory used to validate the tracker.

## Important

**Read the pixel size and frame time from each file's own metadata — never inherit them
between files.** Two acquisitions whose filenames report the same magnification have been
found to differ in pixel size. Each engine script reads these values itself; do not
hardcode them.

Confirm you are reading the **HAADF** dataset and not bright-field. Some files contain
both at identical shape, and BF is contrast-inverted — it will silently produce garbage.
