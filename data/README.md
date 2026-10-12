# Data

The raw LP-STEM acquisitions are **not stored in this repository**. Each `.dm4`
HAADF stack is >100 MB, above GitHub's hard per-file limit, so they are archived on
Zenodo instead, together with the Velox `.txt` metadata file that carries each
acquisition's pixel size and frame time (the engine reads these from the `.txt`, not
from the `.dm4` tag).

## Download

**Zenodo record:** 
`https://doi.org/10.5281/zenodo.23319305`

Download each stack with its `.txt` into a folder named `images_<id>_full/`:

```
images_1051_benchmark.zip/
images_1053_benchmark.zip/
images_1051_full/
├── 20250110_HAADF_1051_substack(33-68).dm4
└── 20250110_HAADF_1051_substack(33-68).txt
images_1053_full/
├── 20250110_HAADF_1053_substack(11-71).dm4
└── 20250110_HAADF_1053_substack(11-71).txt
images_1102_full/
├── 20250110_HAADF_1102_substack(18-51).dm4
└── 20250110_HAADF_1102_substack(18-51).txt
```

The scripts in `tracking_algo/` look for these folders relative to the directory they are
run from (`FOLDER = "images_<id>_full"` in the engine). Adjust that constant if you keep
the data elsewhere.

