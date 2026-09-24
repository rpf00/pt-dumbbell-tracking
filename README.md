# Pt–PVP–Pt Dumbbell Detection and Tracking

Automated detection, tracking and lifetime analysis of transient Pt–PVP–Pt "dumbbell"
intermediates in liquid-phase scanning transmission electron microscopy (LP-STEM) movies
of Pt nanoparticle micelle disassembly.

Supporting code for the manuscript "Resolving Dynamic Interparticle Interactions through Metastable Pt–PVP–Pt Dumbbells during Micelle Disassembly", Figures 3c/3d and Discussion S2.

---

## What the workflow does

Each LP-STEM acquisition (`.emd` file) is converted into two quantitative
measurements, _i.e._, the distribution of **bonded lifetimes** of individual dumbbells, and the
**live dumbbell population** as a function of time.

```
.emd stack
   │
   ├─ A. Ingestion        read pixel size, frame time and HAADF dataset UID from the
   │                      file's own metadata
   │
   ├─ B. Normalization    per-frame p1–p99.9 percentile clip
   │
   ├─ C. Denoising        Gaussian filter, σ specified in nm
   │
   ├─ D. Detection        Laplacian-of-Gaussian blob detection, threshold as a
   │                      fraction of each frame's own contrast percentile
   │
   ├─ E. Pairing          mutual-nearest-neighbour pairs inside the birth band
   │
   ├─ F. Tracking         two-particle propagation to dissociation
   │                      (predict → distance-penalised peak → centroid → step clamp)
   │
   ├─ G. Gating           physical acceptance criteria (below)
   │
   └─ H. Output           lifetime distribution + live population + drift retest
```

### Brief description of each step

**1) Normalization:** The nanoparticles brighten substantially over the course of the data acquisition process. 
Thus, the min–max normalization scheme (per-frame) would provide a fixed detection threshold, making 
the detector itself drift and leading to false dumbbell population estimates. The percentile clipping approach
ensures that sensitivity remains constant across all frames.

**2) Denoising:** The observed noise in liquid-cell water-scatter frames is spatially correlated. 
Some classical denoisers (_e.g._, NL-means, wavelet, total-variation) inject spurious
intensity maxima, which the tracker then follows instead of real particles. We then use a
Gaussian filter to avoid this issue. Its width is chosen by maximizing contrast-to-noise ratio,
subject to the detected particle diameter being consistent with the known ~2.4 nm Pt nanoparticle size. 

**3) Velocity correlation:** Two unrelated particles carried by the same fluid
flow show high positional correlation regardless of any physical association.
Differencing (_i.e._, taking velocity correlation instead) removes this common drift, so co-motion is 
evaluated based on velocities.

**4) Per-frame separation floor.** Two ~2.4 nm particles cannot approach more closely than
their own diameter. We consider pairs that do so as part of trajectories that converged to the same particle 
(and are ultimately discarded).

---

## Core principle

> **Physical criteria are constant in nanometres. Every pixel-space parameter is
> re-derived per movie.**

Pixel size, particle density, contrast and motion differ between acquisitions, so the
detector is re-fitted for each movie. What counts as a dumbbell does not change:

| criterion | value |
|---|---|
| birth (mutual-NN separation at formation) | 2.5 – 6.5 nm |
| death (dissociation) | separation ≥ 7.0 nm |
| minimum bonded duration | 6 consecutive frames |
| bonded-phase mean separation | 2.5 – 6.5 nm |
| hard per-frame separation floor | ≥ 2.4 nm |
| velocity correlation | ≥ 0.50 |

`constants.txt` is split along exactly this line. Its **DETECTOR** block (normalization
percentiles, gaussian σ, blob scale, threshold) must be re-derived for every new
acquisition (those numbers do not transfer automatically). **PHYSICAL CRITERIA** is the table
above and should not be retuned per movie, because if it changes, results stop being
comparable between movies. The **TRACKER GEOMETRY** block scales
automatically with pixel size, but `MAX_STEP` should be measured from the real
frame-to-frame displacement distribution.

---

## Results

| movie | magnification | pixel size (nm) | mean LT (s) | **median LT (s)** | drift retest |
|---|---|---|---|---|---|
| 1051 | 150 kx | 0.651 | 6.6 | **5.6** | r = 0.10, p = 0.41 |
| 1053 | 210 kx | 0.460 | 7.4 | **6.4** | r = 0.16, p = 0.39 |
| 1106 | 150 kx | 0.651 | 6.2 | **5.6** | r = 0.19, p = 0.21 |

The median bonded lifetime is the most relevant result. It is computed per-dumbbell and does not
depend on how many particles are detected. From the above table, we can see it does not change across 
different detector settings, agreeing within ~0.8 s across three independent acquisitions at two magnifications.

The **drift retest** correlates frame-to-frame changes in detection yield against
frame-to-frame changes in dumbbell births. Differencing removes any shared trend, so only
fast coupling survives. A null result means fluctuations in the dumbbell population are
not driven by fluctuations in detector sensitivity. It is not a test of dumbbells against
noise — that is the job of the physical criteria above.

---

## Important observations

1. **Lifetimes are left-truncated and right-censored.** The six-frame minimum means no
   lifetime shorter than `6 × frame time` (~ 4.78 s) can be measured, and dumbbells that are still
   being formed over the final 6 frames are not accounted for.
   Hence why we focus on reporting the life-time median, rather than the mean.
2. **Dumbbell counts do not transfer between movies.** Detection counts, accepted counts and
   population magnitudes depend on the field of view and the detector. Only the lifetime
   distribution is meant to be compared across acquisitions.

---

## Repository layout

```
pt-dumbbell-tracking/
├── README.md
├── data/                     pointer to the raw .emd stacks on Zenodo (not in git)
│   └── README.md
├── tracking_algo/
│   ├── population_engine_optimal.py    the engine (no parameters inside it)
│   └── constants.txt                   every tunable parameter lives here
└── results/
    ├── exp_1051/
    ├── exp_1053/
    └── exp_1106/
        └── constants_used.txt          the exact settings that produced this run
```

> **Update `tracking_algo/constants.txt` to fit your specific problem. The file as
> shipped corresponds to the default values of our experiment (exp_1053).**

There is one engine and one parameter file. The engine reads `constants.txt` 
so new acquisition frames will not need any code changes (only a new set of
constants). Each `results/exp_*/` folder keeps a copy of the constants that produced it.
The user should copy one over `tracking_algo/constants.txt` to reproduce that specific run.

When editing `constants.txt`, the format is `KEY = value` (`#` starting a comment). A
missing key, an unparseable value or an invalid mode is reported by name at startup
rather than failing partway through a long run.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib h5py scikit-image scipy

# download the .emd stacks from Zenodo into data/ first (see data/README.md)
cd tracking_algo

# edit constants.txt to point at your movie and match your detector, then:
python population_engine_optimal.py
```

To reproduce one of the published runs, copy its recorded constants first:

```bash
cp ../results/exp_1051/constants_used.txt constants.txt
python population_engine_optimal.py
```

Each script prints a summary table and writes into `Figs/`:

- `population_dumbbell_dynamics_<TAG>.png` (the three-panel overview)
- `lifetime_distribution_<TAG>.pdf` / `.png` (standalone panel (a))
- `live_population_<TAG>.pdf` / `.png` (standalone panel (c))
- `population_curve_<TAG>.npz` (births/deaths arrays)

Move the contents of `Figs/` into the matching `results/exp_<TAG>/` folder to archive a
run, along with the `constants.txt` that produced it.

The engine prints the detection count (at frame 0) as a sanity check. In our case, 
a few hundred to a few thousand are reasonable values. It might change depending on the users' experiments.

## Requirements

Python 3.9+, with `numpy`, `pandas`, `matplotlib`, `h5py`, `scikit-image` and `scipy`.
