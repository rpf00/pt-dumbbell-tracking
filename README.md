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
   ├─ B. Normalisation    per-frame p1–p99.9 percentile clip
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

**1) Normalization:** The specimen brightens substantially during acquisition as the liquid
layer thins. Under per-frame min–max normalisation a fixed detection threshold therefore
means something different at the start and end of a movie, which makes the detector itself
drift and can manufacture a rise in the dumbbell population. Percentile clipping holds
sensitivity constant across the movie.

**2) Denoising:** Liquid-cell water-scatter noise is *spatially correlated*. Non-local-means,
wavelet and total-variation denoisers interpret it as structure and inject spurious
intensity maxima, which the tracker then follows instead of real particles. A light
Gaussian filter avoids this. Its width is chosen by maximising contrast-to-noise ratio
subject to the blur-deconvolved detected particle diameter remaining consistent with the
known ~2.4 nm Pt nanoparticle size. σ is specified **in nanometres** so the setting is
meaningful across acquisitions of different pixel size.

**3) Velocity correlation:** Two unrelated particles carried by the same fluid
flow show high *positional* correlation regardless of any physical association.
Differencing removes this common-mode drift, so co-motion is assessed on velocities.

**4) Per-frame separation floor.** Two ~2.4 nm particles cannot approach more closely than
their own diameter. Pairs that do are tracker artifacts in which both trajectories have
converged on the same particle, and are rejected.

---

## The core principle

> **Physical criteria are constant in nanometres. Every pixel-space parameter is
> re-derived per movie.**

Pixel size, particle density, contrast and motion differ between acquisitions, so the
*detector* is re-fitted for each movie. What counts as a dumbbell does not change:

| criterion | value |
|---|---|
| birth (mutual-NN separation at formation) | 2.5 – 6.5 nm |
| death (dissociation) | separation ≥ 7.0 nm |
| minimum bonded duration | 6 consecutive frames |
| bonded-phase mean separation | 2.5 – 6.5 nm |
| hard per-frame separation floor | ≥ 2.4 nm |
| velocity correlation | ≥ 0.50 |

`constants.txt` is split along exactly this line. Its **DETECTOR** block (normalisation
percentiles, gaussian σ, blob scale, threshold) must be re-derived for every new
acquisition — those numbers do not transfer. Its **PHYSICAL CRITERIA** block is the table
above and should not be retuned per movie, because if it changes, results stop being
comparable between movies. The **TRACKER GEOMETRY** block sits in between: it scales
automatically with pixel size, but `MAX_STEP` is worth measuring from the real
frame-to-frame displacement distribution, since the scaled default was found to be far
too tight on two of the three movies here.

---

## Results

| movie | magnification | pixel size (nm) | mean LT (s) | **median LT (s)** | drift retest |
|---|---|---|---|---|---|
| 1051 | 150 kx | 0.651 | 6.6 | **5.6** | r = 0.10, p = 0.41 |
| 1053 | 210 kx | 0.460 | 7.4 | **6.4** | r = 0.16, p = 0.39 |
| 1106 | 150 kx | 0.651 | 6.2 | **5.6** | r = 0.19, p = 0.21 |

The median bonded lifetime is the load-bearing number: it is per-dumbbell and does not
depend on how many particles are detected. It was unchanged across detector settings that
varied the accepted-dumbbell count by a factor of 2.2 on a single movie, and agrees to
within ~0.8 s across three independent acquisitions at two magnifications.

The **drift retest** correlates frame-to-frame changes in detection yield against
frame-to-frame changes in dumbbell births. Differencing removes any shared trend, so only
fast coupling survives. A null result means fluctuations in the dumbbell population are
not driven by fluctuations in detector sensitivity. It is not a test of dumbbells against
noise — that is the job of the physical criteria above.

---

## Caveats

These apply to every result in this repository and belong in any figure caption derived
from it.

1. **Lifetimes are left-truncated and right-censored.** The six-frame minimum means no
   lifetime shorter than `6 × frame time` (≈4.78 s) can be measured, and dumbbells still
   bonded when the movie ends are censored. Report the **median**, not the mean.
2. **No population peak can be claimed.** No dumbbell can be born within the final six
   frames, so the terminal decline in the live-population curve is structural. The
   population is still rising when the observable window closes.
3. **Full-frame detection has no ground truth.** The only hand annotation exists in a
   cropped region of movie 1053 and validates the **tracker** (0.57 nm localisation error,
   0.33 nm separation error, no identity swaps over 60 frames) — not the detector. Detection
   is validated by procedure: stationarity, physical particle size, and the drift retest.
4. **Counts do not transfer between movies.** Detection counts, accepted counts and
   population magnitudes depend on the field of view and the detector. Only the lifetime
   *distribution* is meant to be compared across acquisitions.
5. **The FDR pass is redundant by construction.** It runs on the same circular-shift null
   the acceptance gate already uses, so it is an internal consistency check, not
   independent statistical confirmation.

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
> shipped corresponds to the default values of our experiment (movie 1053).**

There is one engine and one parameter file. The engine reads `constants.txt` at startup
and contains no hardcoded values, so a new movie needs no code change — only a new set of
constants. Each `results/exp_*/` folder keeps a copy of the constants that produced it;
copy one over `tracking_algo/constants.txt` to reproduce that run.

Editing `constants.txt`: the format is `KEY = value`, with `#` starting a comment. A
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

- `population_dumbbell_dynamics_<TAG>.png` — the three-panel overview
- `lifetime_distribution_<TAG>.pdf` / `.png` — standalone panel (a), publication quality
- `live_population_<TAG>.pdf` / `.png` — standalone panel (c), publication quality
- `population_curve_<TAG>.npz` — births/deaths arrays, consumed by the drift retest

Move the contents of `Figs/` into the matching `results/exp_<TAG>/` folder to archive a
run, along with the `constants.txt` that produced it.

The engine prints its frame-0 detection count early as a sanity check. A few hundred to a
few thousand is normal; below ~100 or above ~5000 means the detector settings are wrong
and it is worth stopping there rather than after a long run.

The PDFs are vector and are the versions to use for publication; the 600-dpi PNGs are for
drafts and slides.

## Requirements

Python 3.9+, with `numpy`, `pandas`, `matplotlib`, `h5py`, `scikit-image` and `scipy`.
