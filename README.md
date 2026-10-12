# Pt–PVP–Pt Dumbbell Detection and Tracking

Automated detection, tracking and lifetime analysis of transient Pt–PVP–Pt dumbbell
intermediates in liquid-phase scanning transmission electron microscopy (LP-STEM) movies
of Pt nanoparticle micelle disassembly.

Supporting code for the manuscript "Resolving Dynamic Interparticle Interactions through Metastable Pt–PVP–Pt Dumbbells during Micelle Disassembly", Figures 3c/3d and Discussion S2.

---

## About the workflow

Each LP-STEM acquisition (`.dm4` stack with its Velox `.txt` metadata) is converted into two quantitative
measurements, *i.e.*, the distribution of **bonded lifetimes** of individual dumbbells, and the **live dumbbell population** as a function of time.

```
.dm4 stack + .txt
   │
   ├─ A. Ingestion        read pixel size and frame time from the acquisition's own
   │                      .txt metadata; flag corrupt frames (blank, saturated,
   │                      intensity jumps) so they become time gaps
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
   │                      (drift correction → distance-penalised peak → centroid →
   │                      step clamp → re-anchoring to the nearest detection)
   │
   ├─ G. Gating           physical acceptance criteria (below)
   │
   └─ H. Output           lifetime distribution and live population
```

### Brief description of each step

**1) Normalization:** The nanoparticles brighten substantially over the course of the data acquisition process.
Thus, the min–max normalization scheme (per-frame) would provide a fixed detection threshold, making
the detector itself drift and leading to false dumbbell population estimates. The percentile clipping approach
ensures that sensitivity remains constant across all frames.

**2) Denoising:** The observed noise in liquid-cell water-scatter frames is spatially correlated.
Some classical denoisers (*e.g.*, NL-means, wavelet, total-variation) and self-supervised models (_e.g._, N2V) that assume
pixel-independent noise inject spurious intensity maxima, which the tracker then follows instead of real
particles. We then use a light Gaussian filter to avoid this issue. The detection threshold is chosen as the
lowest value for which the detection yield is stationary across the movie and the detected particle diameter
is consistent with the known ~2.4 nm Pt nanoparticle size.

**3) Tracking:** Each particle of a pair is propagated frame by frame after a global drift correction, with
the partner masked so that two tracks cannot collapse onto one particle, and is re-anchored to the nearest
detection whenever one lies within 1.5 nm. The tracker was validated against hand-annotated
crops of the same acquisitions (exps 1051 and 1053), with sub-nanometer localization error.

**4) Velocity correlation:** Two unrelated particles carried by the same fluid
flow show high positional correlation regardless of any physical association.
Differencing (*i.e.*, taking velocity correlation instead) removes this common drift, so co-motion is
evaluated based on velocities.

**5) Per-frame separation floor.** Two ~2.4 nm particles cannot approach more closely than
their own diameter. We consider pairs that do so as part of trajectories that converged to the same particle
(and are ultimately discarded). The floor is set at 2.0 nm, which is the lower bound corresponding to
the contact distance with a 0.4 nm tolerance for localization/annotation error.

---

## Core principle

> **Physical criteria are constant (in nanometers) across different movies. Every pixel-space parameter is
> re-derived per movie.**

Pixel size, particle density, contrast and motion differ between acquisitions, so the
detector is re-fitted for each movie. However, the criteria to define a dumbbell do not change. 
The specific values have been established from the benchmark analysis against the hand-annotated cropped frames.

| criterion                                 | value                |
| ----------------------------------------- | -------------------- |
| birth (mutual-NN separation at formation) | 2.0 – 6.5 nm         |
| death (dissociation)                      | first frame with separation ≥ 6.5 nm |
| minimum bonded duration                   | 4 consecutive frames |
| bonded-phase mean separation              | 2.0 – 6.5 nm         |
| hard per-frame separation floor           | ≥ 2.0 nm             |
| velocity correlation                      | ≥ 0.30               |
| duplicate rejection                       | velocity correlation ≥ 0.90 at < 2.6 nm (two tracks on one particle) |

`constants_used.txt` is split along exactly this line. The `DETECTOR` block (normalization
percentiles, gaussian σ, blob scale, threshold) and the `TRACKER` block (`MAX_STEP`, excluded
frames) are re-derived for every new acquisition by the engine itself (the specific numbers are not transferable). 
`PHYSICAL CRITERIA` is the table above and should not be retuned per movie,since it would lead to results being not 
directly comparable between movies.

---

## Results

| movie | magnification | pixel size (nm) | frame time (s) | accepted (dissociated) | mean LT (s) | **median LT (s)** |
| ----- | ------------- | --------------- | -------------- | ---------------------- | ----------- | ----------------- |
| 1051  | 601 kx        | 0.1626          | 0.797          | 139 (49)               | 8.6         | **7.2**           |
| 1053  | 601 kx        | 0.1626          | 0.797          | 281 (171)              | 10.4        | **8.0**           |
| 1102  | 601 kx        | 0.1626          | 0.797          | 35 (8)                 | 9.2         | **8.0**           |

The median bonded lifetime is the most relevant result. It is computed per-dumbbell and does not
depend on how many particles are detected. From the above table, we can see it agrees within ~0.8 s
across three independent acquisitions, and it was stable within ~1 s across every detector and
gate configuration tried on movie 1053.

Two further acquisitions (1110 and 1117) were recorded with different camera settings. In particular, 1117 had
frames twice as long (1.57 s) and 1110 had higher magnification (850 kx, 0.115 nm/px) with shorter,
noisier frames. In such cases, the particle pairs are sampled differently in time and most of them do not
dissociate within the recording time. For such reasons, their lifetimes are not directly comparable with the three movies
above and are not reported here.

---

## Important remarks

1. **Lifetimes are left-truncated and right-censored.** The four-frame minimum means no
lifetime shorter than `4 × frame time` (~3.2 s) can be measured, and dumbbells that are still
bonded at the end of the recording (39–77 % of the accepted pairs, depending on the movie) are
right-censored, _i.e._, dissociations cannot be reported over the last 4 frames. Hence why we
focus on reporting the life-time median of the dissociated dumbbells,
rather than the mean. The `summary.csv` file also contains the Kaplan–Meier median, which includes the censored pairs.
2. **Dumbbell counts do not transfer between movies.** Detection counts, accepted counts and
population magnitudes depend on the field of view and the detector. Only the lifetime
distribution is meant to be compared across acquisitions.
3. **The live population rises throughout the recordings.** No births can be observed in the last
four frames, so the terminal shape of the curve is a boundary effect. One should interpret the curves
as "increasing over the observation", not as having a peak necessarily.

---

## Repository layout

```
pt-dumbbell-tracking/
├── README.md
├── data/                     pointer to the raw .dm4 stacks and .txt metadata on Zenodo 
│   └── README.md
├── tracking_algo/
│   ├── fullframe_scan.py                 triage of a new movie 
│   ├── dumbbell_engine_full_final.py     the engine (parameters in its CONFIG block)
│   ├── fullframe_plot_final.py           publication figures from a run
│   └── top3_dumbbells_anim_final.py      animation of the three longest-lived dumbbells
└── results/
    ├── exp_1051/
    ├── exp_1053/
    └── exp_1102/
        ├── constants_used.txt          the exact settings (given and derived) that produced this run
        ├── summary.csv                 lifetime statistics, population curves, every accepted lifetime
        ├── engine_dumbbells.csv        one row per accepted dumbbell
        ├── lifetime_distribution.png / live_population.png / mean_separation.png
        └── top3_dumbbells.mp4
```
> **The engine corresponds to the configuration of our experiments and only `FOLDER`
> needs to be changed between movies.** Every per-movie quantity (pixel size, frame time, detector
> threshold, `MAX_STEP`, excluded frames, death distance) is derived from the movie at run time and
> written to `Figs/constants_used_<id>.txt`, which is the file archived in `results/exp_<id>/`.

---

## Running it

```
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib scipy scikit-image ncempy

# download the .dm4 stacks and their .txt metadata from Zenodo into images_<id>_full/ (see data/README.md)
cd tracking_algo

python fullframe_scan.py                  # one QC / detector verdict per movie, writes Figs/frame_qc_<id>.csv

# set FOLDER = "images_<id>_full" in dumbbell_engine_full_final.py, then:
python dumbbell_engine_full_final.py

python fullframe_plot_final.py            # figures for the movies listed in MOVIES
python top3_dumbbells_anim_final.py       # animations (needs ffmpeg for .mp4; falls back to .gif)
```

Each engine run prints a summary table and writes into `Figs/`:

- `engine_<id>-full_final.png` (the four-panel overview)
- `summary_<id>.csv` (statistics, population curves and every accepted lifetime)
- `engine_dumbbells_<id>-full.csv`, `engine_tracks_<id>-full.csv` (accepted dumbbells and their tracks)
- `engine_candidates_<id>-full_final.csv` (every candidate pair with its outcome and reason)
- `death_pool_<id>-full.png` (the separation pool the dissociation distance is checked against)
- `constants_used_<id>.txt`

Move the contents of `Figs/` into the matching `results/exp_<id>/` folder to archive a run.

## Requirements

Python 3.9+, with `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-image` and `ncempy`
(`ffmpeg` on the path for the animations).
