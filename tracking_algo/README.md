# tracking_algo

Four scripts, run in this order from this folder. All of them look for the acquisitions in
`images_<id>_full/` (one `.dm4` stack and its Velox `.txt` per folder) and write to `Figs/`.

| script | what it does | reads | writes |
| --- | --- | --- | --- |
| `fullframe_scan.py` | Triage of every movie before any tracking: reads pixel size and frame time from the `.txt` (and cross-checks the `.dm4` tag), flags corrupt frames (blank, saturated, intensity jumps, low similarity to the previous frame), measures frame-to-frame drift, runs the detector threshold rule on a subsample of frames, and prints a GOOD / MARGINAL / REJECT verdict with reasons. | `images_*_full/` | `Figs/frame_qc_<id>.csv` (the exclusion list the engine uses), `Figs/fullframe_scan_<id>.png`, `Figs/fullframe_scan_summary.csv` |
| `dumbbell_engine_full_final.py` | **The engine.** Set `FOLDER = "images_<id>_full"` in its CONFIG block and run. Normalization, Gaussian denoising, blob detection with a per-movie threshold, mutual-nearest-neighbour pairing, drift-corrected two-particle tracking with re-anchoring to detections, the physical acceptance criteria, and the lifetime / population statistics. Every per-movie quantity is derived at run time and recorded. | `images_<id>_full/`, `Figs/frame_qc_<id>.csv` | `Figs/summary_<id>.csv`, `Figs/constants_used_<id>.txt`, `Figs/engine_dumbbells_<id>-full.csv`, `Figs/engine_tracks_<id>-full.csv`, `Figs/engine_candidates_<id>-full_final.csv`, `Figs/engine_<id>-full_final.png`, `Figs/death_pool_<id>-full.png` |
| `fullframe_plot_final.py` | Publication figures for the movies listed in `MOVIES`: lifetime distribution of the dissociated dumbbells (mean and median), live dumbbell population vs time, and the distribution of mean bonded separation (p25 / p50 / p75). 600 dpi, PNG + PDF, no titles. | `Figs/summary_<id>.csv`, `Figs/engine_dumbbells_<id>-full.csv` | `Figs/lifetime_distribution_<id>`, `Figs/live_population_<id>`, `Figs/mean_separation_<id>` |
| `top3_dumbbells_anim_final.py` | Animation of the three longest-lived dissociated dumbbells of each movie: tracking overlaid on the raw and denoised frames, and separation vs time with the dissociation line. Uses the tracks saved by the engine, so nothing is re-tracked. Needs `ffmpeg` for `.mp4` (falls back to `.gif`). | `images_<id>_full/`, `Figs/engine_dumbbells_<id>-full.csv`, `Figs/engine_tracks_<id>-full.csv` | `Figs/top3_dumbbells_<id>.mp4`, `Figs/top3_dumbbells_final_<id>.png` |

Parameters live in the CONFIG block at the top of `dumbbell_engine_full_final.py`. The physical
criteria (in nanometers) are shared across movies and should not be retuned per acquisition. The
detector threshold, `MAX_STEP` and the excluded frames are measured from each movie and written to
`Figs/constants_used_<id>.txt`, which is the file archived in `../results/exp_<id>/`.

To archive a run, move the contents of `Figs/` into `../results/exp_<id>/`.
