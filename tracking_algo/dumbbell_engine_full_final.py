# ============================================================
#  DUMBBELL POPULATION ENGINE  —  FULL-FRAME (dm4), FINAL CONFIGURATION, GT-free
#
#  One configuration applied identically to every movie (chosen on the 1053 gate grid, 2026-10):
#     birth     mutual nearest neighbours between FLOOR 2.0 nm and DEATH 6.5 nm, neither particle in a live dumbbell
#     death     the FIRST frame with separation >= 6.5 nm (no sustain, no "both found" requirement)
#     accept    bonded >= 4 frames | mean separation in [2.0, 6.5] nm | per-frame floor 2.0 nm
#               | whole-bond velocity correlation >= 0.3 | duplicate rejected if vc >= 0.9 at < 2.6 nm
#     no isolation gate, no two-peak gate, no crop registration, no scorecard.
#  Everything upstream is GT-free and re-derived per movie: pixel size / frame time from the Velox
#  .txt, frame QC (Figs/frame_qc_<id>.csv from fullframe_scan.py), detector threshold from
#  stationarity + physical blob size, drift correction, MAX_STEP from the motion of detections,
#  merge-aware + detection-assisted tracking, separation pool printed (6.5 nm kept unless bimodal).
#  Outputs: Figs/summary_<id>.csv (stats + curves + lifetimes), engine_dumbbells/tracks/candidates CSVs, figure.
# ============================================================
import os, re, glob, json, base64, io, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation

# ------------------------------------------------------------------ CONFIG (physics in nm; everything else measured)
FOLDER          = "images_1053_full"        # images_<id>_full with <name>.dm4 + <name>.txt  (or a crop folder with LOAD_MODE="json")
LOAD_MODE       = "dm4"                     # "dm4" | "json" (annotation crops) | "emd"
CROP_FOLDER     = None                      # FINAL: no crop registration, no scorecard
CROP_FRAME_OFFSET = None                    # dm4 frame = crop frame + offset; None -> found by registration over OFFSET_SEARCH
OFFSET_SEARCH   = range(-6, 7)
REG_NCC_MIN, REG_RATIO_MIN = 0.35, 1.3      # registration accepted if the NCC peak >= REG_NCC_MIN and >= REG_RATIO_MIN x the 2nd-best offset's peak
QC_CSV          = None                      # Figs/frame_qc_<id>.csv from fullframe_scan.py; None -> auto-locate, else recompute
SWEEP_EVERY     = 4
EMD_PATH, EMD_DATASET = None, None
PX_NM_OVERRIDE, DT_OVERRIDE = None, None
MUTUAL_MIN      = 0.0                       # FINAL: isolation gate OFF (never rejects); fraction still logged
CONF_ISO        = 2.0                       # annulus-CNR a detection needs to count as a real neighbour in the isolation test
VALLEY_SIG      = 0.5                       # intensity dip between the two centres (x sigma_bg) required for "two distinct peaks"
TWOPEAK_MIN     = 0.0                       # set per mode below
SEED_VALLEY     = False                     # set per mode below
GATE_MODES      = {"final": dict(TWOPEAK_MIN=0.0, SEED_VALLEY=False)}   # single configuration
PRIMARY_MODE    = "final"
DUP_VC, DUP_SEP_NM = 0.9, 2.6               # duplicate-detection signature: two tracks on one particle co-move at vc~1 at < 2.6 nm
UNCOLLAPSE_MAX  = 3                         # consecutive frames inside contact distance -> the candidate is a merge
SNAP_NM         = 1.5                       # re-anchor a track to a detection within this distance (detection-assisted tracking)
PARTICLE_DIAM_NM = 2.4;  CONTACT_TOL_NM = 0.4;  MERGE_NM = PARTICLE_DIAM_NM - CONTACT_TOL_NM     # 2.0 nm
DEATH_PRIOR_NM  = 6.5                       # the working death distance; overridden only by a genuinely bimodal separation pool
VALLEY_MAX      = 0.5                       # density at the Otsu split must be below this x the smaller mode's peak to override
# detector (blob_log), all in nm; threshold as a fraction of each frame's p99 contrast
DET_GSIG_NM     = 0.50                      # denoise sigma before detection
DET_MIN_DIAM_NM, DET_MAX_DIAM_NM = 1.6, 4.0 # blob scale range (physical 2.4 nm +- annotation/blur slack)
THR_GRID        = [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30, 0.35]
THR_FRAC        = None                      # None -> GT-FREE choice: lowest stationary threshold with a physical blob size
YIELD_CV_MAX    = 0.35                      # per-frame yield CV allowed for a "stationary" detector
SIZE_TOL        = 0.15                      # |mean deconvolved blob diam - PARTICLE_DIAM| / PARTICLE_DIAM allowed (noise storms read 25-35% too small)
PERSIST_FRAMES  = 6                         # a mutual-NN pair must persist this many consecutive frames to enter the death pool
DET_MATCH_NM    = 1.5                       # a detection within this distance of a GT mark is a true positive
# tracker (nm; converted with PX_NM)
TRK_GSIG_NM, R_NM, RR_NM, LAM = 0.50, 0.98, 1.63, 0.003
CONF_ABS, FADE_FRAC, BASE_K   = 1.0, 0.5, 5
MOTION_K, STEP_K, COAST_RELAX = 2.5, 5, 5
DRIFT_BP, DRIFT_UPS, DRIFT_MAX_FRAC = (2.0, 12.0), 10, 0.35
# dumbbell criteria
MIN_BONDED      = 4                         # frames; the minimum for which vc (>=3 steps) and the death call are defined
SUSTAIN         = 1                         # FINAL: death = the FIRST frame >= DEATH_NM
VC_MIN_BOND     = 0.3                       # whole-bond true-velocity correlation
POOL_MIN_FRAC, CP_MIN_GAP_NM = 0.10, 1.5    # pooled-split validity
# GT matching (scorecard only)
GT_MATCH_NM     = 1.5;  GT_MATCH_FRAC = 0.5 # engine pair <-> GT pair: both particles within GT_MATCH_NM in >= this fraction of overlap
PAIR_MIN_FRAMES, PAIR_MIN_MUTUAL = 6, 0.5   # GT pair discovery (same rule as the validation script)
os.makedirs("Figs", exist_ok=True)

# ------------------------------------------------------------------ LOAD  (annotation OPTIONAL: scorecard only)
csvs = glob.glob(os.path.join(FOLDER, "PP_tracks_*.csv")) if LOAD_MODE == "json" else []
HAS_GT = bool(csvs)
EXCLUDED = []
def _parse_txt(path):
    txt = open(path, errors="ignore").read(); out = {}
    for key, name in (("Pixel size", "px"), ("Frame time", "dt")):
        r = re.search(key + r"\s*\t\s*([\-\d.]+)\s*([a-zA-Zµ]+)", txt)
        if r: out[name] = float(r.group(1)) * {"pm": 1e-3, "nm": 1.0, "µm": 1e3, "ms": 1e-3, "s": 1.0, "µs": 1e-6}.get(r.group(2), np.nan)
    return out
if LOAD_MODE == "dm4":
    from ncempy.io import dm
    dm4 = glob.glob(os.path.join(FOLDER, "*.dm4")); assert dm4, f"no .dm4 in {FOLDER}"
    txts = glob.glob(os.path.join(FOLDER, "*.txt")); meta = _parse_txt(txts[0]) if txts else {}
    d = dm.dmReader(dm4[0]); arr = np.asarray(d["data"]).astype(np.float32)
    if arr.ndim == 2: arr = arr[None]
    if arr.shape[0] > arr.shape[1] and arr.shape[0] > arr.shape[2]: arr = np.transpose(arr, (2, 0, 1))
    PX_NM = PX_NM_OVERRIDE or meta.get("px"); DT = DT_OVERRIDE or meta.get("dt"); assert PX_NM and DT, "pixel size / frame time: set the overrides or provide the .txt"
    MID = (re.findall(r"images_(\w+)_full", FOLDER) or [os.path.basename(os.path.normpath(FOLDER))])[0]; TAG = MID + "-full"
    raw_all = {i: arr[i] for i in range(arr.shape[0])}; T0 = 0.0; gt = {}
    # ---- frame QC: use the scan's list if present, else recompute the same rules
    qc = QC_CSV or (f"Figs/frame_qc_{MID}.csv" if os.path.exists(f"Figs/frame_qc_{MID}.csv") else None)
    if qc:
        Q = pd.read_csv(qc); EXCLUDED = [int(f) for f in Q.frame[Q.excluded.astype(bool)]]
    else:
        med = np.array([np.median(raw_all[i]) for i in raw_all]); run = pd.Series(med).rolling(5, center=True, min_periods=1).median().values
        mad = 1.4826 * np.median(np.abs(med - run)) + 1e-9; mx = float(arr.max())
        EXCLUDED = [i for i in raw_all if (raw_all[i] <= 0).mean() > 0.005 or (raw_all[i] >= mx).mean() > 0.01 or abs(med[i] - run[i]) > max(4 * mad, 0.02 * run[i])]
    if len(EXCLUDED) > 0.5 * len(raw_all): print(f"[{TAG}] !! QC would exclude {len(EXCLUDED)}/{len(raw_all)} frames - ignoring the QC list; check frame_qc_{MID}.csv"); EXCLUDED = []
    raw = {i: raw_all[i] for i in raw_all if i not in EXCLUDED}
    tsec = {i: i * DT for i in raw}
    print(f"[{TAG}] dm4 {os.path.basename(dm4[0])}: {arr.shape[0]} frames, excluded {EXCLUDED} ({'from ' + qc if qc else 'recomputed'})")
elif LOAD_MODE == "emd":
    import h5py
    with h5py.File(EMD_PATH, "r") as h:
        arr = np.transpose(h[EMD_DATASET][()], (2, 0, 1)).astype(np.float32)          # (F, H, W): frame axis is last in Velox
        meta_ds = EMD_DATASET.rsplit("/", 1)[0] + "/Metadata"
        PX_NM = PX_NM_OVERRIDE; DT = DT_OVERRIDE
        if meta_ds in h and (PX_NM is None or DT is None):
            try:
                m = json.loads(bytes(h[meta_ds][:, 0]).decode("utf-8", "ignore").rstrip("\x00"))
                if PX_NM is None: PX_NM = float(m["BinaryResult"]["PixelSize"]["width"]) * 1e9
                if DT is None: DT = float(m["Scan"]["FrameTime"])
            except Exception as e: print("metadata parse failed:", e)
        assert PX_NM and DT, "set PX_NM_OVERRIDE / DT_OVERRIDE"
    TAG = os.path.splitext(os.path.basename(EMD_PATH))[0]; raw = {i: arr[i] for i in range(arr.shape[0])}; T0 = 0.0
    tsec = {i: i * DT for i in raw}; gt = {}
else:
    json_paths = sorted(set(sum([glob.glob(os.path.join(FOLDER, s_, "frame_*.json")) for s_ in ("train", "val", "")], []))); assert json_paths
    def _fidx(p): return int(re.findall(r"\d+", os.path.basename(p))[-1])
    raw = {}
    for p in json_paths:
        fi = _fidx(p)
        if fi in raw: continue
        d = json.load(open(p)); img = Image.open(io.BytesIO(base64.b64decode(d["imageData"]))) if d.get("imageData") else Image.open(os.path.join(os.path.dirname(p), d["imagePath"]))
        raw[fi] = np.asarray(img.convert("L"), dtype=np.float32)
    if HAS_GT:
        CSV = csvs[0]; _b = os.path.basename(CSV)
        TAG = (re.findall(r"10\d\d", _b) or ["?"])[0] + "-" + ((re.findall(r"ROI\d*", _b) or re.findall(r"PtDum", _b) or [os.path.basename(os.path.normpath(FOLDER))])[-1])
        gt_df = pd.read_csv(CSV); sel = gt_df.x > 0
        PX_NM = PX_NM_OVERRIDE or float(np.median(gt_df.loc[sel, "x_nm"].values / gt_df.loc[sel, "x"].values))
        _ts = gt_df.drop_duplicates("frame").sort_values("frame").t_s.values; DT = DT_OVERRIDE or float(np.median(np.diff(_ts))); T0 = float(_ts.min())
        gt = {int(f): {int(r.particle): (float(r.x), float(r.y)) for _, r in g.iterrows()} for f, g in gt_df.groupby("frame")}
        tsec = {int(f): float(g.t_s.iloc[0]) for f, g in gt_df.groupby("frame")}
    else:
        TAG = os.path.basename(os.path.normpath(FOLDER)); PX_NM, DT = PX_NM_OVERRIDE, DT_OVERRIDE; assert PX_NM and DT, "no annotation: set PX_NM_OVERRIDE / DT_OVERRIDE"
        gt = {}; T0 = 0.0; tsec = {f: f * DT for f in raw}
frames = sorted(raw); F_ = len(frames); Hh, Ww = raw[frames[0]].shape; YY, XX = np.mgrid[0:Hh, 0:Ww]
for f in frames: tsec.setdefault(f, f * DT)
# ---- optional: register an annotation crop into the full frame -> GT in full-frame coordinates (scorecard only)
if LOAD_MODE == "dm4" and CROP_FOLDER:
    from skimage.feature import match_template
    ccsv = glob.glob(os.path.join(CROP_FOLDER, "PP_tracks_*.csv")); cj = sorted(set(sum([glob.glob(os.path.join(CROP_FOLDER, s_, "frame_*.json")) for s_ in ("train", "val", "")], [])))
    if ccsv and cj:
        cdf = pd.read_csv(ccsv[0]); cpx = float(np.median(cdf.loc[cdf.x > 0, "x_nm"].values / cdf.loc[cdf.x > 0, "x"].values))
        if abs(cpx - PX_NM) / PX_NM > 0.03: print(f"[{TAG}] crop pixel size {cpx:.4f} != frame {PX_NM:.4f} nm: registration needs rescaling, skipped"); CROP_FOLDER = None
    if CROP_FOLDER and ccsv and cj:
        def _cidx(p): return int(re.findall(r"\d+", os.path.basename(p))[-1])
        cimgs = {}
        for p in cj:
            k = _cidx(p); dd = json.load(open(p)); cimgs[k] = np.asarray(Image.open(io.BytesIO(base64.b64decode(dd["imageData"]))).convert("L"), dtype=np.float32)
        def _prep(a): return gaussian_filter((a - a.mean()) / (a.std() + 1e-9), 1.5)
        best = []; _pre = {fi: _prep(raw[fi]) for fi in list(raw)[:max(12, len(OFFSET_SEARCH) + 8)]}
        for k0 in sorted(cimgs)[:8]:                                               # several early crop frames as templates (the first may predate the substack)
            tpl = _prep(cimgs[k0])
            for off in (OFFSET_SEARCH if CROP_FRAME_OFFSET is None else [CROP_FRAME_OFFSET]):
                fi = k0 + off
                if fi not in _pre: continue
                cc = match_template(_pre[fi], tpl, pad_input=False); pk = float(cc.max()); yx = np.unravel_index(int(cc.argmax()), cc.shape)
                best.append((pk, off, int(yx[1]), int(yx[0]), k0))
        best.sort(reverse=True)
        best = [b for i, b in enumerate(best) if all(b[1] != c[1] for c in best[:i])]   # keep the best score per offset
        if best and best[0][0] >= REG_NCC_MIN and (len(best) == 1 or best[0][0] >= REG_RATIO_MIN * best[1][0]):
            pk, off, ox, oy, k0 = best[0]; HAS_GT = True
            gt = {}; tsec_c = {}
            for f, g in cdf.groupby("frame"):
                fi = int(f) + off
                if fi in raw: gt[fi] = {int(r.particle): (float(r.x) + ox, float(r.y) + oy) for _, r in g.iterrows()}
            gt_df = cdf.copy(); gt_df["x"] += ox; gt_df["y"] += oy; gt_df["frame"] = gt_df["frame"] + off; gt_df = gt_df[gt_df.frame.isin(list(raw))]
            print(f"[{TAG}] crop REGISTERED: offset ({ox},{oy}) px, dm4 frame = crop frame + {off} (template: crop frame {k0}), NCC peak {pk:.2f}" + (f" (2nd-best offset {best[1][0]:.2f})" if len(best) > 1 else "") + f" -> {len(gt)} GT frames in full-frame coordinates")
        else:
            print(f"[{TAG}] crop registration NOT accepted (best NCC {best[0][0]:.2f} at offset {best[0][1]}" + (f", 2nd {best[1][0]:.2f}" if len(best) > 1 else "") + ") -> GT-free run"); HAS_GT = False
print(f"[{TAG}] {F_} frames {frames[0]}..{frames[-1]} ({Hh}x{Ww}), PX_NM={PX_NM:.4f}, DT={DT:.3f}s | MERGE/FLOOR {MERGE_NM} nm | annotation: {'yes (scorecard only)' if HAS_GT else 'NONE (method is GT-free)'}")
def norm_pclip(a): lo, hi = np.percentile(a, [1, 99.9]); return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
normed = {f: norm_pclip(raw[f]) for f in frames}
det_stack = {f: gaussian_filter(normed[f], DET_GSIG_NM / PX_NM) for f in frames}
stack = {f: gaussian_filter(normed[f], TRK_GSIG_NM / PX_NM) for f in frames}
RAD = 0.5 * PARTICLE_DIAM_NM / PX_NM; Rpx, RRpx = max(1, round(R_NM / PX_NM)), max(2, round(RR_NM / PX_NM))

# ------------------------------------------------------------------ DRIFT (GT-free)
_bp = {f: gaussian_filter(normed[f], DRIFT_BP[0]) - gaussian_filter(normed[f], DRIFT_BP[1]) for f in frames}
drift = {frames[0]: (0.0, 0.0)}
for i in range(1, F_):
    sh, _, _ = phase_cross_correlation(_bp[frames[i - 1]], _bp[frames[i]], upsample_factor=DRIFT_UPS, normalization=None)
    dx, dy = -float(sh[1]), -float(sh[0]); drift[frames[i]] = (0.0, 0.0) if np.hypot(dx, dy) > DRIFT_MAX_FRAC * min(Hh, Ww) else (dx, dy)
_dmag = [np.hypot(*drift[f]) for f in frames[1:]]
print(f"[{TAG}] drift: max {max(_dmag):.1f}px ({max(_dmag)*PX_NM:.1f} nm) at frame {frames[1:][int(np.argmax(_dmag))]}, median {np.median(_dmag):.1f}px")

# ------------------------------------------------------------------ DETECTOR + GT precision/recall sweep
MIN_SIG = (DET_MIN_DIAM_NM / 2) / np.sqrt(2) / PX_NM; MAX_SIG = (DET_MAX_DIAM_NM / 2) / np.sqrt(2) / PX_NM
_det_cache = {}
def detect_full(f, thr_frac):
    k = (f, round(thr_frac, 4))
    if k not in _det_cache:
        fr = det_stack[f]; th = thr_frac * float(np.percentile(fr, 99))
        _det_cache[k] = [(float(x), float(y), float(sg)) for y, x, sg in blob_log(fr, min_sigma=MIN_SIG, max_sigma=MAX_SIG, num_sigma=5, threshold=th)]
    return _det_cache[k]
def detect(f, thr_frac): return [(x, y) for x, y, _ in detect_full(f, thr_frac)]
def blob_diam_nm(f, thr_frac):
    """mean deconvolved blob diameter: sigma_true^2 = sigma_det^2 - sigma_blur^2 ; diam = 2*sqrt(2)*sigma_true"""
    sg = np.array([s_ for _, _, s_ in detect_full(f, thr_frac)]); blur = DET_GSIG_NM / PX_NM
    if not len(sg): return np.nan
    st = np.sqrt(np.clip(sg ** 2 - blur ** 2, 0, None)); return float(np.mean(2 * np.sqrt(2) * st) * PX_NM)
def pr_at(thr):
    tp = fp = fn = 0; yields = []
    for f in (frames[::SWEEP_EVERY] if LOAD_MODE == "dm4" else frames):
        D = detect(f, thr); yields.append(len(D)); G = list(gt.get(f, {}).values())
        if not G: fp += len(D) if HAS_GT else 0; continue
        if not D: fn += len(G); continue
        Dm = np.hypot(np.array(D)[:, None, 0] - np.array(G)[None, :, 0], np.array(D)[:, None, 1] - np.array(G)[None, :, 1]) * PX_NM
        used = set(); hit = 0
        for gi in np.argsort(Dm.min(0)):                               # greedy one-to-one matching
            cand = [(Dm[di, gi], di) for di in range(len(D)) if di not in used and Dm[di, gi] <= DET_MATCH_NM]
            if cand: used.add(min(cand)[1]); hit += 1
        tp += hit; fn += len(G) - hit; fp += len(D) - hit
    P = tp / max(tp + fp, 1) if HAS_GT else np.nan; Rc = tp / max(tp + fn, 1) if HAS_GT else np.nan; F1 = 2 * P * Rc / max(P + Rc, 1e-9) if HAS_GT else np.nan
    diam = float(np.nanmean([blob_diam_nm(f, thr) for f in frames]))
    return dict(thr_frac=thr, recall=Rc, precision_lower_bound=P, f1=F1, tp=tp, fp=fp, fn=fn, yield_mean=float(np.mean(yields)),
                yield_cv=float(np.std(yields) / max(np.mean(yields), 1e-9)), blob_diam_nm=diam,
                stationary=float(np.std(yields) / max(np.mean(yields), 1e-9)) <= YIELD_CV_MAX, size_ok=abs(diam - PARTICLE_DIAM_NM) / PARTICLE_DIAM_NM <= SIZE_TOL)
det_sweep = pd.DataFrame([pr_at(t) for t in THR_GRID]); det_sweep.to_csv(f"Figs/detector_sweep_{TAG}.csv", index=False)
_ok = det_sweep[det_sweep.stationary & det_sweep.size_ok]
if THR_FRAC is None:
    if len(_ok): THR_FRAC = float(_ok.thr_frac.min()); THR_SRC = "GT-free: lowest stationary threshold with blob diam within +-15% of 2.4 nm"
    else:
        _st = det_sweep[det_sweep.stationary] if det_sweep.stationary.any() else det_sweep
        THR_FRAC = float(_st.iloc[(_st.blob_diam_nm - PARTICLE_DIAM_NM).abs().argmin()].thr_frac); THR_SRC = "GT-free fallback: stationary threshold with blob diam closest to 2.4 nm"
else: THR_SRC = "config"
_r = det_sweep[det_sweep.thr_frac == THR_FRAC].iloc[0]
print(f"[{TAG}] DETECTOR sweep (blob_log, diam {DET_MIN_DIAM_NM}-{DET_MAX_DIAM_NM} nm, sigma {DET_GSIG_NM} nm, match <= {DET_MATCH_NM} nm) -- precision is a LOWER BOUND (annotation is partial):")
print(det_sweep.round(3).to_string(index=False))
print(f"[{TAG}] DETECTOR operating point THR_FRAC={THR_FRAC} ({THR_SRC}): " + (f"GT recall {_r.recall:.2f} | " if HAS_GT else "") + f"yield {_r.yield_mean:.0f}/frame (CV {_r.yield_cv:.2f}) | blob diam {_r.blob_diam_nm:.2f} nm"
      + (f" | GT has {gt_df.groupby('frame').size().mean():.0f} marks/frame" if HAS_GT else "") + ("" if (not HAS_GT or _r.recall >= 0.9) else "   !! recall < 0.9: dim annotated particles are below threshold; expect late births"))
DET = {f: detect(f, THR_FRAC) for f in frames}

# ---- MAX_STEP measured tracker-free (mutual-NN assignment between consecutive detection sets, drift removed)
_steps = []
for i in range(1, F_):
    A, B = np.array(DET[frames[i - 1]]), np.array(DET[frames[i]])
    if len(A) < 2 or len(B) < 2: continue
    B2 = B - np.array(drift[frames[i]]); Dm = np.hypot(A[:, None, 0] - B2[None, :, 0], A[:, None, 1] - B2[None, :, 1])
    nnA = Dm.argmin(1); nnB = Dm.argmin(0); _steps += [Dm[a, nnA[a]] for a in range(len(A)) if nnB[nnA[a]] == a]
if len(_steps) >= 10:
    MAX_STEP = int(np.ceil(np.percentile(_steps, 95))); _mot = f"median {np.median(_steps):.1f}px p95 {np.percentile(_steps,95):.1f}px ({len(_steps)} NN steps)"
else:
    MAX_STEP = int(np.ceil(2.0 * PARTICLE_DIAM_NM / PX_NM)); _mot = f"too few NN steps ({len(_steps)}) -> fallback 2 particle diameters"
STEP_FLOOR = 0.5 * RAD; STEP_REF0 = max(float(np.median(_steps)) if len(_steps) >= 10 else MAX_STEP / 3, STEP_FLOOR)
print(f"[{TAG}] motion (tracker-free, from detections): {_mot} -> MAX_STEP={MAX_STEP}px")

# ------------------------------------------------------------------ pair helpers
def mutual_pairs(pts, lo_nm, hi_nm):
    if len(pts) < 2: return []
    A = np.array(pts); Dm = np.hypot(A[:, None, 0] - A[None, :, 0], A[:, None, 1] - A[None, :, 1]) * PX_NM; np.fill_diagonal(Dm, np.inf); nn = Dm.argmin(1)
    return [(i, nn[i], Dm[i, nn[i]]) for i in range(len(A)) if nn[nn[i]] == i and i < nn[i] and lo_nm <= Dm[i, nn[i]] <= hi_nm]
_DETC = {}
def det_conf(f):
    """detections with annulus-CNR >= CONF_ISO (the ones that count as real neighbours)"""
    if f not in _DETC:
        D = DET[f]; _DETC[f] = [d for d in D if conf_at(f, d[0], d[1]) >= CONF_ISO]
    return _DETC[f]
def mutual_among_detections(f, pa, pb):
    """isolated pair test: no CONFIDENT third detection is closer to a than b is, nor closer to b than a is.
    The pair's own particles are given (tracked positions); only intruders must be confident."""
    sep = np.hypot(pa[0] - pb[0], pa[1] - pb[1])
    D = np.array(det_conf(f), dtype=float)
    if len(D) == 0: return True
    da = np.hypot(D[:, 0] - pa[0], D[:, 1] - pa[1]); db = np.hypot(D[:, 0] - pb[0], D[:, 1] - pb[1])
    third = (da > DET_MATCH_NM / PX_NM) & (db > DET_MATCH_NM / PX_NM)                 # exclude the pair's own detections
    if not third.any(): return True
    return bool(da[third].min() > sep and db[third].min() > sep)
def pool_stats(pool, bin_nm=0.25):
    """Otsu split + valley test on a 1-D separation pool. Returns a dict (always) with 'override' True/False."""
    out = dict(n=len(pool))
    if len(pool) < 30 or np.ptp(pool) <= 0: out.update(override=False, reason="too few samples"); return out
    thr = float(threshold_otsu(pool)); lo, hi = pool[pool < thr], pool[pool >= thr]
    edges = np.arange(np.floor(pool.min()), np.ceil(pool.max()) + bin_nm, bin_nm); h, e = np.histogram(pool, edges); c = 0.5 * (e[1:] + e[:-1])
    hs = np.convolve(h, [0.25, 0.5, 0.25], mode="same")                                   # light smoothing
    peak_lo = hs[c < thr].max() if (c < thr).any() else 0; peak_hi = hs[c >= thr].max() if (c >= thr).any() else 0
    valley = hs[np.abs(c - thr) <= 0.5].min() if (np.abs(c - thr) <= 0.5).any() else hs.min()
    out.update(otsu=thr, frac_lo=len(lo) / len(pool), frac_hi=len(hi) / len(pool), mean_lo=float(lo.mean()) if len(lo) else np.nan, mean_hi=float(hi.mean()) if len(hi) else np.nan,
               gap=float(hi.mean() - lo.mean()) if len(lo) and len(hi) else np.nan, valley_ratio=float(valley / max(min(peak_lo, peak_hi), 1e-9)),
               p=np.percentile(pool, [5, 25, 50, 75, 95]), hist=(h, e))
    ok_modes = out["frac_lo"] >= POOL_MIN_FRAC and out["frac_hi"] >= POOL_MIN_FRAC; ok_gap = np.isfinite(out["gap"]) and out["gap"] >= CP_MIN_GAP_NM; ok_valley = out["valley_ratio"] <= VALLEY_MAX
    out["override"] = ok_modes and ok_gap and ok_valley
    out["reason"] = "bimodal: valley %.2f, gap %.2f nm, modes %.0f%%/%.0f%%" % (out["valley_ratio"], out["gap"], 100 * out["frac_lo"], 100 * out["frac_hi"]) if out["override"] else \
                    "unimodal or weak split: " + ", ".join(x for x, ok in (("valley %.2f > %.2f" % (out["valley_ratio"], VALLEY_MAX), not ok_valley), ("gap %.2f < %.1f nm" % (out["gap"], CP_MIN_GAP_NM), not ok_gap), ("mode fractions %.0f%%/%.0f%%" % (100 * out["frac_lo"], 100 * out["frac_hi"]), not ok_modes)) if ok)
    return out

# ------------------------------------------------------------------ LOCALISER + GATES (ported)
def sigma_bg(s): m = np.median(s); return 1.4826 * np.median(np.abs(s[s < m] - m)) + 1e-9
_sb = {f: sigma_bg(stack[f]) for f in frames}
def _wmask(x0, x1, y0, y1, excl):
    """window-local exclusion mask from a list of (x, y, rad) disks"""
    m = np.ones((y1 - y0, x1 - x0), bool)
    for (ex, ey, er) in (excl or []): m &= np.hypot(XX[y0:y1, x0:x1] - ex, YY[y0:y1, x0:x1] - ey) > er
    return m
def conf_at(f, cx, cy, rr=None, mask=None, excl=None):
    rr = RRpx if rr is None else rr; s = stack[f]
    x0, x1 = max(0, round(cx - 2 * rr)), min(Ww, round(cx + 2 * rr) + 1); y0, y1 = max(0, round(cy - 2 * rr)), min(Hh, round(cy + 2 * rr) + 1)
    if x1 <= x0 or y1 <= y0: return 0.0                                                   # position outside the frame
    p = s[y0:y1, x0:x1]; d = np.hypot(XX[y0:y1, x0:x1] - cx, YY[y0:y1, x0:x1] - cy); inn, ann = d <= rr, (d > rr) & (d <= 2 * rr)
    if excl: mw = _wmask(x0, x1, y0, y1, excl); inn &= mw; ann &= mw
    if mask is not None: inn &= mask[y0:y1, x0:x1]; ann &= mask[y0:y1, x0:x1]
    if inn.sum() < 3 or ann.sum() < 3: return 0.0
    return float((p[inn].mean() - p[ann].mean()) / _sb[f])
def refine(f, cx, cy, r=None, rr=None, lam=LAM, mask=None, excl=None):
    s = stack[f]; r = Rpx if r is None else r; rr = RRpx if rr is None else rr
    x0, x1 = max(0, round(cx - r)), min(Ww, round(cx + r) + 1); y0, y1 = max(0, round(cy - r)), min(Hh, round(cy + r) + 1)
    if x1 <= x0 or y1 <= y0: return float(np.clip(cx, 0, Ww - 1)), float(np.clip(cy, 0, Hh - 1))
    win = s[y0:y1, x0:x1].copy(); ys, xs = YY[y0:y1, x0:x1], XX[y0:y1, x0:x1]
    if excl: win[~_wmask(x0, x1, y0, y1, excl)] = -np.inf
    if mask is not None: win[~mask[y0:y1, x0:x1]] = -np.inf
    k = np.unravel_index(np.argmax(win - lam * ((xs - cx) ** 2 + (ys - cy) ** 2)), win.shape); px, py = x0 + k[1], y0 + k[0]
    a0, a1 = max(0, px - rr), min(Ww, px + rr + 1); b0, b1 = max(0, py - rr), min(Hh, py + rr + 1)
    patch = s[b0:b1, a0:a1]; m = np.ones(patch.shape, bool) if mask is None else mask[b0:b1, a0:a1].copy()
    if excl: m &= _wmask(a0, a1, b0, b1, excl)
    if m.sum() < 3: return float(px), float(py)
    w = np.where(m, patch - patch[m].min(), 0)
    if w.sum() <= 0: return float(px), float(py)
    return float((XX[b0:b1, a0:a1] * w).sum() / w.sum()), float((YY[b0:b1, a0:a1] * w).sum() / w.sum())
def cap_step(p, n, m):
    dx, dy = n[0] - p[0], n[1] - p[1]; d = np.hypot(dx, dy); return (n, False) if d <= m else ((p[0] + dx * m / d, p[1] + dy * m / d), True)
def disk_mask(centres, rad):
    m = np.ones((Hh, Ww), bool)
    for (x, y) in centres: m &= np.hypot(XX - x, YY - y) > rad
    return m

from scipy.ndimage import map_coordinates
def two_peaks(f, pa, pb, n=25):
    """is there an intensity valley between the two centres? (duplicate detections on one particle have none)"""
    xs = np.linspace(pa[0], pb[0], n); ys = np.linspace(pa[1], pb[1], n); prof = map_coordinates(stack[f], [ys, xs], order=1)
    ends = min(prof[:3].max(), prof[-3:].max()); return bool((ends - prof[3:-3].min()) / _sb[f] >= VALLEY_SIG)

class Track:
    """one particle: gated propagation with coasting; exclusion mask supplied per step"""
    def __init__(self, f, pos):
        self.pos = {f: pos}; self.conf = {f: conf_at(f, *pos)}; self.found = {f: True}
        self.base = [self.conf[f]]; self.steps = []; self.ncoast = 0
    def fthr(self): return max(CONF_ABS, FADE_FRAC * float(np.median(self.base)))
    def sthr(self): return MAX_STEP if self.ncoast >= COAST_RELAX else MOTION_K * max(float(np.median(self.steps)) if len(self.steps) >= 3 else STEP_REF0, STEP_FLOOR)
    def step(self, f_prev, f, mask=None, excl=None):
        p = self.pos[f_prev]; q = (float(np.clip(p[0] + drift[f][0], 0, Ww - 1)), float(np.clip(p[1] + drift[f][1], 0, Hh - 1)))
        n = refine(f, *q, excl=excl); n, _ = cap_step(q, n, MAX_STEP); c = conf_at(f, *n, excl=excl); st = np.hypot(n[0] - q[0], n[1] - q[1])
        if c < self.fthr() or st > self.sthr():                       # fade or implausible jump -> coast
            self.pos[f] = q; self.conf[f] = c; self.found[f] = False; self.ncoast += 1
        else:
            self.pos[f] = n; self.conf[f] = c; self.found[f] = True; self.ncoast = 0
            self.base = (self.base + [c])[-BASE_K:]; self.steps = (self.steps + [st])[-STEP_K:]

# ------------------------------------------------------------------ PAIR TRACKING: birth -> sustained death or censor
def snap_to_detection(f, pos, exclude_pos=None, exclude_idx=None):
    """nearest detection within SNAP_NM of pos (refined), not within MERGE_NM of exclude_pos and not exclude_idx.
    Returns (new_pos, det_index) or (None, None)."""
    D = np.array(DET[f], dtype=float)
    if len(D) == 0: return None, None
    d = np.hypot(D[:, 0] - pos[0], D[:, 1] - pos[1]) * PX_NM
    if exclude_idx is not None: d[exclude_idx] = np.inf
    if exclude_pos is not None: d[np.hypot(D[:, 0] - exclude_pos[0], D[:, 1] - exclude_pos[1]) * PX_NM < MERGE_NM] = np.inf
    k = int(np.argmin(d))
    if d[k] > SNAP_NM: return None, None
    n = refine(f, D[k, 0], D[k, 1], r=1); return (n, k)
def track_pair(f0, pa0, pb0, death_nm):
    """returns dict(frames, a, b, sep, found_a, found_b, died, death_frame)"""
    ta, tb = Track(f0, pa0), Track(f0, pb0); fs = [f0]; above = 0; died = False; death_f = None; n_unc = 0; merged = False
    for i in range(frames.index(f0) + 1, F_):
        fp, f = frames[i - 1], frames[i]
        rel = (tb.pos[fp][0] - ta.pos[fp][0], tb.pos[fp][1] - ta.pos[fp][1])                      # last valid relative vector
        qb_pred = (float(np.clip(tb.pos[fp][0] + drift[f][0], 0, Ww - 1)), float(np.clip(tb.pos[fp][1] + drift[f][1], 0, Hh - 1)))
        ta.step(fp, f, excl=[(qb_pred[0], qb_pred[1], RAD)])
        if ta.found[f] and np.hypot(ta.pos[f][0] - qb_pred[0], ta.pos[f][1] - qb_pred[1]) * PX_NM < MERGE_NM:   # a landed on b -> not a find
            ta.pos[f] = (ta.pos[fp][0] + drift[f][0], ta.pos[fp][1] + drift[f][1]); ta.found[f] = False; ta.ncoast += 1
        na_, ka = snap_to_detection(f, ta.pos[f], exclude_pos=qb_pred)                                 # re-anchor a to its detection
        if na_ is not None: ta.pos[f] = na_; ta.found[f] = True; ta.ncoast = 0
        tb.step(fp, f, excl=[(ta.pos[f][0], ta.pos[f][1], RAD)])
        if tb.found[f] and np.hypot(tb.pos[f][0] - ta.pos[f][0], tb.pos[f][1] - ta.pos[f][1]) * PX_NM < MERGE_NM:   # b landed on a -> not a find
            tb.pos[f] = qb_pred; tb.found[f] = False; tb.ncoast += 1
        nb_, kb = snap_to_detection(f, tb.pos[f], exclude_pos=ta.pos[f], exclude_idx=ka)                  # re-anchor b, never to a's detection
        if nb_ is not None: tb.pos[f] = nb_; tb.found[f] = True; tb.ncoast = 0
        if np.hypot(ta.pos[f][0] - tb.pos[f][0], ta.pos[f][1] - tb.pos[f][1]) * PX_NM < MERGE_NM:             # still inside contact: uncollapse
            tb.pos[f] = (ta.pos[f][0] + rel[0], ta.pos[f][1] + rel[1]); tb.found[f] = False; n_unc += 1
            if n_unc >= UNCOLLAPSE_MAX: merged = True; break                                                  # persistent collapse = one particle
        else: n_unc = 0
        fs.append(f)
        sep = np.hypot(ta.pos[f][0] - tb.pos[f][0], ta.pos[f][1] - tb.pos[f][1]) * PX_NM
        above = above + 1 if sep >= death_nm else 0                                            # FINAL: no "both found" requirement
        if above >= SUSTAIN and not died: died = True; death_f = fs[-SUSTAIN]                  # record the death, keep tracking to the end
    A = np.array([ta.pos[f] for f in fs]); B = np.array([tb.pos[f] for f in fs])
    return dict(frames=fs, a=A, b=B, sep=np.hypot(*(A - B).T) * PX_NM, found_a=np.array([ta.found[f] for f in fs]), found_b=np.array([tb.found[f] for f in fs]), died=died, death_frame=death_f, merged=merged)
def vel_corr(A, B):
    v1, v2 = np.diff(A, axis=0), np.diff(B, axis=0)
    if len(v1) < 3: return np.nan
    v1 = v1 - v1.mean(0); v2 = v2 - v2.mean(0); den = np.sqrt((v1 ** 2).sum() * (v2 ** 2).sum()); return float((v1 * v2).sum() / den) if den > 0 else 0.0

# ------------------------------------------------------------------ DISCOVERY: seed every frame, STRICT occupancy, isolated-pair acceptance
def claimed(pt, f, tracks):
    for T in tracks:
        if f in T["fset"]:
            i = T["fset"][f]
            if i < T["n_bond"] and (np.hypot(pt[0] - T["a"][i][0], pt[1] - T["a"][i][1]) * PX_NM < MERGE_NM * 1.5 or np.hypot(pt[0] - T["b"][i][0], pt[1] - T["b"][i][1]) * PX_NM < MERGE_NM * 1.5): return True
    return False
def discover(death_nm, collect_pool=False, log=None):
    accepted = []; rejected = dict(too_short=0, unobservable=0, mean_sep=0, floor=0, vc=0, not_isolated=0, duplicate=0, one_peak=0, duplicate_seed=0); pool = []
    for fi_, f0 in enumerate(frames):
        if LOAD_MODE == "dm4" and fi_ % 5 == 0: print(f"    discovery: seed frame {fi_+1}/{len(frames)}, {len(accepted)} accepted so far", flush=True)
        pts = DET[f0]
        for i, j, d in mutual_pairs(pts, MERGE_NM, death_nm):
            if claimed(pts[i], f0, accepted) or claimed(pts[j], f0, accepted): continue        # STRICT: either particle live elsewhere
            pa0 = refine(f0, *pts[i]); pb0 = refine(f0, *pts[j], excl=[(pa0[0], pa0[1], RAD)])
            if SEED_VALLEY and not two_peaks(f0, pa0, pb0): rejected["duplicate_seed"] = rejected.get("duplicate_seed", 0) + 1; continue
            T = track_pair(f0, pa0, pb0, death_nm)
            n_bond = T["frames"].index(T["death_frame"]) if T["died"] else len(T["frames"])      # bonded frames end at the death frame
            if collect_pool and n_bond >= PERSIST_FRAMES: pool.append(T["sep"][:n_bond])
            bsep = T["sep"][:max(n_bond, 1)]
            iso = float(np.mean([mutual_among_detections(f, T["a"][k], T["b"][k]) for k, f in enumerate(T["frames"][:max(n_bond, 1)])]))
            vc = vel_corr(T["a"][:max(n_bond, 1)], T["b"][:max(n_bond, 1)])
            tp = float(np.mean([two_peaks(f, T["a"][k], T["b"][k]) for k, f in enumerate(T["frames"][:max(n_bond, 1)])]))
            gt_pair_lbl = ""
            if HAS_GT and f0 in gt:
                ids_ = list(gt[f0])
                for ii in range(len(ids_)):
                    for jj in range(ii + 1, len(ids_)):
                        ga_, gb_ = gt[f0][ids_[ii]], gt[f0][ids_[jj]]
                        d1 = max(np.hypot(pa0[0] - ga_[0], pa0[1] - ga_[1]), np.hypot(pb0[0] - gb_[0], pb0[1] - gb_[1])) * PX_NM
                        d2 = max(np.hypot(pa0[0] - gb_[0], pa0[1] - gb_[1]), np.hypot(pb0[0] - ga_[0], pb0[1] - ga_[1])) * PX_NM
                        if min(d1, d2) <= GT_MATCH_NM: gt_pair_lbl = f"{ids_[ii]}-{ids_[jj]}"
            # isolation over the WHOLE trace (so the offline grid can re-cut the bonded phase)
            iso_trace = [mutual_among_detections(f, T["a"][k], T["b"][k]) for k, f in enumerate(T["frames"])]
            C = dict(cand_id=(len(log) if log is not None else -1), f0=f0, t_seed=tsec[f0] - T0, pa0=pa0, pb0=pb0, n_bond=n_bond, died=T["died"], death_frame=T["death_frame"], t_death=(tsec[T["death_frame"]] - T0) if T["died"] else np.nan,
                     mean_sep=float(bsep.mean()), min_sep=float(bsep.min()), iso_frac=iso, twopeak_frac=tp, vc=vc, rigid=bool(np.isfinite(vc) and vc >= 0.95), merged=T["merged"], gt_pair=gt_pair_lbl,
                     trace=dict(frames=T["frames"], sep=T["sep"], a=T["a"], b=T["b"], fa=T["found_a"], fb=T["found_b"], iso=iso_trace),
                     lifetime_s=n_bond * DT, a=T["a"], b=T["b"], frames=T["frames"], found_a=T["found_a"], found_b=T["found_b"])
            if T["merged"]: rejected["floor"] += 1; C.update(outcome="rejected", reason=f"merge: collapsed {UNCOLLAPSE_MAX} consecutive frames at {C['t_seed'] + n_bond*DT:.1f}s"); log is not None and log.append(C); continue
            if n_bond < MIN_BONDED:
                why = "unobservable" if (not T["died"] and frames.index(f0) + MIN_BONDED > F_) else "too_short"; rejected[why] += 1
                C.update(outcome="rejected", reason=f"{why}: {n_bond} bonded frames" + (f", died at {C['t_death']:.1f}s" if T["died"] else "")); log is not None and log.append(C); continue
            if not (MERGE_NM <= bsep.mean() <= death_nm): rejected["mean_sep"] += 1; C.update(outcome="rejected", reason=f"mean_sep {bsep.mean():.2f} nm out of band"); log is not None and log.append(C); continue
            if bsep.min() < MERGE_NM: rejected["floor"] += 1; C.update(outcome="rejected", reason=f"floor: min sep {bsep.min():.2f} nm (merge)"); log is not None and log.append(C); continue
            if iso < MUTUAL_MIN: rejected["not_isolated"] += 1; C.update(outcome="rejected", reason=f"not isolated: mutual-NN in {100*iso:.0f}% of bonded frames"); log is not None and log.append(C); continue
            if np.isfinite(vc) and vc >= DUP_VC and bsep.mean() < DUP_SEP_NM: rejected["duplicate"] = rejected.get("duplicate", 0) + 1; C.update(outcome="rejected", reason=f"duplicate: vc {vc:.2f} at {bsep.mean():.2f} nm (two tracks on one particle)"); log is not None and log.append(C); continue
            if tp < TWOPEAK_MIN: rejected["one_peak"] = rejected.get("one_peak", 0) + 1; C.update(outcome="rejected", reason=f"one peak: valley in only {100*tp:.0f}% of bonded frames"); log is not None and log.append(C); continue
            if np.isfinite(vc) and vc < VC_MIN_BOND: rejected["vc"] += 1; C.update(outcome="rejected", reason=f"vc {vc:.2f} < {VC_MIN_BOND}"); log is not None and log.append(C); continue
            T.update(f0=f0, n_bond=n_bond, mean_sep=float(bsep.mean()), vc=vc, iso_frac=iso, twopeak_frac=tp, lifetime_s=n_bond * DT, fset={f: k for k, f in enumerate(T["frames"])},
                     t_birth=tsec[f0] - T0, t_death=(tsec[T["death_frame"]] - T0) if T["died"] else np.nan)
            accepted.append(T); C.update(outcome="accepted", reason=""); log is not None and log.append(C)
    return accepted, rejected, pool
# ---- separation pool: per-frame mutual-NN separations (samples both states, GT-free); printed either way
_pool = np.array([d for f in frames for _, _, d in mutual_pairs(DET[f], MERGE_NM, DEATH_PRIOR_NM * 1.6)])
PS = pool_stats(_pool)
DEATH_NM = PS["otsu"] if PS.get("override") else DEATH_PRIOR_NM
print(f"[{TAG}] SEPARATION POOL (per-frame mutual-NN pairs, {PS['n']} samples in [{MERGE_NM}, {DEATH_PRIOR_NM*1.6:.1f}] nm):"
      + (f" p5/25/50/75/95 = {np.round(PS['p'],2).tolist()} nm | Otsu split {PS['otsu']:.2f} nm -> modes {PS['mean_lo']:.2f} ({100*PS['frac_lo']:.0f}%) / {PS['mean_hi']:.2f} ({100*PS['frac_hi']:.0f}%), gap {PS['gap']:.2f} nm, valley ratio {PS['valley_ratio']:.2f}" if "otsu" in PS else " too few samples"))
print(f"[{TAG}] DEATH distance: {DEATH_NM:.2f} nm " + (f"from the pool ({PS['reason']})" if PS.get("override") else f"= prior ({PS.get('reason','')}); the pool does not support a data-derived value"))
if "hist" in PS:
    h, e = PS["hist"]; figp, axp = plt.subplots(figsize=(7, 4)); axp.bar(0.5 * (e[1:] + e[:-1]), h, width=e[1] - e[0], color="steelblue", alpha=0.7)
    axp.axvline(PS["otsu"], color="k", ls=":", lw=1.2, label=f"Otsu split {PS['otsu']:.2f} nm (valley {PS['valley_ratio']:.2f})"); axp.axvline(DEATH_NM, color="firebrick", ls="--", lw=1.5, label=f"DEATH in use {DEATH_NM:.2f} nm")
    axp.axvline(MERGE_NM, color="gray", ls=":", lw=1, label=f"floor {MERGE_NM} nm"); axp.set_xlabel("mutual-NN separation (nm)"); axp.set_ylabel("frame-pairs"); axp.set_title(f"{TAG}: separation pool ({PS['n']} samples) — {PS['reason']}", fontsize=9); axp.legend(frameon=False, fontsize=8)
    figp.tight_layout(); figp.savefig(f"Figs/death_pool_{TAG}.png", dpi=200, bbox_inches="tight"); plt.close(figp)
def run_mode(MODE):
    global TWOPEAK_MIN, SEED_VALLEY, CAND, accepted, rejected, died, cens, KM, SC, LIFE
    TWOPEAK_MIN = GATE_MODES[MODE]["TWOPEAK_MIN"]; SEED_VALLEY = GATE_MODES[MODE]["SEED_VALLEY"]
    print("\n" + "#" * 78 + f"\n[{TAG}] FINAL configuration: death = first frame >= {DEATH_NM} nm | floor {MERGE_NM} nm | bonded >= {MIN_BONDED} | vc >= {VC_MIN_BOND} | no isolation / two-peak gates\n" + "#" * 78)
    CAND = []
    accepted, rejected, _ = discover(DEATH_NM, log=CAND)
    pd.DataFrame([{k: v for k, v in C.items() if k not in ("a", "b", "frames", "found_a", "found_b", "pa0", "pb0", "trace")} | dict(seed_ax=C["pa0"][0], seed_ay=C["pa0"][1], seed_bx=C["pb0"][0], seed_by=C["pb0"][1]) for C in CAND]).to_csv(f"Figs/engine_candidates_{TAG}_{MODE}.csv", index=False)
    if True:
        pd.DataFrame([dict(cand_id=C["cand_id"], frame=f, t=tsec[f] - T0, sep_nm=C["trace"]["sep"][k], ax=C["trace"]["a"][k][0], ay=C["trace"]["a"][k][1], bx=C["trace"]["b"][k][0], by=C["trace"]["b"][k][1],
                           found_a=bool(C["trace"]["fa"][k]), found_b=bool(C["trace"]["fb"][k]), iso=bool(C["trace"]["iso"][k]))
                      for C in CAND for k, f in enumerate(C["trace"]["frames"])]).to_csv(f"Figs/engine_candidate_traces_{TAG}.csv", index=False)
    died = [T for T in accepted if T["died"]]; cens = [T for T in accepted if not T["died"]]
    if accepted:
        _as = np.array([T["mean_sep"] for T in accepted])
        print(f"  accepted mean separation: p25/50/75 = {np.percentile(_as,[25,50,75]).round(2).tolist()} nm" + (f"  (pool p25/50/75 = {np.round(PS['p'][1:4],2).tolist()})" if 'p' in PS else ""))
    def _accept_from_log(min_bonded, vc_min, mutual_min):
        ok = [C for C in CAND if not C["merged"] and C["n_bond"] >= min_bonded and MERGE_NM <= C["mean_sep"] <= DEATH_NM and C["min_sep"] >= MERGE_NM and C["iso_frac"] >= mutual_min
              and not (np.isfinite(C["vc"]) and C["vc"] >= DUP_VC and C["mean_sep"] < DUP_SEP_NM) and not (np.isfinite(C["vc"]) and C["vc"] < vc_min)]
        d = [C["lifetime_s"] for C in ok if C["died"]]; return len(ok), len(d), (np.median(d) if d else np.nan)
    print(f"[{TAG}] SENSITIVITY (from the candidate log, occupancy coupling ignored): accepted / died / median lifetime")
    print("    MIN_BONDED  " + "  ".join(f"{mb}: {_accept_from_log(mb, VC_MIN_BOND, MUTUAL_MIN)[0]}/{_accept_from_log(mb, VC_MIN_BOND, MUTUAL_MIN)[1]}/{_accept_from_log(mb, VC_MIN_BOND, MUTUAL_MIN)[2]:.1f}s" for mb in (3, 4, 6, 8)) + f"   (vc>={VC_MIN_BOND}, iso>={MUTUAL_MIN})")
    print("    VC_MIN_BOND " + "  ".join(f"{v:.1f}: {_accept_from_log(MIN_BONDED, v, MUTUAL_MIN)[0]}/{_accept_from_log(MIN_BONDED, v, MUTUAL_MIN)[1]}/{_accept_from_log(MIN_BONDED, v, MUTUAL_MIN)[2]:.1f}s" for v in (0.0, 0.2, 0.3, 0.5)) + f"   (MIN_BONDED={MIN_BONDED}, iso>={MUTUAL_MIN})")
    print("    MUTUAL_MIN  " + "  ".join(f"{m:.1f}: {_accept_from_log(MIN_BONDED, VC_MIN_BOND, m)[0]}/{_accept_from_log(MIN_BONDED, VC_MIN_BOND, m)[1]}/{_accept_from_log(MIN_BONDED, VC_MIN_BOND, m)[2]:.1f}s" for m in (0.0, 0.3, 0.5, 0.7)) + f"   (MIN_BONDED={MIN_BONDED}, vc>={VC_MIN_BOND})")
    print("\n" + "=" * 78); print(f"[{TAG}] ENGINE (GT-free): {len(accepted)} accepted dumbbells | {len(died)} dissociated, {len(cens)} censored | rejected {rejected}")
    if died: print(f"  lifetime (died only): mean {np.mean([T['lifetime_s'] for T in died]):.1f} s, median {np.median([T['lifetime_s'] for T in died]):.1f} s  (left-truncated at {MIN_BONDED*DT:.1f} s)")
    print(f"  criteria: birth mutual-NN in [{MERGE_NM}, {DEATH_NM:.2f}] nm, neither particle in a live dumbbell | duplicate if vc >= {DUP_VC} at < {DUP_SEP_NM} nm | death = first frame >= {DEATH_NM:.2f} nm | bonded >= {MIN_BONDED} fr | mean sep in band | floor {MERGE_NM} nm | no isolation gate | vc >= {VC_MIN_BOND} whole-bond")

    # ---- Kaplan-Meier on all accepted (censored included)
    def kaplan_meier(times, events):
        o = np.argsort(times); t, e = np.asarray(times)[o], np.asarray(events)[o]; S, out = 1.0, []
        for ti in np.unique(t):
            n_at = np.sum(t >= ti); d = np.sum((t == ti) & e); S *= (1 - d / n_at) if n_at else 1; out.append((ti, S))
        return np.array(out)
    KM = kaplan_meier([T["lifetime_s"] for T in accepted], [T["died"] for T in accepted]) if accepted else np.zeros((0, 2))
    if len(KM):
        med = KM[KM[:, 1] <= 0.5, 0]; print(f"  Kaplan-Meier median lifetime (censoring-aware): {med[0]:.1f} s" if len(med) else "  Kaplan-Meier: survival never drops below 0.5 within the window")

    # ------------------------------------------------------------------ SCORECARD vs GT (annotation used only here)
    # GT pairs: same discovery rule as the validation script, with the ENGINE's death distance (apples to apples)
    def gt_pairs(death_nm):
        close, mutual = {}, {}
        for f, g in gt.items():
            P_ = list(g); A = np.array([g[p] for p in P_])
            if len(A) < 2: continue
            Dm = np.hypot(A[:, None, 0] - A[None, :, 0], A[:, None, 1] - A[None, :, 1]) * PX_NM; np.fill_diagonal(Dm, np.inf); nn = Dm.argmin(1)
            for i, p in enumerate(P_):
                for j in range(i + 1, len(P_)):
                    if Dm[i, j] < death_nm:
                        close.setdefault((p, P_[j]), set()).add(f)
                        if nn[i] == j and nn[j] == i: mutual.setdefault((p, P_[j]), set()).add(f)
        def lr(fr):
            fr = sorted(fr); best, cur = [], [fr[0]]
            for a, b in zip(fr, fr[1:]):
                cur = cur + [b] if b == a + 1 else [b]
                if len(cur) > len(best): best = list(cur)
            return best or fr[:1]
        out = []
        for pr, fr in close.items():
            blk = lr(fr)
            if len(blk) >= PAIR_MIN_FRAMES and len(mutual.get(pr, set()) & set(blk)) / len(blk) >= PAIR_MIN_MUTUAL: out.append((pr, blk))
        return out
    def sustained_cross(sep, death, k=SUSTAIN):
        run = 0
        for i, a in enumerate(sep >= death):
            run = run + 1 if a else 0
            if run >= k: return i - k + 1
        return None
    score = []
    for (a_, b_), blk in (gt_pairs(DEATH_NM) if HAS_GT else []):
        fs = [f for f in sorted(gt) if a_ in gt[f] and b_ in gt[f]]; G1 = np.array([gt[f][a_] for f in fs]); G2 = np.array([gt[f][b_] for f in fs]); gsep = np.hypot(*(G1 - G2).T) * PX_NM
        i0 = fs.index(blk[0]); ig = sustained_cross(gsep[i0:], DEATH_NM); T_gt = ((ig) * DT) if ig is not None else np.nan; T_obs = (len(fs) - i0) * DT
        # GT pair's OWN death (validated change-point classifier, as in the validation script) -> T_gt_own
        T_gt_own, own_death = np.nan, np.nan
        cpk = None; best_c = None
        for k in range(6, len(gsep) - 6):
            a, b = gsep[:k], gsep[k:]; cst = a.var() * len(a) + b.var() * len(b)
            if best_c is None or cst < best_c: best_c, cpk = cst, k
        if cpk is not None:
            bon, dis = gsep[:cpk], gsep[cpk:]; ve = 1 - best_c / (gsep.var() * len(gsep) + 1e-9); gap = dis.mean() - bon.mean()
            if gap >= CP_MIN_GAP_NM and ve >= 0.5:
                cands = np.linspace(bon.mean(), dis.mean(), 201); mis = [np.sum(bon >= c) + np.sum(dis < c) for c in cands]; tied = cands[np.array(mis) == min(mis)]
                own_death = float(tied[np.argmin(np.abs(tied - 0.5 * (bon.mean() + dis.mean())))]); igo = sustained_cross(gsep[i0:], own_death); T_gt_own = (igo * DT) if igo is not None else np.nan
        # per-particle detection recall over the bonded frames, and co-detection (both found in the same frame)
        bfr = [f for f in fs[i0:(i0 + ig) if ig is not None else len(fs)] if f in DET]
        def _det_hit(f, pt):
            D = np.array(DET[f], dtype=float)
            return bool(len(D) and float(np.min(np.hypot(D[:, 0] - pt[0], D[:, 1] - pt[1])) * PX_NM) <= DET_MATCH_NM)
        da = [_det_hit(f, gt[f][a_]) for f in bfr]; db = [_det_hit(f, gt[f][b_]) for f in bfr]; both = [x and y for x, y in zip(da, db)]
        first_both = next((k for k, v in enumerate(both) if v), None)
        # best-matching engine dumbbell: both particles within GT_MATCH_NM in >= GT_MATCH_FRAC of the overlapping frames
        best = None; nearest = None
        for T in accepted:
            ov = [f for f in T["frames"] if f in fs]
            if len(ov) < 3: continue
            ok = 0
            for f in ov:
                i, j = T["fset"][f], fs.index(f)
                da = min(np.hypot(*(T["a"][i] - G1[j])), np.hypot(*(T["a"][i] - G2[j]))) * PX_NM; db = min(np.hypot(*(T["b"][i] - G1[j])), np.hypot(*(T["b"][i] - G2[j]))) * PX_NM
                ok += (da <= GT_MATCH_NM and db <= GT_MATCH_NM)
            frac = ok / len(ov)
            if nearest is None or frac > nearest[1]: nearest = (T, frac)
            if frac >= GT_MATCH_FRAC and (best is None or frac > best[1]): best = (T, frac)
        # coverage: fraction of the GT pair's BONDED frames in which some accepted dumbbell has both particles on it
        bonded_fr = fs[i0:(i0 + ig) if ig is not None else len(fs)]; covered = 0
        for f in bonded_fr:
            j = fs.index(f); hit = False
            for T in accepted:
                if f not in T["fset"]: continue
                i = T["fset"][f]
                dA = min(np.hypot(*(T["a"][i] - G1[j])), np.hypot(*(T["a"][i] - G2[j]))) * PX_NM; dB = min(np.hypot(*(T["b"][i] - G1[j])), np.hypot(*(T["b"][i] - G2[j]))) * PX_NM
                if dA <= GT_MATCH_NM and dB <= GT_MATCH_NM: hit = True; break
            covered += hit
        gt_coverage = covered / max(len(bonded_fr), 1)
        row = dict(gt_pair=f"{a_}-{b_}", gt_bonded_start_s=tsec[blk[0]] - T0, T_gt=T_gt, T_gt_own_death=T_gt_own, gt_own_death_nm=own_death, T_obs=T_obs, gt_censored=ig is None, matched=best is not None,
                   gt_coverage=gt_coverage,
                   det_recall_a=float(np.mean(da)) if da else np.nan, det_recall_b=float(np.mean(db)) if db else np.nan, codetect_frac=float(np.mean(both)) if both else np.nan,
                   first_codetect_s=(first_both * DT) if first_both is not None else np.nan,
                   nearest_engine_T=nearest[0]["lifetime_s"] if nearest else np.nan, nearest_match_frac=nearest[1] if nearest else np.nan)
        if best:
            T, frac = best; row.update(engine_birth_s=T["t_birth"], engine_T=T["lifetime_s"], engine_died=T["died"], match_frac=frac, engine_mean_sep=T["mean_sep"], engine_vc=T["vc"])
            if ig is not None: row["abs_err_s"] = T["lifetime_s"] - T_gt if T["died"] else np.nan; row["rel_err"] = row["abs_err_s"] / T_gt if T_gt > 0 else np.nan; row["missed_death"] = not T["died"]
            else: row["false_early_death"] = T["died"]
        score.append(row)
    SC = pd.DataFrame(score)
    if HAS_GT and not len(SC): print(f"[{TAG}] no GT pair qualifies (bonded run >= {MIN_BONDED} frames, majority mutual-NN) -> no scorecard")
    if HAS_GT and len(SC):
        SC.to_csv(f"Figs/engine_scorecard_{TAG}_{MODE}.csv", index=False)
        print("\n" + "-" * 100); print(f"[{TAG}] SCORECARD vs GT  (engine DEATH {DEATH_NM:.2f} nm; T_gt_own = at the GT pair's own change-point death)")
        tbl = SC.copy()
        tbl["GT lifetime"] = [("censored, T_obs %.1f" % r.T_obs) if r.gt_censored else ("%.1f s" % r.T_gt) for _, r in tbl.iterrows()]
        tbl["GT own-death"] = [("%.2f nm -> %.1f s" % (r.gt_own_death_nm, r.T_gt_own_death)) if np.isfinite(r.gt_own_death_nm) else "-" for _, r in tbl.iterrows()]
        tbl["engine"] = [("born %.1f s, T %.1f s %s (match %.0f%%)" % (r.engine_birth_s, r.engine_T, "died" if r.engine_died else "censored", 100 * r.match_frac)) if r.matched else "NOT MATCHED" for _, r in tbl.iterrows()]
        tbl["coverage"] = ["%.0f%% of GT bond" % (100 * r.gt_coverage) for _, r in tbl.iterrows()]
        tbl["verdict"] = ["" if not r.matched else (("MISSED DEATH" if r.missed_death else ("%+.1f s (%+.0f%%)" % (r.abs_err_s, 100 * r.rel_err))) if not r.gt_censored else ("FALSE EARLY DEATH" if r.false_early_death else "agrees")) for _, r in tbl.iterrows()]
        tbl["co-detected"] = ["%.0f%% (first %.1f s)" % (100 * r.codetect_frac, r.first_codetect_s) if np.isfinite(r.codetect_frac) else "-" for _, r in tbl.iterrows()]
        print(tbl[["gt_pair", "GT lifetime", "GT own-death", "co-detected", "coverage", "engine", "verdict"]].to_string(index=False))
        # ---- candidate cross-reference: which candidates were seeded ON this GT pair's particles, and what happened to them
        print(f"\n[{TAG}] CANDIDATE CROSS-REFERENCE (candidates seeded within {GT_MATCH_NM} nm of both GT particles):")
        for (a_, b_), blk in gt_pairs(DEATH_NM):
            hits = []
            for C in CAND:
                f = C["f0"]
                if a_ not in gt.get(f, {}) or b_ not in gt.get(f, {}): continue
                ga, gb = gt[f][a_], gt[f][b_]
                d1 = max(np.hypot(C["pa0"][0] - ga[0], C["pa0"][1] - ga[1]), np.hypot(C["pb0"][0] - gb[0], C["pb0"][1] - gb[1])) * PX_NM
                d2 = max(np.hypot(C["pa0"][0] - gb[0], C["pa0"][1] - gb[1]), np.hypot(C["pb0"][0] - ga[0], C["pb0"][1] - ga[1])) * PX_NM
                if min(d1, d2) <= GT_MATCH_NM: hits.append(C)
            if not hits:
                # was the pair ever eligible to seed? (both detected, mutual-NN, in band, unclaimed) -> explain the first bonded frames
                why = []
                for f in [f for f in blk[:6] if f in DET]:
                    pts = DET[f]; D = np.array(pts); ga, gb = gt[f][a_], gt[f][b_]
                    ia = int(np.argmin(np.hypot(D[:, 0] - ga[0], D[:, 1] - ga[1]))); ib = int(np.argmin(np.hypot(D[:, 0] - gb[0], D[:, 1] - gb[1])))
                    da = np.hypot(*(D[ia] - ga)) * PX_NM; db = np.hypot(*(D[ib] - gb)) * PX_NM
                    if da > DET_MATCH_NM or db > DET_MATCH_NM: why.append(f"f{f}: not both detected"); continue
                    mp = {(i, j) for i, j, _ in mutual_pairs(pts, MERGE_NM, DEATH_NM)}
                    if (ia, ib) not in mp and (ib, ia) not in mp:
                        Dm = np.hypot(D[:, None, 0] - D[None, :, 0], D[:, None, 1] - D[None, :, 1]) * PX_NM; np.fill_diagonal(Dm, np.inf)
                        why.append(f"f{f}: not mutual-NN (nearest to a at {Dm[ia].min():.1f} nm, to b at {Dm[ib].min():.1f} nm; sep {Dm[ia, ib]:.1f})"); continue
                    why.append(f"f{f}: eligible but claimed by a live accepted pair")
                print(f"  GT ({a_},{b_}): NEVER SEEDED. first bonded frames: " + "; ".join(why))
            else:
                print(f"  GT ({a_},{b_}): {len(hits)} candidate(s) seeded on it:")
                for C in hits[:6]:
                    print(f"      seed f{C['f0']} ({C['t_seed']:.1f}s): {C['outcome']:8s} {C['reason']:52s} | bonded {C['n_bond']} fr = {C['lifetime_s']:.1f} s, " + (f"died at {C['t_death']:.1f}s" if C["died"] else "censored") + f", sep {C['mean_sep']:.2f} (min {C['min_sep']:.2f}) nm, iso {100*C['iso_frac']:.0f}%, 2-peak {100*C['twopeak_frac']:.0f}%, vc {C['vc']:.2f}" + (" RIGID" if C["rigid"] else "") + f", found a/b {100*C['found_a'].mean():.0f}%/{100*C['found_b'].mean():.0f}%")
                if len(hits) > 6: print(f"      ... {len(hits)-6} more")
        print("-" * 100)
    else: print(f"[{TAG}] no annotation -> no scorecard")


    return dict(accepted=list(accepted), rejected=dict(rejected), SC=SC.copy() if HAS_GT and len(SC) else pd.DataFrame())

RESULTS = {m: run_mode(m) for m in GATE_MODES}
# ---- restore the primary mode's state for the figure / un-suffixed CSVs below
MODE = PRIMARY_MODE; accepted = RESULTS[MODE]["accepted"]; rejected = RESULTS[MODE]["rejected"]; SC = RESULTS[MODE]["SC"]
died = [T for T in accepted if T["died"]]; cens = [T for T in accepted if not T["died"]]

# ---- ONE summary file per movie: Figs/summary_<id>.csv  (long format: section, key, frame, t, <mode columns>)
def _curves(acc):
    b = np.zeros(F_); d = np.zeros(F_); live = np.zeros(F_)
    for T in acc:
        b[frames.index(T["f0"])] += 1
        if T["died"]: d[frames.index(T["death_frame"])] += 1
        for f in T["frames"][:T["n_bond"]]: live[frames.index(f)] += 1
    return np.cumsum(b), np.cumsum(d), live
def _km_median(acc):
    if not acc: return np.nan
    t = np.array([T["lifetime_s"] for T in acc]); e = np.array([T["died"] for T in acc], bool); o = np.argsort(t); t, e = t[o], e[o]; S = 1.0; out = []
    for ti in np.unique(t):
        n = (t >= ti).sum(); dd = ((t == ti) & e).sum(); S *= (1 - dd / n); out.append((ti, S))
    out = np.array(out); m = out[out[:, 1] <= 0.5, 0]; return float(m[0]) if len(m) else np.nan
rows = []
modes = list(GATE_MODES)
def add(section, key, vals, frame=None, t=None): rows.append(dict(section=section, key=key, frame=frame, t=t, **{m: v for m, v in zip(modes, vals)}))
add("meta", "movie", [TAG] * len(modes)); add("meta", "frames_used", [F_] * len(modes)); add("meta", "DT_s", [DT] * len(modes)); add("meta", "death_nm", [DEATH_NM] * len(modes)); add("meta", "floor_nm", [MERGE_NM] * len(modes))
add("meta", "thr_frac", [THR_FRAC] * len(modes)); add("meta", "gt_registered", [bool(HAS_GT)] * len(modes))
add("meta", "px_nm", [PX_NM] * len(modes)); add("meta", "max_step_px", [MAX_STEP] * len(modes)); add("meta", "excluded_frames", [" ".join(map(str, EXCLUDED)) if EXCLUDED else "none"] * len(modes))
# ---- constants_used_<id>.txt: every given and derived parameter of this run (archive it with the results)
_pool_line = (f"{PS['n']} mutual-NN separations, " + ("bimodal -> override" if PS.get("override") else "unimodal -> prior kept") + (f" (valley {PS['valley_ratio']:.2f}, gap {PS['gap']:.2f} nm)" if "otsu" in PS else "")) if isinstance(PS, dict) else "n/a"
_acc = RESULTS[PRIMARY_MODE]["accepted"]; _d = [T["lifetime_s"] for T in _acc if T["died"]]
_sw = det_sweep[det_sweep.thr_frac == THR_FRAC].iloc[0] if len(det_sweep) else None
_yld = float(_sw.yield_mean) if _sw is not None else float("nan"); _bd = float(_sw.blob_diam_nm) if _sw is not None else float("nan")
_medd = float(np.median(_d)) if _d else float("nan"); _meand = float(np.mean(_d)) if _d else float("nan")
_acc = RESULTS[PRIMARY_MODE]["accepted"]; _d = [T["lifetime_s"] for T in _acc if T["died"]]
_cu = f"""# ============================================================================
#  constants_used.txt  —  every parameter of dumbbell_engine_full_final.py for this run
#  (written by the engine; "derived" = measured from the movie at run time)
# ============================================================================
FOLDER          = {FOLDER}
DM4             = {os.path.basename(dm4[0]) if LOAD_MODE == 'dm4' else 'n/a'}
PX_NM           = {PX_NM:.4f}            # from the .txt (the .dm4 tag is not trusted)
DT              = {DT:.4f}            # seconds per frame, from the .txt
FRAMES_IN_STACK = {len(raw_all) if LOAD_MODE == 'dm4' else F_}
EXCLUDED_FRAMES = {' '.join(map(str, EXCLUDED)) if EXCLUDED else 'none'}            # derived (frame QC)
FRAMES_USED     = {F_}
# DETECTOR (derived threshold)
NORM            = p{int(1)}-p99.9 percentile clip
DET_GSIG_NM     = {DET_GSIG_NM}
DET_DIAM_NM     = {DET_MIN_DIAM_NM}-{DET_MAX_DIAM_NM}
PARTICLE_DIAM_NM = {PARTICLE_DIAM_NM}
THR_FRAC        = {THR_FRAC}            # derived: lowest stationary threshold with blob diam within +-15% of 2.4 nm
YIELD_PER_FRAME = {_yld:.0f}            # derived
BLOB_DIAM_NM    = {_bd:.2f}            # derived
CONF_ISO        = {CONF_ISO}
# TRACKER
R_px, RR_px     = {Rpx}, {RRpx}
LAMBDA          = {LAM}
MAX_STEP_px     = {MAX_STEP}            # derived: p95 of mutual-NN displacements of detections
SNAP_NM         = {SNAP_NM}
PARTNER_MASK_px = {RAD}
MERGE_NM        = {MERGE_NM}
UNCOLLAPSE_MAX  = {UNCOLLAPSE_MAX}
# PHYSICAL CRITERIA (shared across movies)
SEP_MIN_NM      = {MERGE_NM}
SEP_MAX_NM      = {DEATH_NM}
DEATH_NM        = {DEATH_NM}            # death = first frame >= this (SUSTAIN {SUSTAIN}, no found requirement)
DEATH_POOL      = {_pool_line}
MEAN_SEP_BAND   = {MERGE_NM}-{DEATH_NM}
MIN_SEP_NM      = {MERGE_NM}
MIN_BONDED      = {MIN_BONDED}            # left-truncation at {MIN_BONDED*DT:.1f} s
VC_MIN          = {VC_MIN_BOND}
DUP_VC, DUP_SEP = {DUP_VC}, {DUP_SEP_NM}
ISOLATION_GATE  = {'on (MUTUAL_MIN %.2f)' % MUTUAL_MIN if MUTUAL_MIN > 0 else 'off'}
TWO_PEAK_GATE   = {'on' if GATE_MODES[PRIMARY_MODE]['TWOPEAK_MIN'] > 0 else 'off'}
# RESULT
ACCEPTED, DISSOCIATED, CENSORED = {len(_acc)}, {len(_d)}, {len(_acc) - len(_d)}
MEDIAN_LT_S     = {_medd:.1f}
MEAN_LT_S       = {_meand:.1f}
KM_MEDIAN_S     = {_km_median(_acc):.1f}
"""
open(f"Figs/constants_used_{MID if LOAD_MODE == 'dm4' else TAG}.txt", "w").write(_cu)
for key, fn in (("accepted", lambda a: len(a)), ("died", lambda a: sum(T["died"] for T in a)), ("censored", lambda a: sum(not T["died"] for T in a)),
                ("median_lifetime_died_s", lambda a: float(np.median([T["lifetime_s"] for T in a if T["died"]])) if any(T["died"] for T in a) else np.nan),
                ("mean_lifetime_died_s", lambda a: float(np.mean([T["lifetime_s"] for T in a if T["died"]])) if any(T["died"] for T in a) else np.nan),
                ("km_median_s", _km_median), ("mean_sep_p25", lambda a: float(np.percentile([T["mean_sep"] for T in a], 25)) if a else np.nan),
                ("mean_sep_p50", lambda a: float(np.percentile([T["mean_sep"] for T in a], 50)) if a else np.nan), ("mean_sep_p75", lambda a: float(np.percentile([T["mean_sep"] for T in a], 75)) if a else np.nan),
                ("vc_median", lambda a: float(np.median([T["vc"] for T in a if np.isfinite(T["vc"])])) if a else np.nan), ("pop_peak", lambda a: int(_curves(a)[2].max()) if a else 0)):
    add("stats", key, [fn(RESULTS[m]["accepted"]) for m in modes])
for key in sorted(set().union(*[RESULTS[m]["rejected"].keys() for m in modes])): add("rejected", key, [RESULTS[m]["rejected"].get(key, 0) for m in modes])
if HAS_GT:
    for _, r0 in (RESULTS[modes[0]]["SC"] if len(RESULTS[modes[0]]["SC"]) else pd.DataFrame()).iterrows():
        pr = r0.gt_pair
        def g(m, col):
            sc = RESULTS[m]["SC"]; rr = sc[sc.gt_pair == pr]
            return (rr.iloc[0][col] if (len(rr) and col in rr) else np.nan)
        for col in ("T_gt", "T_gt_own_death", "gt_own_death_nm", "gt_censored", "matched", "gt_coverage", "engine_birth_s", "engine_T", "engine_died", "match_frac", "abs_err_s", "codetect_frac", "nearest_match_frac"):
            add("gt_pair_" + pr, col, [g(m, col) for m in modes])
C = {m: _curves(RESULTS[m]["accepted"]) for m in modes}
for i, f in enumerate(frames):
    add("curve", "live", [C[m][2][i] for m in modes], frame=f, t=tsec[f] - T0)
    add("curve", "births_cum", [C[m][0][i] for m in modes], frame=f, t=tsec[f] - T0)
    add("curve", "deaths_cum", [C[m][1][i] for m in modes], frame=f, t=tsec[f] - T0)
for m in modes:
    for T in RESULTS[m]["accepted"]: add("lifetimes", m, [T["lifetime_s"] if mm == m else np.nan for mm in modes], frame=T["f0"], t=float(T["died"]))
SUMMARY = pd.DataFrame(rows); SUMMARY.to_csv(f"Figs/summary_{MID if LOAD_MODE == 'dm4' else TAG}.csv", index=False)
print("\n" + "=" * 78 + f"\n[{TAG}] SUMMARY  (Figs/summary_{MID if LOAD_MODE == 'dm4' else TAG}.csv)")
print(SUMMARY[SUMMARY.section.isin(["stats", "rejected"]) | SUMMARY.section.str.startswith("gt_pair")][["section", "key"] + modes].to_string(index=False))
print("=" * 78)

# ------------------------------------------------------------------ FIGURES (engine's three + detector)
tt = np.array([tsec[f] - T0 for f in frames])
births = np.zeros(F_); deaths = np.zeros(F_); live = np.zeros(F_)
for T in accepted:
    births[frames.index(T["f0"])] += 1
    if T["died"]: deaths[frames.index(T["death_frame"])] += 1
    for f in T["frames"][:T["n_bond"]]: live[frames.index(f)] += 1
fig, ax = plt.subplots(2, 2, figsize=(13, 9)); ax = ax.ravel()
if HAS_GT:
    ax[0].plot(det_sweep.thr_frac, det_sweep.recall, "-s", ms=4, color="darkorange", label="GT recall")
ax[0].plot(det_sweep.thr_frac, det_sweep.yield_cv, "-^", ms=4, color="gray", label="yield CV"); ax[0].axhline(YIELD_CV_MAX, color="gray", ls=":", lw=1)
ax0b = ax[0].twinx(); ax0b.plot(det_sweep.thr_frac, det_sweep.blob_diam_nm, "-d", ms=4, color="purple", label="blob diam (nm)"); ax0b.axhline(PARTICLE_DIAM_NM, color="purple", ls=":", lw=1); ax0b.set_ylabel("blob diameter (nm)", color="purple")
ax[0].axvline(THR_FRAC, color="firebrick", ls="--", lw=1, label=f"operating point {THR_FRAC}"); ax[0].set_xlabel("blob threshold (fraction of frame p99)"); ax[0].set_ylim(0, 1.02); ax[0].set_title("(a) detector"); ax[0].legend(frameon=False, fontsize=8, loc="center right")
lt = [T["lifetime_s"] for T in died]; lc = [T["lifetime_s"] for T in cens]; mx = max(lt + lc + [DT * 4])
bins = np.arange(0, mx + DT * 2, DT * 2)
if lt: ax[1].hist(lt, bins=bins, color="steelblue", alpha=0.75, label=f"dissociated (n={len(lt)})")
if lc: ax[1].hist(lc, bins=bins, color="gray", alpha=0.4, label=f"censored (n={len(lc)})")
ax[1].axvline(MIN_BONDED * DT, color="k", ls=":", lw=1, label=f"left-truncation {MIN_BONDED*DT:.1f} s")
if HAS_GT and len(SC):
    for _, r in SC[~SC.gt_censored].iterrows(): ax[1].axvline(r.T_gt, color="darkorange", lw=1.5, alpha=0.8)
    ax[1].plot([], [], color="darkorange", lw=1.5, label="GT lifetimes (dissociating pairs)")
ax[1].set_xlabel("bonded lifetime (s)"); ax[1].set_ylabel("count"); ax[1].set_title("(b) dumbbell lifetime distribution"); ax[1].legend(frameon=False, fontsize=8)
ax[2].plot(tt, np.cumsum(births), "-", color="green", lw=2, label="cumulative births"); ax[2].plot(tt, np.cumsum(deaths), "-", color="firebrick", lw=2, label="cumulative deaths")
ax[2].axvspan(tt[-1] - MIN_BONDED * DT, tt[-1], color="gray", alpha=0.15, label=f"last {MIN_BONDED} frames: births unobservable"); ax[2].set_xlabel("time (s)"); ax[2].set_title("(c) births and deaths"); ax[2].legend(frameon=False, fontsize=8)
for m, col in zip(GATE_MODES, ("steelblue", "k")): ax[3].plot(tt, C[m][2], "-", color=col, lw=2, label=f"{m} ({len(RESULTS[m]['accepted'])} accepted, {sum(T['died'] for T in RESULTS[m]['accepted'])} died)")
ax[3].set_xlabel("time (s)"); ax[3].set_ylabel("bonded dumbbells"); ax[3].set_title("(d) live dumbbell population"); ax[3].set_ylim(bottom=0); ax[3].legend(frameon=False, fontsize=8)
fig.suptitle(f"{TAG}: GT-free dumbbell engine (final) — {len(accepted)} accepted ({len(died)} died, {len(cens)} censored) | death first frame >= {DEATH_NM:.1f} nm, floor {MERGE_NM} nm, bonded >= {MIN_BONDED}, vc >= {VC_MIN_BOND} | detector THR {THR_FRAC}", fontsize=10)
plt.tight_layout(); fig.savefig(f"Figs/engine_{TAG}_final.png", dpi=300, bbox_inches="tight")
for m in GATE_MODES:
    acc_m = RESULTS[m]["accepted"]
    pd.DataFrame([dict(id=i, f0=T["f0"], t_birth=T["t_birth"], t_death=T["t_death"], lifetime_s=T["lifetime_s"], died=T["died"], mean_sep=T["mean_sep"], vc=T["vc"], iso_frac=T["iso_frac"], twopeak_frac=T["twopeak_frac"]) for i, T in enumerate(acc_m)]).to_csv(f"Figs/engine_dumbbells_{TAG}_{m}.csv", index=False)
    pd.DataFrame([dict(id=i, frame=f, t=tsec[f] - T0, ax=T["a"][k][0], ay=T["a"][k][1], bx=T["b"][k][0], by=T["b"][k][1], sep_nm=T["sep"][k], bonded=k < T["n_bond"], found_a=bool(T["found_a"][k]), found_b=bool(T["found_b"][k]))
                  for i, T in enumerate(acc_m) for k, f in enumerate(T["frames"])]).to_csv(f"Figs/engine_tracks_{TAG}_{m}.csv", index=False)
pd.DataFrame([dict(id=i, f0=T["f0"], t_birth=T["t_birth"], t_death=T["t_death"], lifetime_s=T["lifetime_s"], died=T["died"], mean_sep=T["mean_sep"], vc=T["vc"], iso_frac=T["iso_frac"], twopeak_frac=T["twopeak_frac"]) for i, T in enumerate(accepted)]).to_csv(f"Figs/engine_dumbbells_{TAG}.csv", index=False)
# v9: the tracked trajectories of every accepted dumbbell (the original engine kept these in memory only)
pd.DataFrame([dict(id=i, frame=f, t=tsec[f] - T0, ax=T["a"][k][0], ay=T["a"][k][1], bx=T["b"][k][0], by=T["b"][k][1], sep_nm=T["sep"][k],
                   bonded=k < T["n_bond"], found_a=bool(T["found_a"][k]), found_b=bool(T["found_b"][k]))
              for i, T in enumerate(accepted) for k, f in enumerate(T["frames"])]).to_csv(f"Figs/engine_tracks_{TAG}.csv", index=False)
plt.show()
print(f"saved -> Figs/engine_{TAG}_final.png, Figs/death_pool_{TAG}.png, Figs/engine_dumbbells_{TAG}.csv, Figs/engine_tracks_{TAG}.csv, Figs/engine_candidates_{TAG}.csv, Figs/engine_scorecard_{TAG}.csv, Figs/detector_sweep_{TAG}.csv")
