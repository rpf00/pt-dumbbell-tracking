# ============================================================
#  fullframe_scan.py — triage of full-frame .dm4 movies before running the dumbbell engine
#
#  For each folder images_<id>_full/ containing <name>.dm4 and <name>.txt (Velox export):
#    1. load the stack (ncempy) and the metadata (.txt: pixel size, frame time, dwell, mag, gain);
#       cross-check the pixel size in the file against the .txt
#    2. per-frame QC: median / p1 / p99 intensity, zero-pixel and saturated-pixel fractions,
#       normalised cross-correlation with the previous frame, global drift (band-passed cross-
#       correlation, no phase whitening) -> frames flagged as corrupt / discontinuous
#    3. detector: the engine's GT-free threshold rule (lowest stationary threshold with deconvolved
#       blob diameter within +-15 % of 2.4 nm) on a subsample of frames; yield, CV, blob size, CNR
#    4. a verdict per movie: GOOD / MARGINAL / REJECT with the reasons, written to
#       Figs/fullframe_scan_summary.csv, and a one-page figure per movie.
#  Nothing here needs annotation. The frame-exclusion list is written per movie so the engine can
#  use it: Figs/frame_qc_<id>.csv
# ============================================================
import os, re, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from skimage.feature import blob_log
from skimage.registration import phase_cross_correlation

FOLDERS = sorted(glob.glob("images_*_full"))
PARTICLE_DIAM_NM = 2.4
DET_GSIG_NM, DET_MIN_DIAM_NM, DET_MAX_DIAM_NM = 0.50, 1.6, 4.0
THR_GRID = [0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30]
YIELD_CV_MAX, SIZE_TOL = 0.35, 0.15
SWEEP_EVERY = 4                       # detector sweep on every n-th frame (1024^2 blob_log is slow)
ZERO_FRAC_MAX, SAT_FRAC_MAX = 0.005, 0.01
NCC_REL_MIN, MED_MAD_K = 0.3, 4.0     # frame flagged if NCC to previous < NCC_REL_MIN x the movie median NCC, or median > K MAD from the running median
DRIFT_BP = (2.0, 12.0)
OUT = "Figs"; os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ loaders
def parse_txt(path):
    txt = open(path, errors="ignore").read(); m = {}
    def grab(key, pat=r"([\-\d.]+)\s*([a-zA-Zµ]+)"):
        r = re.search(key + r"\s*\t\s*" + pat, txt); return (float(r.group(1)), r.group(2)) if r else (None, None)
    v, u = grab("Pixel size"); m["px_nm"] = v * {"pm": 1e-3, "nm": 1.0, "µm": 1e3}.get(u, np.nan) if v else None
    v, u = grab("Frame time"); m["dt_s"] = v * {"ms": 1e-3, "s": 1.0, "µs": 1e-6}.get(u, np.nan) if v else None
    v, u = grab("Dwell time"); m["dwell_us"] = v * {"ns": 1e-3, "µs": 1.0, "us": 1.0, "ms": 1e3}.get(u, np.nan) if v else None
    r = re.search(r"STEM Magnification\s*\t\s*([\d.]+)\s*kx", txt); m["mag_kx"] = float(r.group(1)) if r else None
    r = re.search(r"HAADF\s*\tEnabled.*?Gain\s*\t\s*([\d.]+)\s*dB", txt, re.S); m["haadf_gain_db"] = float(r.group(1)) if r else None
    r = re.search(r"Defocus\s*\t\s*([\-\d.]+)\s*nm", txt); m["defocus_nm"] = float(r.group(1)) if r else None
    r = re.search(r"substack\((\d+)-(\d+)\)", os.path.basename(path)); m["substack"] = (int(r.group(1)), int(r.group(2))) if r else None
    return m
def load_dm4(path):
    from ncempy.io import dm
    d = dm.dmReader(path); a = np.asarray(d["data"]).astype(np.float32)
    if a.ndim == 2: a = a[None]
    if a.shape[0] > a.shape[1] and a.shape[0] > a.shape[2]: a = np.transpose(a, (2, 0, 1))   # frame axis last -> first
    px = None
    try:
        ps, pu = d.get("pixelSize"), d.get("pixelUnit")
        if ps is not None: px = float(ps[-1]) * {"pm": 1e-3, "nm": 1.0, "µm": 1e3, "um": 1e3}.get(str(pu[-1]), np.nan)
    except Exception: pass
    return a, px
def load_movie(folder):
    dm4 = glob.glob(os.path.join(folder, "*.dm4")); assert dm4, f"no .dm4 in {folder}"
    txt = glob.glob(os.path.join(folder, "*.txt")); meta = parse_txt(txt[0]) if txt else {}
    stack, px_file = load_dm4(dm4[0])
    px = meta.get("px_nm") or px_file; meta["px_nm_file"] = px_file; meta["px_nm"] = px
    if px_file and meta.get("px_nm") and abs(px_file - meta["px_nm"]) / meta["px_nm"] > 0.02: print(f"  !! pixel size disagrees: .txt {meta['px_nm']:.4f} vs file {px_file:.4f} nm")
    return stack, meta, os.path.basename(dm4[0])

# ------------------------------------------------------------------ helpers
def norm_pclip(a): lo, hi = np.percentile(a, [1, 99.9]); return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
def ncc(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12))
def sigma_bg(s): m = np.median(s); return 1.4826 * np.median(np.abs(s[s < m] - m)) + 1e-9

# ------------------------------------------------------------------ run
rows = []
for folder in FOLDERS:
    mid = re.findall(r"images_(\w+)_full", folder)[0]
    try: stack, meta, name = load_movie(folder)
    except Exception as e: print(f"[{mid}] load failed: {e}"); rows.append(dict(movie=mid, verdict="REJECT", reason=f"load failed: {e}")); continue
    F, H, W = stack.shape; PX = meta["px_nm"]; DT = meta.get("dt_s") or np.nan
    print(f"\n[{mid}] {name}: {F} frames {H}x{W}, {PX:.4f} nm/px, DT {DT:.3f} s, dwell {meta.get('dwell_us')} us, {meta.get('mag_kx')} kx, gain {meta.get('haadf_gain_db')} dB, defocus {meta.get('defocus_nm')} nm, dtype max {stack.max():.0f}")
    # ---- frame QC
    sat = float(stack.max()); q = []
    small = [zoom(gaussian_filter(norm_pclip(stack[i]), 3.0), 0.25, order=1) for i in range(F)]   # smoothed 256^2 for NCC
    bp = [gaussian_filter(norm_pclip(stack[i]), DRIFT_BP[0]) - gaussian_filter(norm_pclip(stack[i]), DRIFT_BP[1]) for i in range(F)]
    for i in range(F):
        a = stack[i]; r = dict(frame=i, median=float(np.median(a)), p1=float(np.percentile(a, 1)), p99=float(np.percentile(a, 99)),
                                zero_frac=float((a <= 0).mean()), sat_frac=float((a >= sat).mean()) if sat > 0 else 0.0,
                                ncc_prev=ncc(small[i], small[i - 1]) if i else np.nan)
        if i:
            sh, _, _ = phase_cross_correlation(bp[i - 1], bp[i], upsample_factor=4, normalization=None); r["drift_px"] = float(np.hypot(sh[0], sh[1]))
        else: r["drift_px"] = 0.0
        q.append(r)
    Q = pd.DataFrame(q); med_run = Q["median"].rolling(5, center=True, min_periods=1).median(); mad = 1.4826 * (Q["median"] - med_run).abs().median() + 1e-9
    Q["flag_zero"] = Q.zero_frac > ZERO_FRAC_MAX; Q["flag_sat"] = Q.sat_frac > SAT_FRAC_MAX
    Q["flag_ncc"] = Q.ncc_prev < max(0.05, NCC_REL_MIN * np.nanmedian(Q.ncc_prev)); Q["flag_jump"] = (Q["median"] - med_run).abs() > np.maximum(MED_MAD_K * mad, 0.02 * med_run)
    Q["flag_drift"] = Q.drift_px * PX > 0.35 * min(H, W) * PX * 0.5
    Q["excluded"] = Q.flag_zero | Q.flag_sat | Q.flag_ncc | Q.flag_jump
    Q.to_csv(f"{OUT}/frame_qc_{mid}.csv", index=False)
    good = [int(i) for i in Q.frame[~Q.excluded]]; n_ex = int(Q.excluded.sum())
    print(f"  QC: {n_ex}/{F} frames excluded (zero {int(Q.flag_zero.sum())}, sat {int(Q.flag_sat.sum())}, low NCC {int(Q.flag_ncc.sum())}, intensity jump {int(Q.flag_jump.sum())}) | "
          f"median intensity {Q['median'].iloc[0]:.0f} -> {Q['median'].iloc[-1]:.0f} | drift median {Q.drift_px.median():.1f}px ({Q.drift_px.median()*PX:.2f} nm), max {Q.drift_px.max():.1f}px | NCC(prev) median {Q.ncc_prev.median():.2f}")
    # ---- detector sweep on good frames (subsampled)
    sub = good[::SWEEP_EVERY] or good[:1]
    MIN_SIG = (DET_MIN_DIAM_NM / 2) / np.sqrt(2) / PX; MAX_SIG = (DET_MAX_DIAM_NM / 2) / np.sqrt(2) / PX; blur = DET_GSIG_NM / PX
    dets = {i: gaussian_filter(norm_pclip(stack[i]), blur) for i in sub}
    sw = []
    for thr in THR_GRID:
        ys, ds = [], []
        for i in sub:
            fr = dets[i]; bl = blob_log(fr, min_sigma=MIN_SIG, max_sigma=MAX_SIG, num_sigma=5, threshold=thr * float(np.percentile(fr, 99)))
            ys.append(len(bl)); sg = bl[:, 2] if len(bl) else np.array([])
            ds.append(float(np.mean(2 * np.sqrt(2) * np.sqrt(np.clip(sg ** 2 - blur ** 2, 0, None))) * PX) if len(sg) else np.nan)
        ys = np.array(ys); sw.append(dict(thr=thr, yield_mean=ys.mean(), yield_cv=ys.std() / max(ys.mean(), 1e-9), blob_diam_nm=float(np.nanmean(ds))))
    SW = pd.DataFrame(sw); SW["stationary"] = SW.yield_cv <= YIELD_CV_MAX; SW["size_ok"] = (SW.blob_diam_nm - PARTICLE_DIAM_NM).abs() / PARTICLE_DIAM_NM <= SIZE_TOL
    ok = SW[SW.stationary & SW.size_ok]
    thr_sel = float(ok.thr.min()) if len(ok) else float(SW.iloc[(SW.blob_diam_nm - PARTICLE_DIAM_NM).abs().argmin()].thr); sel = SW[SW.thr == thr_sel].iloc[0]
    # CNR proxy on the good frames: (p99.5 - median) / sigma_bg of the detector image
    cnr = float(np.mean([(np.percentile(dets[i], 99.5) - np.median(dets[i])) / sigma_bg(dets[i]) for i in sub]))
    print(f"  detector: threshold {thr_sel} ({'passes both gates' if len(ok) else 'FALLBACK: no threshold passed'}) -> yield {sel.yield_mean:.0f}/frame (CV {sel.yield_cv:.2f}), blob diam {sel.blob_diam_nm:.2f} nm | CNR proxy {cnr:.1f}")
    # ---- verdict
    reasons = []
    if n_ex / F > 0.2: reasons.append(f"{n_ex}/{F} frames corrupt")
    if not len(ok): reasons.append("no stationary + physical-size detector threshold")
    if Q.drift_px.median() * PX > PARTICLE_DIAM_NM: reasons.append(f"median drift {Q.drift_px.median()*PX:.1f} nm/frame > particle diameter")
    if cnr < 3: reasons.append(f"CNR proxy {cnr:.1f} < 3")
    if sel.yield_mean < 20: reasons.append(f"only {sel.yield_mean:.0f} detections/frame")
    verdict = "GOOD" if not reasons else ("MARGINAL" if len(reasons) == 1 else "REJECT")
    print(f"  VERDICT: {verdict}" + (f" — {'; '.join(reasons)}" if reasons else ""))
    rows.append(dict(movie=mid, file=name, frames=F, excluded=n_ex, px_nm=PX, dt_s=DT, dwell_us=meta.get("dwell_us"), mag_kx=meta.get("mag_kx"), gain_db=meta.get("haadf_gain_db"),
                     drift_med_nm=Q.drift_px.median() * PX, drift_max_nm=Q.drift_px.max() * PX, ncc_med=Q.ncc_prev.median(), thr=thr_sel, yield_mean=sel.yield_mean, yield_cv=sel.yield_cv,
                     blob_diam_nm=sel.blob_diam_nm, cnr=cnr, verdict=verdict, reason="; ".join(reasons)))
    # ---- figure
    fig, ax = plt.subplots(2, 2, figsize=(13, 9)); ax = ax.ravel()
    ax[0].plot(Q.frame, Q["median"], "-", color="k", label="median"); ax[0].plot(Q.frame, Q.p99, "-", color="gray", label="p99"); ax[0].plot(Q.frame, Q.p1, "-", color="lightgray", label="p1")
    for i in Q.frame[Q.excluded]: ax[0].axvline(i, color="red", alpha=0.3)
    ax[0].set_xlabel("frame"); ax[0].set_ylabel("raw intensity"); ax[0].set_title(f"(a) intensity per frame — {n_ex} excluded (red)"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(Q.frame, Q.drift_px * PX, "-o", ms=3, color="firebrick", label="drift to previous frame (nm)"); ax[1].axhline(PARTICLE_DIAM_NM, color="gray", ls=":", lw=1)
    ax1b = ax[1].twinx(); ax1b.plot(Q.frame, Q.ncc_prev, "-", color="steelblue", label="NCC to previous"); ax1b.set_ylim(0, 1); ax1b.set_ylabel("NCC", color="steelblue")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("drift (nm)", color="firebrick"); ax[1].set_title("(b) drift and frame-to-frame similarity")
    ax[2].plot(SW.thr, SW.yield_cv, "-^", color="gray", label="yield CV"); ax[2].axhline(YIELD_CV_MAX, color="gray", ls=":", lw=1); ax[2].set_ylim(0, 1.02)
    ax2b = ax[2].twinx(); ax2b.plot(SW.thr, SW.blob_diam_nm, "-d", color="purple", label="blob diam (nm)"); ax2b.axhline(PARTICLE_DIAM_NM, color="purple", ls=":", lw=1); ax2b.set_ylabel("blob diameter (nm)", color="purple")
    ax[2].axvline(thr_sel, color="firebrick", ls="--", lw=1); ax[2].set_xlabel("blob threshold (fraction of p99)"); ax[2].set_title(f"(c) detector — threshold {thr_sel}, {sel.yield_mean:.0f}/frame")
    i0 = sub[len(sub) // 2]; fr = dets[i0]; bl = blob_log(fr, min_sigma=MIN_SIG, max_sigma=MAX_SIG, num_sigma=5, threshold=thr_sel * float(np.percentile(fr, 99)))
    ax[3].imshow(fr, cmap="gray", vmin=np.percentile(fr, 2), vmax=np.percentile(fr, 99.5)); ax[3].plot(bl[:, 1], bl[:, 0], "o", mfc="none", mec="cyan", ms=4, mew=0.6)
    ax[3].set_title(f"(d) frame {i0}: {len(bl)} detections"); ax[3].set_xticks([]); ax[3].set_yticks([])
    fig.suptitle(f"{mid}: {name} — {verdict}" + (f" ({'; '.join(reasons)})" if reasons else ""), fontsize=11); plt.tight_layout()
    fig.savefig(f"{OUT}/fullframe_scan_{mid}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/fullframe_scan_summary.csv", index=False)
print("\n" + S[[c for c in ["movie", "frames", "excluded", "px_nm", "dt_s", "dwell_us", "drift_med_nm", "ncc_med", "thr", "yield_mean", "yield_cv", "blob_diam_nm", "cnr", "verdict", "reason"] if c in S]].round(2).to_string(index=False))
