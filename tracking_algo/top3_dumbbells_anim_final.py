# ============================================================
#  top3_dumbbells_anim_final.py — ANIMATION of the TOP-3 dumbbells (by lifetime), full frame
#   3 rows (dumbbells) x 3 cols: tracking on RAW frames | tracking on DENOISED | separation vs time
#
#  Uses the tracks SAVED by dumbbell_engine_full_final.py (Figs/engine_tracks_<id>-full.csv and
#  Figs/engine_dumbbells_<id>-full.csv), so nothing is re-tracked and no detector has to "match".
#  Frames: the .dm4 in images_<id>_full/, pixel size + frame time from the .txt (as in the engine).
#  Top 3 = the three longest-lived DISSOCIATED dumbbells (a death line is drawn); if fewer than
#  three died, censored ones fill in (no death line).
#  Output: Figs/top3_dumbbells_<id>.mp4 (gif fallback) + Figs/top3_dumbbells_final_<id>.png (300 dpi).
# ============================================================
import os, re, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from scipy.ndimage import gaussian_filter

MOVIES       = ["1051", "1053", "1102"]
DEATH_NM     = 6.5
BIRTH_MIN, BIRTH_MAX = 2.0, 6.5
DET_GSIG_NM  = 0.50                 # the engine's detector blur (col 2)
TAIL, CROP_NM = 10, 30.0            # trail length (frames), crop half-width (nm)
FRAME_STEP   = 2
PREFER_DIED  = True
# per-movie selection filters (bonded phase): separation must fluctuate (std >= min_sep_std nm) and both particles must be
# found in >= min_found of the bonded frames -> excludes pairs that were coasted at a frozen separation. Others: no filter.
SELECT = {"1051": dict(min_sep_std=0.4, min_found=0.8)}
os.makedirs("Figs", exist_ok=True)

def parse_txt(path):
    txt = open(path, errors="ignore").read(); out = {}
    for key, name in (("Pixel size", "px"), ("Frame time", "dt")):
        r = re.search(key + r"\s*\t\s*([\-\d.]+)\s*([a-zA-Zµ]+)", txt)
        if r: out[name] = float(r.group(1)) * {"pm": 1e-3, "nm": 1.0, "µm": 1e3, "ms": 1e-3, "s": 1.0, "µs": 1e-6}.get(r.group(2), np.nan)
    return out
def load_dm4(folder):
    from ncempy.io import dm
    p = glob.glob(os.path.join(folder, "*.dm4"))[0]; a = np.asarray(dm.dmReader(p)["data"]).astype(np.float32)
    if a.ndim == 2: a = a[None]
    if a.shape[0] > a.shape[1] and a.shape[0] > a.shape[2]: a = np.transpose(a, (2, 0, 1))
    return a
def pclip(a): lo, hi = np.percentile(a, [1, 99.9]); return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

for mid in MOVIES:
    folder = f"images_{mid}_full"; tag = f"{mid}-full"
    try:
        D = pd.read_csv(f"Figs/engine_dumbbells_{tag}.csv"); T = pd.read_csv(f"Figs/engine_tracks_{tag}.csv")
    except FileNotFoundError as e: print(f"[{mid}] {e} -> skipped"); continue
    meta = parse_txt(glob.glob(os.path.join(folder, "*.txt"))[0]); PX, DT = meta["px"], meta["dt"]
    raw = load_dm4(folder); F_, Hh, Ww = raw.shape
    qc = f"Figs/frame_qc_{mid}.csv"; BAD = [int(f) for f in pd.read_csv(qc).query("excluded").frame] if os.path.exists(qc) else []
    print(f"[{mid}] {F_} frames, {PX:.4f} nm/px, DT {DT:.3f} s, corrupt frames skipped in the render: {BAD}")
    raw01 = np.stack([pclip(fr) for fr in raw]); den = np.stack([gaussian_filter(fr, DET_GSIG_NM / PX) for fr in raw01])
    # ---- top 3 by lifetime
    D["died"] = D.died.astype(bool)
    # bonded-phase separation std and found fraction, from the saved tracks
    stats = {}
    for i, t in T.groupby("id"):
        t = t.sort_values("frame"); nb = int(round(float(D[D.id == i].lifetime_s.iloc[0]) / DT)); nb = max(2, min(nb, len(t)))
        stats[i] = (float(t.sep_nm.values[:nb].std()), float((t.found_a.values[:nb].astype(bool) & t.found_b.values[:nb].astype(bool)).mean()))
    D["sep_std"] = D.id.map(lambda i: stats.get(i, (np.nan, np.nan))[0]); D["found_frac"] = D.id.map(lambda i: stats.get(i, (np.nan, np.nan))[1])
    pool = D
    if mid in SELECT:
        sel = SELECT[mid]; pool = D[(D.sep_std >= sel["min_sep_std"]) & (D.found_frac >= sel["min_found"])]
        print(f"   selection filter for {mid}: sep std >= {sel['min_sep_std']} nm and found >= {sel['min_found']:.0%} -> {len(pool)}/{len(D)} dumbbells eligible")
        if len(pool) < 3: pool = D; print("   (fewer than 3 eligible -> filter ignored)")
    order = pool.sort_values(["died", "lifetime_s"], ascending=[False, False]) if PREFER_DIED else pool.sort_values("lifetime_s", ascending=False)
    top = order.head(3)
    D3 = []
    for _, r in top.iterrows():
        t = T[T.id == r.id].sort_values("frame"); A = t[["ax", "ay"]].values; B = t[["bx", "by"]].values; fr = t.frame.values.astype(int); sep = t.sep_nm.values
        nb = int(round(r.lifetime_s / DT)); death_f = int(round(r.t_death / DT)) if r.died else None
        d = dict(id=int(r.id), A=A, B=B, frames=fr, sep=sep, died=bool(r.died), lifetime=float(r.lifetime_s), death_f=death_f, nb=nb, mean_sep=float(r.mean_sep), vc=float(r.vc))
        cx = int((A[:nb, 0].mean() + B[:nb, 0].mean()) / 2); cy = int((A[:nb, 1].mean() + B[:nb, 1].mean()) / 2); half = int(CROP_NM / PX)
        cx = min(max(cx, half), Ww - half); cy = min(max(cy, half), Hh - half)
        d.update(xlo=cx - half, xhi=cx + half, ylo=cy - half, yhi=cy + half); D3.append(d)
        print(f"   dumbbell #{d['id']}: born frame {fr[0]}, lifetime {d['lifetime']:.1f} s, {'died frame %d' % death_f if d['died'] else 'censored'}, mean sep {d['mean_sep']:.2f} nm (std {r.sep_std:.2f}), found {100*r.found_frac:.0f}%, vc {d['vc']:+.2f}")
    if not D3: print(f"[{mid}] no accepted dumbbells -> skipped"); continue

    def draw_track(ax, bg, d, g):
        f0, last = d["frames"][0], (d["death_f"] if d["died"] else d["frames"][-1])
        crop = bg[d["ylo"]:d["yhi"], d["xlo"]:d["xhi"]]; lo, hi = np.percentile(crop, [2, 98]) if crop.size else (0, 1); hi = hi if hi > lo else lo + 1e-6
        ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi, extent=[d["xlo"], d["xhi"], d["yhi"], d["ylo"]], origin="upper")
        state = "Not yet formed" if g < f0 else ("Dissociated" if g > last else "Bonded")
        if f0 <= g <= d["frames"][-1]:
            k = int(np.searchsorted(d["frames"], g)); k = min(k, len(d["frames"]) - 1); s = max(0, k - TAIL)
            ax.plot(d["A"][s:k + 1, 0], d["A"][s:k + 1, 1], "-", color="red", lw=2, alpha=.8); ax.plot(d["B"][s:k + 1, 0], d["B"][s:k + 1, 1], "-", color="cyan", lw=2, alpha=.8)
            if g <= last: ax.plot([d["A"][k, 0], d["B"][k, 0]], [d["A"][k, 1], d["B"][k, 1]], "-", color="yellow", lw=1.5)
            ax.scatter(d["A"][k, 0], d["A"][k, 1], s=70, facecolors="none", edgecolors="red", lw=2); ax.scatter(d["B"][k, 0], d["B"][k, 1], s=70, facecolors="none", edgecolors="cyan", lw=2)
        ax.set_xlim(d["xlo"], d["xhi"]); ax.set_ylim(d["yhi"], d["ylo"]); ax.axis("off"); return state

    fig, axes = plt.subplots(len(D3), 3, figsize=(10, 3 * len(D3)), dpi=90, gridspec_kw={"width_ratios": [1, 1, 1.5]}, squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.07, hspace=0.50, wspace=0.28)
    time_text = fig.text(0.5, 0.955, "", ha="center", fontsize=14, weight="bold")
    for row, d in enumerate(D3):
        p = axes[row, 1].get_position(); fig.text(p.x0 + p.width / 2, p.y0 - 0.018, f"lifetime {d['lifetime']:.1f} s" + ("" if d["died"] else " (censored)"), ha="center", fontsize=11, weight="bold")
    def draw(g):
        for row, d in enumerate(D3):
            aRaw, aDen, aSep = axes[row]; aRaw.clear(); aDen.clear(); aSep.clear(); gi = min(g, F_ - 1)
            st = draw_track(aRaw, raw01[gi], d, g); draw_track(aDen, den[gi], d, g)
            aRaw.set_title(f"#{d['id']}  Original  |  {st}", fontsize=9)
            if row == 0: aDen.set_title("Denoised Frame", fontsize=9)
            tg = d["frames"] * DT; upto = d["frames"] <= g
            aSep.plot(tg[upto], d["sep"][upto], "-o", ms=3, color="k")
            aSep.axhline(DEATH_NM, ls="--", color="orange", lw=1.3, label=f"dissoc at {DEATH_NM} nm"); aSep.axhspan(BIRTH_MIN, BIRTH_MAX, color="green", alpha=0.08, label=f"birth {BIRTH_MIN:g}-{BIRTH_MAX:g} nm")
            if d["died"]: aSep.axvline(d["death_f"] * DT, ls="--", color="purple", lw=1.5, label="death")
            aSep.set_xlim(0, F_ * DT); aSep.set_ylim(0, max(d["sep"].max() * 1.1, DEATH_NM + 1)); aSep.set_ylabel("sep (nm)", fontsize=8)
            if row == len(D3) - 1: aSep.set_xlabel("time (s)")
            aSep.legend(fontsize=6, loc="upper left")
        time_text.set_text(f"t = {g * DT:.1f} s"); return []
    seq = [f for f in range(0, F_, FRAME_STEP) if f not in BAD]
    anim = FuncAnimation(fig, draw, frames=seq, interval=140, blit=False); saved = None
    try:
        if FFMpegWriter.isAvailable(): saved = f"Figs/top3_dumbbells_{mid}.mp4"; anim.save(saved, writer=FFMpegWriter(fps=7, bitrate=2000))
    except Exception as e: print("mp4 writer failed, falling back to gif:", e); saved = None
    if saved is None: saved = f"Figs/top3_dumbbells_{mid}.gif"; anim.save(saved, writer=PillowWriter(fps=7))
    draw(F_ - 1); png = f"Figs/top3_dumbbells_final_{mid}.png"; fig.savefig(png, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"   saved {saved} ({len(seq)} frames, {os.path.getsize(saved)/1e6:.1f} MB) + {png}")
