# ============================================================
#  fullframe_plot_final.py — publication figures from the final engine run, one movie at a time
#
#  Reads Figs/summary_<id>.csv (dumbbell_engine_full_final.py) and Figs/engine_dumbbells_<id>-full.csv.
#  Writes, per movie, at 600 dpi (PNG + PDF), no titles, axis labels only:
#     Figs/lifetime_distribution_<id>.png   histogram of bonded lifetimes of DISSOCIATED dumbbells,
#                                           mean (red solid) and median (green dashed)
#     Figs/live_population_<id>.png         number of live dumbbells vs time (purple)
#     Figs/mean_separation_<id>.png         histogram of the mean bonded separation of accepted
#                                           dumbbells, with p25 / p50 / p75 marked
# ============================================================
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt

MOVIES = ["1051", "1053", "1102"]
OUT, DPI = "Figs", 600
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14})

def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(f"{OUT}/{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

for mid in MOVIES:
    f = f"{OUT}/summary_{mid}.csv"
    if not os.path.exists(f): print(f"[{mid}] {f} missing -> skipped"); continue
    S = pd.read_csv(f); col = [c for c in S.columns if c not in ("section", "key", "frame", "t")][0]
    DT = float(S[S.key == "DT_s"][col].iloc[0])
    # ---- (1) lifetime distribution (dissociated only)
    lt = S[S.section == "lifetimes"]; x = lt[col].astype(float).values; died = lt.t.astype(float).values == 1
    L = x[died]
    fig, ax = plt.subplots(figsize=(7, 5.6))
    bins = np.arange(L.min() - DT / 2, L.max() + DT, DT) if len(L) else 10
    ax.hist(L, bins=bins, color="steelblue", edgecolor="black", linewidth=1.2)
    ax.axvline(L.mean(), color="firebrick", lw=3, label=f"mean {L.mean():.1f} s")
    ax.axvline(np.median(L), color="green", lw=3, ls="--", label=f"median {np.median(L):.1f} s")
    ax.set_xlabel("bonded lifetime (s)"); ax.set_ylabel("# dumbbells"); ax.legend(frameon=True, edgecolor="none")
    save(fig, f"lifetime_distribution_{mid}")
    # ---- (2) live population
    c = S[S.section == "curve"]; live = c[c.key == "live"]; t = live.t.astype(float).values; n = live[col].astype(float).values
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.plot(t, n, "-", color="purple", lw=3); ax.set_xlabel("time (s)"); ax.set_ylabel("# live dumbbells"); ax.set_ylim(bottom=0)
    save(fig, f"live_population_{mid}")
    # ---- (3) mean bonded separation of accepted dumbbells
    d = f"{OUT}/engine_dumbbells_{mid}-full.csv"
    if os.path.exists(d):
        D = pd.read_csv(d); sep = D.mean_sep.values; q = np.percentile(sep, [25, 50, 75])
        fig, ax = plt.subplots(figsize=(7, 5.6))
        ax.hist(sep, bins=np.arange(2.0, 6.75, 0.25), color="steelblue", edgecolor="black", linewidth=1.2)
        for v, ls, lab in zip(q, (":", "--", ":"), ("p25", "p50", "p75")): ax.axvline(v, color="firebrick" if lab == "p50" else "gray", lw=3 if lab == "p50" else 2, ls=ls, label=f"{lab} {v:.2f} nm")
        ax.set_xlabel("mean bonded separation (nm)"); ax.set_ylabel("# dumbbells"); ax.legend(frameon=True, edgecolor="none")
        save(fig, f"mean_separation_{mid}")
    else: print(f"[{mid}] {d} missing -> no separation histogram")
    print(f"[{mid}] n={len(x)} died={died.sum()} | lifetime mean {L.mean():.1f} median {np.median(L):.1f} s | population peak {n.max():.0f}" + (f" | mean sep p25/50/75 {q.round(2).tolist()}" if os.path.exists(d) else ""))
