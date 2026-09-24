# ============================================================
#  MULTI-FRAME dumbbell discovery — BONDED-LIFETIME   [1106 BUILD]
#
#  This is population_engine_optimal.py adapted for 20250110_HAADF-BF_STEM_150_kx_1106.emd.
#  1053 is the fixed reference; every 1051 parameter below was MEASURED, not assumed:
#    - detector (sigma 0.60 nm, blob scale = 2.4 nm physical diameter, thr 0.16*p99):
#      passed the stationarity check (deconvolved diameter flat at 2.21 nm across the movie)
#    - MAX_STEP = 10 px: measured p95 of frame-to-frame motion (engine default 3 covered
#      only the 13th percentile of real motion)
#    - birth/death/floor: KEPT from 1053 UNCHANGED — deliberately NOT re-tuned. Only 18% of
#      1106's nearest-neighbour pairs fall in the 2.5-6.5 nm band (vs 78% on 1051): 1106 is
#      a MORE DISPERSED field (NN median 10.7 nm). We do NOT widen the band to capture more
#      pairs (that would be population p-hacking). Instead we run the SAME criteria and ask:
#      do the ~18% of pairs that DO qualify give a lifetime consistent with 1053/1051?
#      Expect a LOW accepted count; the median lifetime is the test, not the count.
#    - tracker: 0 swap collisions over 12 isolated seeds at MAX_STEP=10
#
#  KEY STRUCTURAL DIFFERENCE vs the 1053 engine: the threshold is PER-FRAME FRACTIONAL
#  (0.16 * that frame's p99), not a single absolute value. blob_log's threshold therefore
#  cannot live in the BLOB dict; detect() computes it per frame. See THR_FRAC / detect().
#
#  OPERATING POINT from the stationarity/CNR grid search (GROUND-TRUTH-FREE).
#
#  DETECTOR (this movie only — re-derive per .emd, the numbers do not transfer):
#     normalisation : p1–p99.9 percentile clip     (NOT min-max)
#     gaussian sigma: 0.40 nm  = 0.870 px @ 0.45999 nm/px
#     blob threshold: 0.1101   (= 0.16 x this config's contrast p99)
#  Selected by: contrast trend ~0 across the movie (gate) · CNR x1.32 vs raw
#  (objective) · deconvolved blob diameter 2.44 nm vs known 2.4 nm (physical guard).
#
#  WHY NOT min-max: per-frame min-max normalisation made the detector NON-STATIONARY.
#  Raw median drifts 5798 -> 13704 over this movie and yield swung 39-1029 (CV 0.58);
#  births/frame then correlated with detection yield on FIRST DIFFERENCES at r=+0.527
#  (p=2e-6), i.e. the population curve was partly the detector. Percentile clipping
#  removes it: contrast CV 0.054 -> 0.014, and the coupling fell to r=+0.16 (p=0.18).
#
#  NOTE ON GROUND TRUTH: the only defensible GT is the hand annotation in CROP space
#  (PP_tracks_*.csv on HAADF011-070), never registered to the full 1024x1024 frame.
#  The full-frame pair (255,286) is NOT ground truth and nothing here is calibrated
#  against it. Earlier headers claiming "GT-calibrated / GT dumbbell PASSES" were wrong.
#
#  PHYSICAL CRITERIA (unchanged — these are not what was retuned):
#    birth 2.5-6.5 nm | death sep>=7 nm | bonded>=6 fr | mean_sep in [2.5,6.5]
#    HARD per-frame floor sep>=2.4 nm | mean_vc>=0.50 (TRUE velocity)
#  Separation = center-to-center of intensity-weighted centroids.
#  Produces `accepted` (with 'confirmed'), F_, DT for the plotting cell.
# ============================================================
import os, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_log

os.makedirs("Figs", exist_ok=True)      # MUST precede any save (a late makedirs cost a full run)

EMD    = "20250110_HAADF-BF_STEM_150_kx_1106.emd"
EMD_DS = "Data/Image/5c2019b3a90c4fbf8b557aa3f494a535/Data"    # HAADF, NOT 6b581c77 (BF-S)
PX_NM, DT = 0.6505247971734647, 0.79671549

# ---- tag = last 4 digits of the .emd filename (drives every output name) ----
_stem   = os.path.splitext(os.path.basename(EMD))[0]
_grps   = re.findall(r"\d+", _stem)
TAGNAME = _grps[-1][-4:] if _grps else "0000"

SCALE = 0.16260 / PX_NM

# ---- DETECTOR (measured for 1051; see header) ----
GSIG_NM   = 0.60                  # gaussian sigma in NANOMETRES
GSIG      = GSIG_NM / PX_NM       # -> 0.922 px on 1106
NORM_LO, NORM_HI = 1.0, 99.9      # percentile clip
# detection SCALE as a physical particle DIAMETER (2.4 nm) -> blob sigma (px):
MIN_DIAM_NM, MAXSIG_MULT = 2.4, 2.5
MIN_SIG = (MIN_DIAM_NM/2)/np.sqrt(2)/PX_NM      # 1.304 px
MAX_SIG = MAXSIG_MULT*MIN_SIG                   # 3.261 px
THR_FRAC = 0.16                   # per-frame threshold = THR_FRAC * that frame's p99

# ---- tracker geometry: R/RR/LAMBDA on SCALE; MAX_STEP MEASURED (not scaled) ----
R, RR = max(1,round(3*SCALE)), max(2,round(10*SCALE))
MAX_STEP = 10                     # MEASURED p95 of motion (=9.1px; engine default
                                  # round(12*SCALE)=3 covered only the 13th pct)
LAMBDA = 0.003 / SCALE**2

# ---- separation criteria (physical, scale-independent) ----
SEP_MIN, SEP_MAX = 2.5/PX_NM, 6.5/PX_NM        # birth band (mutual-NN)
BLOB = dict(min_sigma=MIN_SIG, max_sigma=MAX_SIG, num_sigma=5)   # threshold applied per-frame in detect()
SEP_DISSOC_NM = 7.0                            # death threshold
MEAN_SEP_MIN, MEAN_SEP_MAX = 2.5, 6.5          # bonded-phase MEAN-sep band
MIN_SEP_NM    = 2.4                            # PER-FRAME physical floor
VC_MIN = 0.50                                  # whole-bond TRUE-velocity corr gate
MIN_BONDED = 6                                 # permanence: >=6 consecutive bonded frames
NPERM, TRACK_LEN = 1000, 12
MATCH_TOL = 3.0 / PX_NM
FDR_Q = 0.05
USE_TRUE_VELOCITY = True        # False reverts to POSITION corr (flow-confounded); if you
                                # flip it, VC_MIN=0.50 is no longer calibrated.
rng = np.random.default_rng(0)

with h5py.File(EMD,"r") as f:
    raw = np.transpose(f[EMD_DS][()],(2,0,1)).astype(np.float32)
F_,Hh,Ww = raw.shape

def norm01(a):
    """p1–p99.9 percentile clip. Replaces min-max, which let a drifting intensity
    range change what `threshold` meant on every frame (see header)."""
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, NORM_LO), np.percentile(a, NORM_HI)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

print(f"[{TAGNAME}] denoising {F_} frames  (norm p{NORM_LO}-p{NORM_HI}, sigma {GSIG_NM} nm = {GSIG:.3f} px, "
      f"blob {MIN_SIG:.2f}-{MAX_SIG:.2f}px, thr {THR_FRAC}*p99, MAX_STEP {MAX_STEP})...")
stack = np.stack([gaussian_filter(norm01(fr), GSIG) for fr in raw])

def euc(a,b): return float(np.hypot(a[0]-b[0],a[1]-b[1]))
def cap_vec(dx,dy,m):
    d=np.hypot(dx,dy)
    if d<=m or d==0: return dx,dy
    s=m/d; return dx*s,dy*s
def refine(clean,cx,cy,r=R,rr=RR,lam=LAMBDA):     # intensity-weighted centroid
    x0,x1=max(0,round(cx-r)),min(Ww,round(cx+r)+1); y0,y1=max(0,round(cy-r)),min(Hh,round(cy+r)+1)
    if x1<=x0 or y1<=y0: return float(np.clip(cx,0,Ww-1)),float(np.clip(cy,0,Hh-1))
    win=clean[y0:y1,x0:x1]; ys,xs=np.mgrid[y0:y1,x0:x1]
    k=np.unravel_index(np.argmax(win-lam*((xs-cx)**2+(ys-cy)**2)),win.shape)
    px,py=x0+k[1],y0+k[0]
    a0,a1=max(0,px-rr),min(Ww,px+rr+1); b0,b1=max(0,py-rr),min(Hh,py+rr+1)
    patch=clean[b0:b1,a0:a1]; w=patch-patch.min()
    if w.sum()<=0: return float(px),float(py)
    ys2,xs2=np.mgrid[b0:b1,a0:a1]
    return float((xs2*w).sum()/w.sum()),float((ys2*w).sum()/w.sum())
def detect(fr):
    # per-frame fractional threshold: 0.16 * this frame's p99 (NOT a fixed absolute value)
    th = THR_FRAC * np.percentile(fr, 99)
    return [(float(x),float(y)) for y,x,s in blob_log(fr, threshold=th, **BLOB)]

# ---- SANITY CHECK: stop here if detection is wrong, not after hours of tracking ----
_n0 = len(detect(stack[0]))
print(f"  frame 0 detections: {_n0}")
print("  (expect a few hundred to low thousands for 1051. If <100 or >5000, re-check")
print("   GSIG_NM / THR_FRAC / MIN_DIAM_NM before letting the run continue.)")

def track_bonded(f0,p1,p2):
    """track from birth until sep>=SEP_DISSOC (death) or end. Returns traj + alive_len + died."""
    p1=refine(stack[f0],*p1); p2=refine(stack[f0],*p2); v1=v2=(0.,0.)
    A=[p1]; B=[p2]; died=False
    for f in range(f0+1,F_):
        vx1,vy1=cap_vec(*v1,MAX_STEP); vx2,vy2=cap_vec(*v2,MAX_STEP)
        n1=refine(stack[f],p1[0]+vx1,p1[1]+vy1); n2=refine(stack[f],p2[0]+vx2,p2[1]+vy2)
        dx1,dy1=cap_vec(n1[0]-p1[0],n1[1]-p1[1],MAX_STEP); dx2,dy2=cap_vec(n2[0]-p2[0],n2[1]-p2[1],MAX_STEP)
        n1=(p1[0]+dx1,p1[1]+dy1); n2=(p2[0]+dx2,p2[1]+dy2); v1=(dx1,dy1); v2=(dx2,dy2); p1,p2=n1,n2
        A.append(n1); B.append(n2)
        if euc(n1,n2)*PX_NM >= SEP_DISSOC_NM:      # DEATH
            died=True; break
    a=np.array(A); b=np.array(B)
    alive_len = len(a)-1 if died else len(a)       # frames with sep<7 (bonded phase)
    return a, b, alive_len, died

def _vel(p): return p if not USE_TRUE_VELOCITY else np.diff(p,axis=0)
def vel_corr(p1,p2):
    # NOTE: np.diff(x,0) is a NO-OP (0 is `n`, not `axis`). The real differencing happens
    # in _vel. This composition is correct but fragile — do not "simplify" it.
    v1=np.diff(_vel(p1),0); v2=np.diff(_vel(p2),0)
    if len(v1)<3: return 0.0
    v1c=v1-v1.mean(0); v2c=v2-v2.mean(0)
    return float((v1c*v2c).sum()/(np.sqrt((v1c**2).sum()*(v2c**2).sum())+1e-9))
def p_shift(p1,p2,nperm=NPERM):
    q1=_vel(p1); q2=_vel(p2)
    v1=np.diff(q1,0); v2=np.diff(q2,0)
    if len(v1)<4: return 1.0
    obs=vel_corr(p1,p2); v1c=v1-v1.mean(0); pref=np.sqrt((v1c**2).sum()); nu=np.empty(nperm)
    for i in range(nperm):
        v2s=np.roll(v2,rng.integers(1,len(v2)-1),axis=0); v2c=v2s-v2s.mean(0)
        nu[i]=(v1c*v2c).sum()/(pref*np.sqrt((v2c**2).sum())+1e-9)
    return (np.sum(nu>=obs)+1)/(nperm+1)
# (the old `persistence()` helper is gone — the 'frac' gate it fed was retired and it
#  was never called, so it only invited the assumption that the gate still ran.)

def evaluate_bonded(a,b,alive_len):
    # permanence + tightness + per-frame floor + whole-bond velocity co-motion.
    if alive_len < MIN_BONDED: return None                       # permanence
    ab, bb = a[:alive_len], b[:alive_len]
    sep=np.hypot(*(ab-bb).T)*PX_NM; mean_sep=float(sep.mean())
    if not (MEAN_SEP_MIN<=mean_sep<=MEAN_SEP_MAX): return None    # tightness (mean)
    if sep.min() < MIN_SEP_NM: return None                        # HARD per-frame floor
    vc=vel_corr(ab,bb)
    return dict(mean_sep=mean_sep, max_sep=float(sep.max()), min_sep=float(sep.min()),
                mean_vc=vc, passes=(vc>=VC_MIN))

# ---------- time-limited occupancy (released after death) ----------
accepted=[]; all_tracked=[]
def claimed_at(pt, frame):
    for T in all_tracked:
        idx=frame-T['seed']
        if 0<=idx<T['alive_len']:                    # only while the dumbbell is ALIVE
            if euc(pt,T['a'][idx])<MATCH_TOL or euc(pt,T['b'][idx])<MATCH_TOL: return True
    return False

seed_frames=range(2, F_-MIN_BONDED)   # 1106: skip the frame-0/1 settling step
n_new=[]; n_claimed=[]
for k in seed_frames:
    pts=detect(stack[k]); n=len(pts)
    if n<2: n_new.append(0); n_claimed.append(0); continue
    P=np.array(pts)                                  # vectorised distance matrix
    Dm=np.hypot(P[:,None,0]-P[None,:,0], P[:,None,1]-P[None,:,1])
    np.fill_diagonal(Dm, np.inf)
    nn=Dm.argmin(1)
    cidx=[(i,nn[i]) for i in range(n) if nn[nn[i]]==i and i<nn[i] and SEP_MIN<=Dm[i,nn[i]]<=SEP_MAX]
    nnew=ncl=0
    for (i,j) in cidx:
        pa,pb=np.array(pts[i]),np.array(pts[j])
        if claimed_at(pa,k) and claimed_at(pb,k):    # both already in an ALIVE dumbbell
            ncl+=1; continue
        a,b,alive_len,died = track_bonded(k,pa,pb)
        all_tracked.append(dict(seed=k,a=a,b=b,alive_len=alive_len)); nnew+=1
        m=evaluate_bonded(a,b,alive_len)
        if m and m['passes']:
            accepted.append(dict(seed=k, pair=(i,j), a=a, b=b, alive_len=alive_len,
                died=died, death_frame=k+alive_len if died else None,
                lifetime_frames=alive_len, lifetime_s=alive_len*DT, **m))
    n_new.append(nnew); n_claimed.append(ncl)
    if k%10==0: print(f"  frame {k:2d}: {n} blobs, {len(cidx)} pairs, {nnew} new, {ncl} claimed  (accepted {len(accepted)})")

# ---------- decoy-FREE FDR: BH on each candidate's whole-bonded-phase circular-shift p ----------
for c in accepted:
    c['p_bonded'] = p_shift(c['a'][:c['alive_len']], c['b'][:c['alive_len']])
def bh(p,q=FDR_Q):
    p=np.asarray(p,float); m=len(p)
    if m==0: return np.zeros(0,bool)
    o=np.argsort(p); th=q*np.arange(1,m+1)/m; bel=p[o]<=th; msk=np.zeros(m,bool)
    if bel.any(): msk[o[:np.max(np.where(bel)[0])+1]]=True
    return msk
conf=bh([c['p_bonded'] for c in accepted])
for c,s in zip(accepted,conf): c['confirmed']=bool(s)

# ---------- results ----------
res=pd.DataFrame([{kk:c[kk] for kk in ['seed','pair','lifetime_frames','lifetime_s','died',
      'mean_sep','min_sep','max_sep','mean_vc']} for c in accepted])
if len(res): res=res.sort_values("lifetime_s",ascending=False).reset_index(drop=True)
nconf=sum(1 for c in accepted if c['confirmed'])
print("\n"+"="*92)
print(f"[{TAGNAME}] DETECTOR: norm p{NORM_LO}-p{NORM_HI} | sigma {GSIG_NM} nm ({GSIG:.3f} px) | "
      f"blob {MIN_SIG:.2f}-{MAX_SIG:.2f}px | thr {THR_FRAC}*p99 | MAX_STEP {MAX_STEP}")
print(f"PHYSICAL criteria: birth 2.5-6.5nm | death sep>={SEP_DISSOC_NM}nm | bonded>={MIN_BONDED}fr | "
      f"mean_sep in [{MEAN_SEP_MIN},{MEAN_SEP_MAX}]nm | HARD per-frame floor sep>={MIN_SEP_NM}nm | "
      f"mean_vc>={VC_MIN} ({'TRUE-velocity' if USE_TRUE_VELOCITY else 'position'})")
print(f"ACCEPTED  (without FDR): {len(res)}   (seed0: {(res.seed==0).sum() if len(res) else 0}, "
      f"later: {(res.seed>0).sum() if len(res) else 0})")
print(f"CONFIRMED (with BH-FDR<{FDR_Q} on circular-shift p): {nconf}")
print("  NOTE: the FDR runs on the SAME circular-shift null the accept gate uses, so it is")
print("        redundant by construction. Do NOT report it as independent confirmation.")
med_s = mean_s = float('nan')
if len(res):
    print(f"  observed dissociation (died): {res.died.sum()}   censored: {(~res.died).sum()}")
    d=res[res.died]
    if len(d):
        mean_s = float(d.lifetime_s.mean()); med_s = float(d.lifetime_s.median())
        print(f"  lifetime accepted (died only): mean {mean_s:.1f}s  median {med_s:.1f}s")
        print(f"  REPORT THE MEDIAN — detector-independent (6.4 s across three detectors). The")
        print(f"  mean moves because the distribution is left-truncated at MIN_BONDED*DT = "
              f"{MIN_BONDED*DT:.2f}s and right-censored ({(~res.died).sum()}).")
    print("\nTOP 20 by lifetime:")
    print(res.head(20).round(3).to_string(index=False))
print("="*92)

# ---------- figure: lifetime distribution + births/deaths + live population ----------
fig,ax=plt.subplots(1,3,figsize=(16,4.5))
if len(res) and res.died.any():
    ax[0].hist(res[res.died].lifetime_s,bins=15,color='steelblue',edgecolor='k')
    ax[0].axvline(mean_s,color='firebrick',lw=2,label=f'mean {mean_s:.1f}s')
    ax[0].axvline(med_s,color='darkgreen',lw=2,ls='--',label=f'median {med_s:.1f}s')
    ax[0].legend(fontsize=8)
ax[0].set_xlabel('bonded lifetime (s)'); ax[0].set_ylabel('# dumbbells'); ax[0].set_title('(a) lifetime distribution')

births=np.zeros(F_); deaths=np.zeros(F_)
for c in accepted:
    births[c['seed']]+=1
    if c['died'] and c['death_frame'] is not None and c['death_frame']<F_: deaths[c['death_frame']]+=1
tt=np.arange(F_)*DT
ax[1].plot(tt,np.cumsum(births),'-',color='green',label='cumulative births')
ax[1].plot(tt,np.cumsum(deaths),'-',color='red',label='cumulative deaths')
ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('count'); ax[1].set_title('(b) dumbbell births vs deaths'); ax[1].legend(fontsize=8)

live=np.cumsum(births)-np.cumsum(deaths)
np.savez(f"Figs/population_curve_{TAGNAME}.npz",
         births=births, deaths=deaths, DT=DT, F=F_, MIN_BONDED=MIN_BONDED)
# NB the shaded "no births possible" marker is removed for presentation, but the effect is
# REAL: no dumbbell can be BORN in the last MIN_BONDED frames, so the terminal decline is
# structural, and the population is still RISING at the end of the observable window.
# Do not claim a peak — keep this caveat in the figure caption.
ax[2].plot(tt,live,'-',color='purple',lw=2)
ax[2].set_xlabel('time (s)'); ax[2].set_ylabel('# live dumbbells'); ax[2].set_title('(c) live dumbbell population')
plt.tight_layout()
_png = f"Figs/population_dumbbell_dynamics_{TAGNAME}.png"
plt.savefig(_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"saved -> {_png}")
print(f"saved -> Figs/population_curve_{TAGNAME}.npz  (for drift_retest_matched.py)")

# ---------- standalone publication panels: (a) lifetime, (c) live population ----------
# Re-draws the two reported panels on their own, without titles, at publication quality.
# Vector PDF is the primary output (resolution-independent); the 600-dpi PNG is for drafts.
def _save_panel(fig, name):
    for ext, kw in (("pdf", {}), ("png", dict(dpi=600))):
        fig.savefig(f"Figs/{name}_{TAGNAME}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"saved -> Figs/{name}_{TAGNAME}.pdf  +  .png (600 dpi)")

# (a) bonded-lifetime distribution
figA, axA = plt.subplots(figsize=(5, 4))
if len(res) and res.died.any():
    axA.hist(res[res.died].lifetime_s, bins=15, color='steelblue', edgecolor='k')
    axA.axvline(mean_s, color='firebrick', lw=2, label=f'mean {mean_s:.1f} s')
    axA.axvline(med_s, color='darkgreen', lw=2, ls='--', label=f'median {med_s:.1f} s')
    axA.legend(frameon=False, fontsize=9)
axA.set_xlabel('bonded lifetime (s)')
axA.set_ylabel('# dumbbells')
_save_panel(figA, "lifetime_distribution")

# (c) live dumbbell population
figC, axC = plt.subplots(figsize=(5, 4))
axC.plot(tt, live, '-', color='purple', lw=2)
axC.set_xlabel('time (s)')
axC.set_ylabel('# live dumbbells')
_save_panel(figC, "live_population")
