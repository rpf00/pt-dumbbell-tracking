# ============================================================
#  MULTI-FRAME dumbbell discovery - BONDED-LIFETIME population engine
#
#  Detects Pt nanoparticles in an LP-STEM .emd stack, pairs them, tracks each
#  pair to dissociation, and gates on physical criteria to produce dumbbell
#  lifetime and live-population statistics.
#
#  ALL PARAMETERS LIVE IN constants.txt (same folder). Edit that file, not this
#  one. The shipped defaults are the locked operating point for movie 1053.
#
#  THE CORE PRINCIPLE
#    Physical criteria are constant in NANOMETRES; every pixel-space parameter
#    is re-derived per movie. The detector block in constants.txt must be
#    re-fitted for each new acquisition (stationarity gate / CNR objective /
#    deconvolved-diameter guard). The physical criteria block must not be.
#
#  WHY PERCENTILE NORMALISATION, NOT MIN-MAX
#    The specimen brightens during acquisition (raw median drifted 5798 ->
#    13704 on 1053). Under per-frame min-max, a fixed threshold is a DIFFERENT
#    detector on the first and last frame: yield swung 39-1029 (CV 0.58) and
#    dumbbell births correlated with detection yield on first differences at
#    r=+0.527 (p=2e-6) - i.e. the population curve was partly the detector.
#    Percentile clipping removes it (contrast CV 0.054 -> 0.014, coupling
#    r=+0.16, p=0.39).
#
#  GROUND TRUTH
#    The only defensible ground truth is the hand annotation in CROP space
#    (PP_tracks_*.csv on HAADF011-070), which was never registered to the full
#    1024x1024 frame. It validates the TRACKER (0.57 nm localisation, 0.33 nm
#    separation, 0 identity swaps), not the detector. Nothing here is
#    calibrated against a full-frame object.
#
#  Separation = centre-to-centre of intensity-weighted centroids.
#  Produces `accepted` (with 'confirmed'), F_, DT for the plotting section.
# ============================================================
import os, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_log

os.makedirs("Figs", exist_ok=True)      # MUST precede any save (a late makedirs cost a full run)

# ============================================================
#  LOAD CONSTANTS  -  everything tunable comes from constants.txt
# ============================================================
CONST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constants.txt")

def load_constants(path):
    """Parse `KEY = value` lines, ignoring blank lines and # comments."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"constants.txt not found at {path}\n"
            "It must sit beside population_engine_optimal.py.")
    cfg = {}
    for ln, line in enumerate(open(path), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{ln}: expected 'KEY = value', got: {line!r}")
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg

CFG = load_constants(CONST_FILE)

def _get(key, cast=str):
    if key not in CFG:
        raise KeyError(f"'{key}' is missing from constants.txt")
    raw = CFG[key]
    if cast is bool:
        if raw.lower() not in ("true", "false"):
            raise ValueError(f"constants.txt: {key} must be True or False, got {raw!r}")
        return raw.lower() == "true"
    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"constants.txt: {key} = {raw!r} is not a valid {cast.__name__}")

def _choice(key, options):
    v = _get(key).lower()
    if v not in options:
        raise ValueError(f"constants.txt: {key} must be one of {options}, got {v!r}")
    return v

# ---- file ----
EMD       = _get("EMD")
EMD_DS    = _get("EMD_DS")
PX_NM     = _get("PX_NM", float)
DT        = _get("DT", float)

# ---- tag = last 4 digits of the .emd filename (drives every output name) ----
_stem   = os.path.splitext(os.path.basename(EMD))[0]
_grps   = re.findall(r"\d+", _stem)
TAGNAME = _grps[-1][-4:] if _grps else "0000"

SCALE = 0.16260 / PX_NM        # crop-space reference -> this movie's pixel space

# ---- detector ----
NORM_LO   = _get("NORM_LO", float)
NORM_HI   = _get("NORM_HI", float)
GSIG_NM   = _get("GSIG_NM", float)
GSIG      = GSIG_NM / PX_NM                      # gaussian sigma in px

BLOB_MODE = _choice("BLOB_MODE", ("scale", "diameter"))
if BLOB_MODE == "scale":
    MIN_SIG = _get("MIN_SIG_MULT", float) * SCALE
    MAX_SIG = _get("MAX_SIG_MULT", float) * SCALE
else:                                            # size from a physical NP diameter
    MIN_DIAM_NM = _get("MIN_DIAM_NM", float)
    MIN_SIG = (MIN_DIAM_NM / 2) / np.sqrt(2) / PX_NM
    MAX_SIG = _get("MAXSIG_MULT", float) * MIN_SIG

THRESHOLD_MODE = _choice("THRESHOLD_MODE", ("absolute", "fractional"))
THRESHOLD = _get("THRESHOLD", float)             # used when mode = absolute
THR_FRAC  = _get("THR_FRAC", float)              # used when mode = fractional
_THR_DESC = (f"{THRESHOLD}" if THRESHOLD_MODE == "absolute" else f"{THR_FRAC}*p99")

# ---- tracker geometry ----
R  = max(1, round(_get("R_MULT",  float) * SCALE))
RR = max(2, round(_get("RR_MULT", float) * SCALE))
_ms = _get("MAX_STEP")
MAX_STEP = max(2, round(12 * SCALE)) if _ms.lower() == "auto" else int(_ms)
LAMBDA = _get("LAMBDA_REF", float) / SCALE**2

# ---- physical criteria (nm -> px where the code needs px) ----
SEP_MIN, SEP_MAX = _get("SEP_MIN_NM", float)/PX_NM, _get("SEP_MAX_NM", float)/PX_NM
BLOB = dict(min_sigma=MIN_SIG, max_sigma=MAX_SIG, num_sigma=_get("NUM_SIGMA", int))
SEP_DISSOC_NM = _get("SEP_DISSOC_NM", float)
MEAN_SEP_MIN, MEAN_SEP_MAX = _get("MEAN_SEP_MIN", float), _get("MEAN_SEP_MAX", float)
MIN_SEP_NM = _get("MIN_SEP_NM", float)
VC_MIN     = _get("VC_MIN", float)
MIN_BONDED = _get("MIN_BONDED", int)
MATCH_TOL  = _get("MATCH_TOL_NM", float) / PX_NM

# ---- statistics ----
USE_TRUE_VELOCITY = _get("USE_TRUE_VELOCITY", bool)   # False = POSITION corr
                                                      # (flow-confounded; VC_MIN would
                                                      #  no longer be calibrated)
NPERM      = _get("NPERM", int)
TRACK_LEN  = _get("TRACK_LEN", int)
FDR_Q      = _get("FDR_Q", float)
rng = np.random.default_rng(_get("RNG_SEED", int))

print(f"[{TAGNAME}] constants loaded from {CONST_FILE}")
if not os.path.exists(EMD):
    raise FileNotFoundError(
        f"{EMD} not found in {os.getcwd()}.\n"
        "The raw stacks are archived on Zenodo - see data/README.md. Either put the\n"
        "file here or give a full path in constants.txt.")

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
      f"blob {MIN_SIG:.2f}-{MAX_SIG:.2f}px, thr {_THR_DESC}, MAX_STEP {MAX_STEP})...")
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
    """THRESHOLD_MODE=absolute uses a single fixed value; =fractional recomputes the
    threshold as THR_FRAC x this frame's own p99. The fractional form is the safer
    default on a movie whose contrast is not perfectly stationary."""
    th = THRESHOLD if THRESHOLD_MODE == "absolute" else THR_FRAC * np.percentile(fr, 99)
    return [(float(x),float(y)) for y,x,s in blob_log(fr, threshold=th, **BLOB)]

# ---- SANITY CHECK: stop here if detection is wrong, not after hours of tracking ----
_n0 = len(detect(stack[0]))
print(f"  frame 0 detections: {_n0}")
print("  (expect a few hundred to low thousands. If <100 or >5000, re-check GSIG_NM /")
print("   the threshold / the blob scale in constants.txt before letting this run on.)")

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

seed_frames=range(0, F_-MIN_BONDED)
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
      f"blob {MIN_SIG:.2f}-{MAX_SIG:.2f}px | thr {_THR_DESC} | MAX_STEP {MAX_STEP}")
print(f"PHYSICAL criteria: birth {SEP_MIN*PX_NM:.1f}-{SEP_MAX*PX_NM:.1f}nm | death sep>={SEP_DISSOC_NM}nm | bonded>={MIN_BONDED}fr | "
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
        print(f"  REPORT THE MEDIAN — it is detector-independent (unchanged across detectors")
        print(f"  varying the accepted count by 2.2x, and across movies). The")
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
print(f"saved -> Figs/population_curve_{TAGNAME}.npz")

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
