"""Quality metrics for PIV output folders (no ground truth needed).

For every analysed frame pair (near-duplicate pairs from out-dup.dat are skipped) the
"moving" cells are those with phi_bar >= thr and a non-zero vector. On them we report:
  outlier  : fraction of vectors failing the normalized median test (Westerweel & Scarano
             2005, eps=0.1, residual > 2) -- random/noisy vectors fail it, coherent flow passes
  limit    : fraction of vectors sitting at the edge of the +-wROI search window
  coh      : |<(u,v)>| / <|V|>  (1 = all vectors identical, ~0 = isotropic noise)
  n        : number of moving cells
"""
import numpy as np, os, sys
from scipy import ndimage as ndi

def load(d, f):
    g = lambda n: np.loadtxt(f"{d}/{n}_{f:04d}.dat")
    return g("out-u"), g("out-v"), g("out-phi")

def nmt_outliers(U, V, M, eps=0.1, thr=2.0, min_nb=3):
    """Normalized median test on the cells in mask M. Returns (outlier_mask, tested_mask)."""
    ny, nx = U.shape
    Uu = np.where(M, U, np.nan); Vv = np.where(M, V, np.nan)
    def stack(A):
        P = np.pad(A, 1, constant_values=np.nan)
        return np.stack([P[1+dy:1+dy+ny, 1+dx:1+dx+nx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    SU, SV = stack(Uu), stack(Vv)
    nb = np.sum(~np.isnan(SU), axis=0)
    with np.errstate(all="ignore"):
        import warnings; warnings.simplefilter("ignore")
        mu = np.nanmedian(SU, axis=0); mv = np.nanmedian(SV, axis=0)
        ru = np.abs(U - mu) / (np.nanmedian(np.abs(SU - mu), axis=0) + eps)
        rv = np.abs(V - mv) / (np.nanmedian(np.abs(SV - mv), axis=0) + eps)
    r = np.hypot(ru, rv)
    tested = M & (nb >= min_nb)
    return (r > thr) & tested, tested

def blob_filter(mask, min_cells):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0: return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.isin(lab, 1 + np.where(sizes >= min_cells)[0])
    return keep

def evaluate(d, f0, f1, thr, wROI=32, verbose=False, blob_min=0):
    dup = np.loadtxt(f"{d}/out-dup.dat").astype(int)
    frames = np.loadtxt(f"{d}/out-frames.dat").astype(int)
    rows = []
    for f, isdup in zip(frames, dup):
        if f < f0 or f > f1 or isdup: continue
        U, V, P = load(d, f)
        M = (P >= thr)
        if blob_min: M = blob_filter(M, blob_min)
        M = M & ((U != 0) | (V != 0))
        if M.sum() < 30: continue
        out, tested = nmt_outliers(U, V, M)
        mag = np.hypot(U[M], V[M])
        lim = (np.maximum(np.abs(U[M]), np.abs(V[M])) >= wROI - 1).mean()
        coh = np.hypot(U[M].mean(), V[M].mean()) / mag.mean()
        rows.append((f, M.sum(), out.sum() / max(tested.sum(), 1), lim, coh, mag.mean()))
        if verbose: print(rows[-1])
    R = np.array(rows)
    return dict(n=R[:, 1].mean(), outlier=R[:, 2].mean(), limit=R[:, 3].mean(), coh=R[:, 4].mean(), meanV=R[:, 5].mean(), nframes=len(R))

if __name__ == "__main__":
    d, f0, f1, thr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    print(evaluate(d, f0, f1, thr, verbose=True))
