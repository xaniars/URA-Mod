"""Synthetic ground-truth test for piv_movie_v12.

gen : build a lossless 2-frames-per-case video from a real pre-processed jet frame (texture)
      with known displacement fields and fresh Gaussian noise in every frame.
eval: compare piv output (u,v in px, y pointing up) with the truth at the cell centers.
Frames 2k, 2k+1 form pair k  (run the PIV with every=2).
"""
import numpy as np, cv2, sys, json

H = W = 768
SIGMA_NOISE = 0.65    # gray levels, per frame (measured from the quiet region of the real video)
GAIN = 8.0            # gray levels -> 8-bit counts around 128 (as in out-pre_*.png)

def texture(png="results/w2_p3/out-pre_1418.png", blur=1.0):
    t = (cv2.imread(png, 0).astype(np.float32) - 128.0) / GAIN
    return cv2.GaussianBlur(t, (0, 0), blur)

def cases():
    cs = []
    for k in range(8):                                   # uniform shifts, uniformly spread fractional parts
        cs.append(("uniform_%d" % k, 5.0 + k / 8.0, -3.0 + ((3 * k) % 8) / 8.0))
    cs.append(("uniform_large", 20.37, -12.61))
    cs.append(("shear", "shear"))
    cs.append(("vortex", "vortex"))
    return cs

def field(case, X, Y):
    """Displacement (u, v) [px, y up] of a feature located at (X, Y) in image 1."""
    if case[0].startswith("uniform"):
        return np.full_like(X, case[1]), np.full_like(X, case[2])
    if case[1] == "shear":
        return 6.0 + 5.0 * np.sin(2 * np.pi * Y / 420.0), 2.0 * np.sin(2 * np.pi * X / 520.0)
    if case[1] == "vortex":                              # solid-body core + decaying outside, centre (330, 400)
        cx, cy, G, a = 330.0, 400.0, 9.0, 90.0
        dx, dy = X - cx, Y - cy; r2 = dx * dx + dy * dy + 1e-9
        f = (1 - np.exp(-r2 / a ** 2)) / r2 * a ** 2 / a   # bounded
        return -G * dy * f / a * 1.0 + 4.0, G * dx * f / a * 1.0

def gen(out="processed/synth.avi", truth="results/synth_truth.npz", seed=1):
    rng = np.random.default_rng(seed)
    T = texture()
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float32)
    Xu, Yu = cols, (H - 1) - rows                        # y-up coordinates of each pixel
    w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"FFV1"), 60, (W, H), True)
    names = []
    for c in cases():
        # I2(x) = T(x - d): iterate to evaluate the field at the source position
        sx, sy = Xu.copy(), Yu.copy()
        for _ in range(4):
            u, v = field(c, sx, sy); sx, sy = Xu - u, Yu - v
        I1 = T
        I2 = cv2.remap(T, sx.astype(np.float32), ((H - 1) - sy).astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        for I in (I1, I2):
            n = I + rng.normal(0, SIGMA_NOISE, I.shape).astype(np.float32)
            g = np.clip(np.round(128 + GAIN * n), 0, 255).astype(np.uint8)
            w.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        names.append(c[0])
    w.release()
    np.savez(truth, names=np.array(names))
    print("wrote", out, len(names), "pairs")

def evaluate(d, wROI=32):
    X = np.loadtxt(f"{d}/out-x.dat"); Y = np.loadtxt(f"{d}/out-y.dat")
    T = texture()
    lstd = cv2.GaussianBlur(T * T, (0, 0), 16) ** 0.5     # local texture strength
    cyi = ((H - 1) - Y).astype(int); cxi = X.astype(int)
    inside = ((X - 384) ** 2 + (Y - 384) ** 2 < 330 ** 2) & (lstd[cyi, cxi] > 0.6)
    res = {}
    for k, c in enumerate(cases()):
        f = 2 * k
        U = np.loadtxt(f"{d}/out-u_{f:04d}.dat"); V = np.loadtxt(f"{d}/out-v_{f:04d}.dat")
        tu, tv = field(c, X, Y)
        eu, ev = U - tu, V - tv
        m = inside
        err = np.hypot(eu[m], ev[m])
        res[c[0]] = dict(n=int(m.sum()), rms=float(np.sqrt(np.mean(err ** 2))), median=float(np.median(err)),
                         bad=float(np.mean(err > 1.0)), rms_good=float(np.sqrt(np.mean(err[err <= 1.0] ** 2))) if (err <= 1).any() else np.nan,
                         bias_u=float(np.median(eu[m])), bias_v=float(np.median(ev[m])),
                         tu=float(np.mean(tu[m])), tv=float(np.mean(tv[m])))
    return res

if __name__ == "__main__":
    if sys.argv[1] == "gen": gen()
    else: print(json.dumps(evaluate(sys.argv[2]), indent=1))
