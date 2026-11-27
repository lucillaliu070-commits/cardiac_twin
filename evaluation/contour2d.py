import argparse
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.draw import line
from skimage.measure import find_contours
from skimage.morphology import binary_dilation, disk

from common import Template, gt_sequence, load_config, need, pred_sequence, sax_files, subjects, summarize

CFG, TPL, LABELS = None, None, False


def cut(W, A):
    n = A[:3, 2] / np.linalg.norm(A[:3, 2])
    d = (W - A[:3, 3]) @ n
    d[d == 0] = 1e-9
    df = d[TPL.faces]
    m = (df.min(1) < 0) & (df.max(1) > 0)
    if not m.any():
        return np.zeros((0, 2, 2)), np.zeros(0, int)
    f, dd = TPL.faces[m], df[m]
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cross = dd[:, a] * dd[:, b] < 0
        w = dd[:, a] / (dd[:, a] - dd[:, b])
        p = W[f[:, a]] + (W[f[:, b]] - W[f[:, a]]) * w[:, None]
        pts.append(np.where(cross[:, None], p, np.nan))
    P = np.stack(pts, 1)
    seg = np.stack([P[i][~np.isnan(P[i][:, 0])][:2] for i in range(len(P))])
    ij = (np.c_[seg.reshape(-1, 3), np.ones(len(seg) * 2)] @ np.linalg.inv(A).T)[:, :2].reshape(-1, 2, 2)
    return ij, TPL.face_label[m]


def raster_bf(Sp, Sr, pixel=1.0, tol_mm=2.0, margin=5.0):
    pts = np.concatenate([Sp.reshape(-1, 2), Sr.reshape(-1, 2)])
    x0, y0 = pts.min(0) - margin
    x1, y1 = pts.max(0) + margin
    W, H = int(np.ceil((x1 - x0) / pixel) + 1), int(np.ceil((y1 - y0) / pixel) + 1)

    def draw(S):
        m = np.zeros((H, W), bool)
        c = np.round((S[..., 0] - x0) / pixel).astype(int)
        r = np.round((y1 - S[..., 1]) / pixel).astype(int)
        for (r0, r1), (c0, c1) in zip(r, c):
            rr, cc = line(r0, c0, r1, c1)
            m[rr.clip(0, H - 1), cc.clip(0, W - 1)] = True
        return m

    mp, mr = draw(Sp), draw(Sr)
    tol = disk(max(1, int(np.ceil(tol_mm / pixel))))
    precision = (mp & binary_dilation(mr, tol)).sum() / mp.sum()
    recall = (mr & binary_dilation(mp, tol)).sum() / mr.sum()
    return 2 * precision * recall / (precision + recall) if precision + recall else np.nan


def pooled(Sa, Sb):
    P, R = (np.unique(s.reshape(-1, 2).round(5), axis=0) for s in (Sa, Sb))
    mcd = .5 * (cKDTree(R).query(P)[0].mean() + cKDTree(P).query(R)[0].mean())
    return float(mcd), float(100 * raster_bf(Sa, Sb))


def label_segments(L, k, sp):
    out = [np.stack([c[:-1], c[1:]], 1) for c in ((c - 1) * sp for c in find_contours(np.pad(L == k, 1).astype(float), 0.5)) if len(c) > 1]
    return np.concatenate(out) if out else np.zeros((0, 2, 2))


def inside(ij, s, k, H, W):
    seg = ij[s == k]
    return ((seg[..., 0] >= 0) & (seg[..., 0] <= H - 1) & (seg[..., 1] >= 0) & (seg[..., 1] <= W - 1)).all(1).sum()


def planes(sid, t):
    depths = set(int(d) for d in CFG.get("contour_sax_depths", [4, 5, 6, 7, 8]))
    out = [(v.upper(), -1, need(CFG, "lax_label").format(sid=sid, t=t, view=v)) for v in ("2ch", "3ch", "4ch")]
    out += [("SAX", d, f) for d, f in sax_files(CFG, "sax_label_glob", sid, t).items() if d in depths]
    return [p for p in out if os.path.exists(p[2])]


def frame_rows(sid, t, X, G):
    rows = []
    for view, depth, f in planes(sid, t):
        header = nib.load(f)
        A = header.affine
        sp = np.linalg.norm(A[:3, :2], axis=0)
        cx, cg = cut(X, A), cut(G, A)
        base = dict(sid=sid, t=t, view=view, depth=depth)
        if min(len(cx[0]), len(cg[0])) >= 5:
            mcd, bf = pooled(cx[0] * sp, cg[0] * sp)
            rows.append(dict(base, comparison="pred_vs_ref", mcd=mcd, bf=bf))
        if not LABELS:
            continue
        L = np.asarray(header.dataobj)
        L = (L[..., 0] if L.ndim == 3 else L).astype(np.int16)
        if not L.any():
            continue
        H, W = L.shape
        keep = [k for k in range(1, 6) if (L == k).sum() >= 10 and all(inside(*c, k, H, W) >= 5 for c in (cx, cg))]
        if not keep:
            continue
        lab = np.concatenate([label_segments(L, k, sp) for k in keep])
        for name, (ij, s) in (("pred", cx), ("ref", cg)):
            mcd, bf = pooled(ij[np.isin(s, keep)] * sp, lab)
            rows.append(dict(base, comparison=f"{name}_vs_label", mcd=mcd, bf=bf, n_structures=len(keep)))
    return rows


def job(sid):
    P, G = pred_sequence(CFG, sid), gt_sequence(CFG, sid)
    return [r for t in range(len(G)) for r in frame_rows(sid, t, P[t], G[t])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    CFG, LABELS = load_config(a.config), a.labels
    TPL = Template(CFG)
    out = need(CFG, "out_dir")
    with Pool(a.workers) as p:
        df = pd.DataFrame([r for rows in p.map(job, subjects(CFG, a.subjects), chunksize=1) for r in rows])
    os.makedirs(out, exist_ok=True)
    df.to_csv(f"{out}/{a.tag}_contour2d_items.csv", index=False)
    items = df.groupby(["comparison", "view"])[["bf", "mcd"]].agg(["mean", "std", "count"])
    items.columns = [f"{c}_{s}" for c, s in items.columns]
    items.reset_index().to_csv(f"{out}/{a.tag}_contour2d_summary.csv", index=False)
    per_subject, _ = summarize(df, ["bf", "mcd"], ["comparison", "view"])
    per_subject.to_csv(f"{out}/{a.tag}_contour2d_subjects.csv", index=False)
    print(items.round(3).to_string(), flush=True)
