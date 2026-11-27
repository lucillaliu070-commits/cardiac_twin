import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff

from common import STRUCTURES, Template, dice, ed_es, grid, gt_sequence, load_config, need, pred_sequence, subjects, summarize, voxelize

ALL_METRICS = ("mae", "mse", "assd", "cd", "hd", "nsd", "dice")
SUB = 3
BARY = np.array([(i, j, SUB - i - j) for i in range(SUB + 1) for j in range(SUB + 1 - i)], float) / SUB
CFG, TPL, METRICS, PARTS, TAUS, SPACING = None, None, None, None, None, None


def hausdorff(x, g):
    x, g = x.astype(np.float32), g.astype(np.float32)
    return float(max(directed_hausdorff(x, g)[0], directed_hausdorff(g, x)[0]))


def vertex_area(W, F):
    a = 0.5 * np.linalg.norm(np.cross(W[F[:, 1]] - W[F[:, 0]], W[F[:, 2]] - W[F[:, 0]]), axis=1)
    w = np.zeros(len(W))
    np.add.at(w, F.ravel(), np.repeat(a / 3.0, 3))
    return w


def dense(W, F):
    return np.einsum("kb,fbd->fkd", BARY, W[F]).reshape(-1, 3)


def nsd(A, B, F):
    wa, wb = vertex_area(A, F), vertex_area(B, F)
    da = cKDTree(dense(B, F)).query(A)[0]
    db = cKDTree(dense(A, F)).query(B)[0]
    return {f"nsd_{t:g}": float((wa[da <= t].sum() + wb[db <= t].sum()) / (wa.sum() + wb.sum())) for t in TAUS}


def frame_metrics(P, G):
    rows = {part: {} for part in PARTS}
    for part in PARTS:
        idx = TPL.parts[part]
        x, g = P[idx], G[idx]
        r = rows[part]
        if {"mae", "mse"} & METRICS:
            d = x - g
            r["mae"], r["mse"] = float(np.abs(d).mean()), float((d ** 2).mean())
        if {"assd", "cd"} & METRICS:
            dp, dg = cKDTree(g).query(x)[0], cKDTree(x).query(g)[0]
            r["assd"], r["cd"] = float(.5 * (dp.mean() + dg.mean())), float((dp ** 2).mean() + (dg ** 2).mean())
        if "hd" in METRICS:
            r["hd"] = hausdorff(x, g)
        if "nsd" in METRICS and part in TPL.surfaces:
            vidx, F = TPL.surfaces[part]
            r.update(nsd(P[vidx], G[vidx], F))
    if "dice" in METRICS:
        lo, dims = grid(P, G, SPACING)
        union_p = union_g = None
        for s in STRUCTURES:
            vidx, F = TPL.surfaces[s]
            a, b = voxelize(P[vidx], F, lo, dims, SPACING), voxelize(G[vidx], F, lo, dims, SPACING)
            if s in rows:
                rows[s]["dice"] = dice(a, b)
            union_p = a if union_p is None else union_p | a
            union_g = b if union_g is None else union_g | b
        if "whole" in rows:
            rows["whole"]["dice"] = dice(union_p, union_g)
    return rows


def job(sid):
    P, G = pred_sequence(CFG, sid), gt_sequence(CFG, sid)
    ed, es = ed_es(CFG, TPL, G)
    out = []
    for t in range(len(G)):
        for part, r in frame_metrics(P[t], G[t]).items():
            out.append(dict(sid=sid, t=t, phase="ED" if t == ed else "ES" if t == es else "", part=part, **r))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--metrics", default=",".join(ALL_METRICS))
    ap.add_argument("--parts", default="whole,LV,Myo,RV,LA,RA,BiV")
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    CFG, METRICS, PARTS = load_config(a.config), set(a.metrics.split(",")), a.parts.split(",")
    TPL = Template(CFG)
    TAUS = tuple(float(x) for x in CFG.get("nsd_tau_mm", [1.0, 2.0, 3.0]))
    SPACING = float(CFG.get("voxel_mm", 1.0))
    out = need(CFG, "out_dir")
    with Pool(a.workers) as p:
        df = pd.DataFrame([r for rows in p.map(job, subjects(CFG, a.subjects), chunksize=1) for r in rows])
    os.makedirs(out, exist_ok=True)
    df.to_csv(f"{out}/{a.tag}_surface_frames.csv", index=False)
    cols = [c for c in df.columns if c not in ("sid", "t", "phase", "part")]
    for name, sub in (("all", df), ("edes", df[df.phase != ""])):
        per_subject, summary = summarize(sub, cols, ["part"])
        per_subject.to_csv(f"{out}/{a.tag}_surface_subjects_{name}.csv", index=False)
        summary.to_csv(f"{out}/{a.tag}_surface_summary_{name}.csv", index=False)
        print(name, summary.round(4).to_string(index=False), sep="\n", flush=True)
