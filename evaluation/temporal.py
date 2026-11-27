import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import STRUCTURES, Template, dice, ed_es, grid, gt_sequence, load_config, need, pred_sequence, subjects, voxelize

STRUCTURE_SETS = {"whole": STRUCTURES, "biv": ("LV", "Myo", "RV")}
CFG, TPL, SET, FACES, REGION, SPACING = None, None, None, None, None, None


def jm(X):
    d3 = X[3:] - 3.0 * X[2:-1] + 3.0 * X[1:-2] - X[:-3]
    return float(np.linalg.norm(d3, axis=2).mean())


def sequence_scale(W, S):
    return float(np.mean([W[t].std(0).mean() / S[t].std(0).mean() for t in range(len(W))]))


def areas(X):
    a, b, c = X[FACES[:, 0]], X[FACES[:, 1]], X[FACES[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def regional_strain(X, ed, es):
    n = REGION.max() + 1
    return np.bincount(REGION, areas(X[es]), n) / np.bincount(REGION, areas(X[ed]), n) - 1


def cycle_dsc(X, structure):
    vidx, F = TPL.surfaces[structure]
    A, B = X[0][vidx], X[-1][vidx]
    lo, dims = grid(A, B, SPACING)
    return dice(voxelize(A, F, lo, dims, SPACING), voxelize(B, F, lo, dims, SPACING))


def build_regions(reference, k):
    from scipy.cluster.vq import kmeans2
    faces, owner = TPL.global_faces(STRUCTURE_SETS[SET])
    centre = reference[faces].mean(1)
    region = np.zeros(len(faces), int)
    for i in range(owner.max() + 1):
        j = np.flatnonzero(owner == i)
        region[j] = kmeans2(centre[j], k, minit="++", seed=0)[1] + i * k
    return faces, region


def job(sid):
    P, G = pred_sequence(CFG, sid), gt_sequence(CFG, sid)
    ed, es = ed_es(CFG, TPL, G)
    vidx = np.concatenate([TPL.parts[s] for s in STRUCTURE_SETS[SET]])
    p, g = P[:, vidx], G[:, vidx]
    dp, dg = p - p[ed:ed + 1], g - g[ed:ed + 1]
    r = dict(sid=sid, ed=ed, es=es,
             disp_err=float(np.linalg.norm(dp - dg, axis=2).mean()),
             edes_err=float(np.linalg.norm(dp[es] - dg[es], axis=1).mean()),
             edes_amp_ratio=float(np.linalg.norm(dp[es], axis=1).mean() / np.linalg.norm(dg[es], axis=1).mean()),
             strain_err=float(np.abs(regional_strain(P, ed, es) - regional_strain(G, ed, es)).mean() * 100))
    for s in STRUCTURE_SETS[SET]:
        r[f"cycle_dsc_{s}"], r[f"cycle_dsc_ref_{s}"] = cycle_dsc(P, s), cycle_dsc(G, s)
    r["cycle_dsc"] = float(np.mean([r[f"cycle_dsc_{s}"] for s in STRUCTURE_SETS[SET]]))
    r["cycle_dsc_ref"] = float(np.mean([r[f"cycle_dsc_ref_{s}"] for s in STRUCTURE_SETS[SET]]))
    placements = [("", P, G)]
    if CFG.get("pred_frame_key") and CFG.get("gt_frame_mesh"):
        Pf, Gf = pred_sequence(CFG, sid, "pred_frame_key"), gt_sequence(CFG, sid, "gt_frame_mesh")
        placements.append(("_seq", sequence_scale(P, Pf) * Pf, sequence_scale(G, Gf) * Gf))
    for suffix, X, Y in placements:
        for part, idx in TPL.parts.items():
            r[f"jm{suffix}_{part}"], r[f"jm_ref{suffix}_{part}"] = jm(X[:, idx]), jm(Y[:, idx])
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--parts", choices=tuple(STRUCTURE_SETS), default="whole")
    ap.add_argument("--build-regions", dest="reference")
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    CFG, SET = load_config(a.config), a.parts
    TPL = Template(CFG)
    SPACING = float(CFG.get("voxel_mm", 1.0))
    path = need(CFG, "strain_regions").format(parts=SET)
    if a.reference:
        f, reg = build_regions(np.load(a.reference).astype(np.float64), int(CFG.get("strain_regions_per_structure", 6)))
        np.savez(path, faces=f, region=reg)
        print(f"{len(f)} faces, {reg.max() + 1} regions", flush=True)
    else:
        if not a.tag:
            ap.error("--tag is required")
        out = need(CFG, "out_dir")
        z = np.load(path)
        FACES, REGION = z["faces"], z["region"]
        assert (FACES == TPL.global_faces(STRUCTURE_SETS[SET])[0]).all()
        with Pool(a.workers) as p:
            df = pd.DataFrame(p.map(job, subjects(CFG, a.subjects), chunksize=1))
        os.makedirs(out, exist_ok=True)
        df.to_csv(f"{out}/{a.tag}_temporal_{SET}_subjects.csv", index=False)
        num = df.drop(columns=["sid", "ed", "es"])
        summary = pd.DataFrame({"mean": num.mean(), "std": num.std(), "count": num.count()})
        summary.to_csv(f"{out}/{a.tag}_temporal_{SET}_summary.csv")
        print(summary.round(4).to_string(), flush=True)
