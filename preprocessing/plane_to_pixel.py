import argparse
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd

from cardiac_frame import load_frames
from common import LAX_VIEWS, lax_file, load_config, need, sax_files, subjects

N_PLANES = 16
CFG, GEOMETRY, FRAMES = None, None, None


def plane_to_pixel(A, crop_i, crop_j, lo, ratio, offset):
    u, v, o = A[:3, 0], A[:3, 1], A[:3, 3]
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    I = np.r_[u / (u @ u), -(u @ o) / (u @ u)]
    J = np.r_[v / (v @ v), -(v @ o) / (v @ v)]
    row = J * ratio[1]
    row[3] += (-crop_j - lo[1] + .5) * ratio[1] - .5 + offset[1]
    col = I * ratio[2]
    col[3] += (-crop_i - lo[2] + .5) * ratio[2] - .5 + offset[2]
    return np.stack([row, col, np.r_[n, -(n @ o)]])


def sax_slice_index(k, lo, ratio, offset):
    return int(round((k - lo[0] + .5) * ratio[0] - .5 + offset[0]))


def frame_planes(sid, t):
    G = np.zeros((N_PLANES, 3, 4), np.float32)
    valid = np.zeros(N_PLANES, np.float32)
    z = np.zeros(N_PLANES, np.float32)
    for view in LAX_VIEWS + ("sax",):
        g = GEOMETRY.get((sid, t, view))
        if g is None:
            continue
        crop_i, crop_j, lo, ratio, offset = g
        if view != "sax":
            slot = LAX_VIEWS.index(view)
            A = nib.load(lax_file(CFG, "lax_image", sid, t, view)).affine
            G[slot], valid[slot] = plane_to_pixel(A, crop_i, crop_j, lo, ratio, offset), 1.
            continue
        slot = len(LAX_VIEWS)
        for k, f in enumerate(sax_files(CFG, "sax_image_glob", sid, t).values()):
            zz = sax_slice_index(k, lo, ratio, offset)
            if 0 <= zz < int(CFG.get("n_sax_slices", 12)) and slot < N_PLANES:
                G[slot], valid[slot], z[slot] = plane_to_pixel(nib.load(f).affine, crop_i, crop_j, lo, ratio, offset), 1., zz
                slot += 1
    return G, valid, z


def job(sid):
    ts = [t for t in range(int(CFG.get("n_frames", 25))) if (sid, t) in FRAMES]
    if not ts:
        return sid, 0
    planes = [frame_planes(sid, t) for t in ts]
    np.savez(f"{need(CFG, 'out_dir')}/geometry/{sid}.npz", t=np.array(ts),
             cardiac_frame=np.stack([FRAMES[(sid, t)] for t in ts]).astype(np.float32),
             plane_to_pixel=np.stack([p[0] for p in planes]), plane_valid=np.stack([p[1] for p in planes]),
             sax_slice=np.stack([p[2] for p in planes]))
    return sid, len(ts)


def read_geometry(path):
    df = pd.read_csv(path)
    return {(str(r.subject), int(r.t), str(r.view)): (r.crop_i, r.crop_j, np.array([r.lo_z, r.lo_y, r.lo_x], float),
                                                      np.array([r.ratio_z, r.ratio_y, r.ratio_x], float),
                                                      np.array([r.offset_z, r.offset_y, r.offset_x], float))
            for r in df.itertuples()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    CFG = load_config(a.config)
    GEOMETRY = read_geometry(need(CFG, "input_geometry_csv"))
    FRAMES = load_frames(f"{need(CFG, 'out_dir')}/cardiac_frame_pred.csv")
    os.makedirs(f"{need(CFG, 'out_dir')}/geometry", exist_ok=True)
    with Pool(a.workers) as p:
        done = p.map(job, subjects(CFG))
    print(f"{sum(n > 0 for _, n in done)} subjects, {sum(n for _, n in done)} frames")
