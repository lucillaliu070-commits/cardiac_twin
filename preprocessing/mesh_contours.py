import argparse
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np

from cardiac_frame import load_frames
from common import LAX_VIEWS, lax_file, load_config, need, sax_files, subjects, to_world, vertex_structure

N_STRUCTURES = 5
CFG, FACES, FACE_STRUCTURE, GT_FRAMES = None, None, None, None


def cut(W, A, faces, face_structure):
    n = A[:3, 2] / np.linalg.norm(A[:3, 2])
    d = (W - A[:3, 3]) @ n
    d[d == 0] = 1e-9
    df = d[faces]
    m = (df.min(1) < 0) & (df.max(1) > 0)
    if not m.any():
        return np.zeros((0, 2, 2)), np.zeros(0, int)
    f, dd = faces[m], df[m]
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cross = dd[:, a] * dd[:, b] < 0
        w = dd[:, a] / (dd[:, a] - dd[:, b])
        p = W[f[:, a]] + (W[f[:, b]] - W[f[:, a]]) * w[:, None]
        pts.append(np.where(cross[:, None], p, np.nan))
    P = np.stack(pts, 1)
    seg = np.stack([P[i][~np.isnan(P[i][:, 0])][:2] for i in range(len(P))])
    ij = (np.c_[seg.reshape(-1, 3), np.ones(len(seg) * 2)] @ np.linalg.inv(A).T)[:, :2].reshape(-1, 2, 2)
    return ij, face_structure[m]


def cut_points(W, A, step):
    ij, structure = cut(W, A, FACES, FACE_STRUCTURE)
    out = {}
    for k in np.unique(structure):
        seg = ij[structure == k]
        P = np.concatenate([np.c_[seg[:, e], np.zeros(len(seg)), np.ones(len(seg))] @ A.T for e in (0, 1)], 1)[:, [0, 1, 2, 4, 5, 6]]
        a, b = P[:, :3], P[:, 3:]
        n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / step)).astype(int)
        r = np.repeat(np.arange(len(a)), n)
        m = np.concatenate([np.arange(x) for x in n])
        out[int(k)] = a[r] + (b[r] - a[r]) * ((m + .5) / n[r])[:, None]
    return out


def label_planes(sid, t):
    planes = [(v, -1, lax_file(CFG, "lax_label", sid, t, v)) for v in LAX_VIEWS]
    planes += [("sax", depth, f) for depth, f in sax_files(CFG, "sax_label_glob", sid, t).items()]
    return [p for p in planes if os.path.exists(p[2])]


def gt_mesh(sid, t):
    W = np.load(need(CFG, "gt_mesh").format(sid=sid, t=t)).astype(np.float64)
    if CFG.get("gt_mesh_space", "world") == "frame":
        W = to_world(W, GT_FRAMES[(sid, t)])
    return W


def subject(sid):
    step = float(CFG.get("contour_step_mm", 0.5))
    planes, pts, plab = [], [], []
    for t in range(int(CFG.get("n_frames", 25))):
        W = gt_mesh(sid, t)
        for view, depth, f in label_planes(sid, t):
            header = nib.load(f)
            A = header.affine.astype(np.float64)
            points = {k: v for k, v in cut_points(W, A, step).items() if len(v) >= 5}
            if not points:
                continue
            start = sum(len(p) for p in pts)
            for k in sorted(points):
                pts.append(points[k].astype(np.float32))
                plab.append(np.full(len(points[k]), k, np.int8))
            planes.append(dict(p_time=t, p_view=(LAX_VIEWS + ("sax",)).index(view), p_depth=depth, p_c=A[:3, 3],
                               p_n=A[:3, 2] / np.linalg.norm(A[:3, 2]), p_ainv=np.linalg.inv(A)[:2],
                               p_shape=header.shape[:2], p_start=start, p_count=sum(len(p) for p in pts) - start))
    if not planes:
        return sid, 0
    z = {k: np.array([p[k] for p in planes]) for k in planes[0]}
    z = {k: v.astype(np.float32) if v.dtype == np.float64 else v.astype(np.int32) for k, v in z.items()}
    z["pts"], z["plab"] = np.concatenate(pts), np.concatenate(plab)
    np.savez_compressed(f"{need(CFG, 'out_dir')}/contours/{sid}.npz", **z)
    return sid, len(planes)


def loss_inputs(contours_npz, frames, views=(0, 1, 2, 3), device="cpu"):
    import torch
    z = dict(np.load(contours_npz)) if isinstance(contours_npz, str) else contours_npz
    index = {int(t): i for i, t in enumerate(frames)}
    sel = np.flatnonzero(np.isin(z["p_time"], list(index)) & np.isin(z["p_view"], views))
    counts = np.zeros((len(sel), N_STRUCTURES), np.int64)
    chunks = []
    for q, p in enumerate(sel):
        s0, n = int(z["p_start"][p]), int(z["p_count"][p])
        lab, xyz = z["plab"][s0:s0 + n], z["pts"][s0:s0 + n]
        chunks.append([xyz[lab == s + 1] for s in range(N_STRUCTURES)])
        counts[q] = [len(c) for c in chunks[-1]]
    contours = np.zeros((len(sel), N_STRUCTURES, max(int(counts.max(initial=0)), 1), 3), np.float32)
    for q, per_structure in enumerate(chunks):
        for s, c in enumerate(per_structure):
            contours[q, s, :len(c)] = c
    t = lambda x, dtype=torch.float32: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    planes = dict(frame=t([index[int(x)] for x in z["p_time"][sel]], torch.long), view=t(z["p_view"][sel], torch.long),
                  normal=t(z["p_n"][sel]), centre=t(z["p_c"][sel]), to_pixel=t(z["p_ainv"][sel]), size=t(z["p_shape"][sel]))
    return planes, t(contours), t(counts, torch.long)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subjects", nargs="*")
    a = ap.parse_args()
    CFG = load_config(a.config)
    FACES = np.load(need(CFG, "faces")).astype(np.int64)
    FACE_STRUCTURE = vertex_structure(CFG)[FACES[:, 0]]
    if CFG.get("gt_mesh_space", "world") == "frame":
        GT_FRAMES = load_frames(f"{need(CFG, 'out_dir')}/cardiac_frame_gt.csv")
    exclude = set(CFG.get("contour_exclude_subjects") or [])
    todo = [s for s in subjects(CFG, "train") if s not in exclude]
    if a.subjects:
        todo = [s for s in todo if s in set(a.subjects)]
    os.makedirs(f"{need(CFG, 'out_dir')}/contours", exist_ok=True)
    with Pool(a.workers) as p:
        done = p.map(subject, todo, chunksize=2)
    print(f"{sum(n > 0 for _, n in done)} / {len(todo)} training subjects, {sum(n for _, n in done)} planes")
