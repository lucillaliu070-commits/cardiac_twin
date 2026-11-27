import argparse
import json
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.spatial import distance
from skimage.measure import label as connected_components

from common import lax_file, load_config, need, subjects

CFG = None
LABEL = None


def largest_component(segmentation):
    labels = connected_components(segmentation)
    assert labels.max() != 0
    return labels == np.argmax(np.bincount(labels.flat)[1:]) + 1


def landmark_gen(path, output_size=160, affine_mtx=None):
    seg = nib.load(path).get_fdata()
    seg_LV = largest_component((seg == 2).astype(seg.dtype)).astype(seg.dtype)
    seg_RV = largest_component((seg == 3).astype(seg.dtype)).astype(seg.dtype)
    seg_RA = largest_component((seg == 4).astype(seg.dtype)).astype(seg.dtype)
    seg_LA = largest_component((seg == 5).astype(seg.dtype)).astype(seg.dtype)
    seg_binary = largest_component((seg != 0).astype(seg.dtype)).astype(seg.dtype)
    seg_LA_dilation = scipy.ndimage.binary_dilation(seg_LA).astype(seg.dtype)
    mvc = np.mean(np.where(np.multiply(seg_LV, seg_LA_dilation) == 1), axis=1)
    LV_points = np.array(np.where(seg_LV == 1)).T
    apex = LV_points[np.argmax(np.linalg.norm(LV_points - mvc, axis=1))]
    lvc = np.mean(np.where(seg_LV == 1), axis=1)
    rvc = np.mean(np.where(seg_RV == 1), axis=1)
    seg_RA_dilation = scipy.ndimage.binary_dilation(seg_RA).astype(seg.dtype)
    tvc = np.mean(np.where(np.multiply(seg_RV, seg_RA_dilation) == 1), axis=1)
    coh = np.mean(np.where(seg_binary == 1), axis=1)
    L = np.ones((4, 6))
    L[:3, :] = np.array([apex, mvc, tvc, lvc, rvc, coh]).T
    if affine_mtx is None:
        affine_mtx = nib.load(path).get_qform()
    Lp = (affine_mtx @ L)[:3]
    apex_ps, mvc_ps, tvc_ps, coh_ps = Lp[:, 0], Lp[:, 1], Lp[:, 2], Lp[:, 5]
    vec_apex_mvc = apex - mvc
    surface = scipy.ndimage.binary_dilation(seg_binary).astype(seg.dtype) - seg_binary
    fg = np.array(np.where(surface == 1)).T
    far = fg[np.argmin((fg - mvc) @ vec_apex_mvc)]
    max_distance = distance.euclidean((affine_mtx @ np.append(far, 1))[:3], apex_ps)
    v3 = mvc_ps - apex_ps
    vt = tvc_ps - mvc_ps
    v2 = np.cross(v3, vt)
    v2f = v2 / np.linalg.norm(v2)
    v1 = np.cross(v2, v3)
    v1f = v1 / np.linalg.norm(v1)
    v3b = np.cross(v1, v2)
    v3f = v3b / np.linalg.norm(v3b)
    new_space = (np.round(max_distance) * 1.3) / output_size
    R = np.zeros((3, 3))
    R[:, 0], R[:, 1], R[:, 2] = v1f * new_space, v2f * new_space, v3f * new_space
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = coh_ps - R @ np.array([80, 80, 80])
    return dict(new_affine=A, new_space=new_space, max_distance=max_distance, landmarks_ps=Lp)


def job(key):
    sid, t = key
    row = dict(subject=sid, t=t)
    try:
        if LABEL == "pred":
            qform = nib.load(lax_file(CFG, "lax_image", sid, t, "4ch")).get_qform()
            frame = landmark_gen(need(CFG, "four_ch_label_pred").format(sid=sid, t=t), affine_mtx=qform)
        else:
            frame = landmark_gen(lax_file(CFG, "lax_label", sid, t, "4ch"))
    except Exception as e:
        return dict(row, error=str(e)[:200])
    if not np.isfinite(frame["new_affine"]).all():
        return dict(row, error="non-finite landmarks")
    return dict(row, cardiac_frame=json.dumps(np.round(frame["new_affine"], 8).tolist()),
                new_space=frame["new_space"], max_distance=frame["max_distance"], source="own")


def borrow_nearest(df):
    ok = df.cardiac_frame.notna()
    for i in df.index[~ok]:
        cand = df[(df.subject == df.at[i, "subject"]) & ok]
        if cand.empty:
            df.at[i, "source"] = "none"
            continue
        j = (cand.t - df.at[i, "t"]).abs().idxmin()
        for c in ("cardiac_frame", "new_space", "max_distance"):
            df.at[i, c] = df.at[j, c]
        df.at[i, "source"] = f"nearest_t{int(df.at[j, 't'])}"
    return df


def load_frames(path):
    df = pd.read_csv(path).dropna(subset=["cardiac_frame"])
    return {(str(r.subject), int(r.t)): np.array(json.loads(r.cardiac_frame)) for r in df.itertuples()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", choices=("pred", "gt"), required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    CFG, LABEL = load_config(a.config), a.label
    for key in ("out_dir", "lax_image", "four_ch_label_pred") if LABEL == "pred" else ("out_dir", "lax_label"):
        need(CFG, key)
    keys =[(s, t) for s in subjects(CFG) for t in range(int(CFG.get("n_frames", 25)))]
    with Pool(a.workers) as p:
        df = pd.DataFrame(p.map(job, keys, chunksize=8))
    for c in ("cardiac_frame", "new_space", "max_distance", "source", "error"):
        if c not in df:
            df[c] = np.nan
    df["source"] = df.source.astype(object)
    if LABEL == "pred":
        df = borrow_nearest(df)
    out = need(CFG, "out_dir")
    os.makedirs(out, exist_ok=True)
    df.to_csv(f"{out}/cardiac_frame_{LABEL}.csv", index=False)
    print(f"{len(df)} frames, {int(df.cardiac_frame.notna().sum())} with a frame, sources {df.source.value_counts().to_dict()}")
