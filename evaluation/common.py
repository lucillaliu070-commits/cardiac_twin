import glob
import re

import numpy as np
import pandas as pd
import yaml

STRUCTURES = ("LV", "Myo", "RV", "LA", "RA")
LABEL_NAMES = {"LV": "LV", "LV_MYO": "Myo", "RV": "RV", "RA": "RA", "LA": "LA"}
LABEL_ID = {"Myo": 1, "LV": 2, "RV": 3, "RA": 4, "LA": 5}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def need(cfg, key):
    value = cfg.get(key)
    if value in (None, ""):
        raise ValueError(f"config: '{key}' is empty")
    return value


def n_frames(cfg):
    return int(cfg.get("n_frames", 25))


def subjects(cfg, only=None):
    df = pd.read_csv(need(cfg, "split_csv"))
    names = sorted(df[df.split.str.lower() == str(cfg.get("split", "test")).lower()].subject.astype(str))
    return [s for s in names if not only or s in set(only)]


class Template:
    def __init__(self, cfg):
        names = np.array([LABEL_NAMES[x] for x in open(need(cfg, "vertex_labels")).read().split()])
        self.n = len(names)
        self.faces = np.load(need(cfg, "faces")).astype(np.int64)
        self.parts = {"whole": np.arange(self.n), **{s: np.flatnonzero(names == s) for s in STRUCTURES},
                      "BiV": np.flatnonzero(np.isin(names, ("LV", "Myo", "RV")))}
        self.surfaces = {s: self.surface(self.parts[s]) for s in STRUCTURES}
        self.surfaces["whole"] = (self.parts["whole"], self.faces)
        self.face_label = np.array([LABEL_ID[x] for x in names])[self.faces[:, 0]]

    def surface(self, vidx):
        inside = np.zeros(self.n, bool)
        inside[vidx] = True
        local = -np.ones(self.n, np.int64)
        local[vidx] = np.arange(len(vidx))
        return vidx, local[self.faces[inside[self.faces].all(1)]]

    def global_faces(self, structures):
        faces, owner = [], []
        for i, s in enumerate(structures):
            inside = np.zeros(self.n, bool)
            inside[self.parts[s]] = True
            f = self.faces[inside[self.faces].all(1)]
            faces.append(f)
            owner.append(np.full(len(f), i))
        return np.concatenate(faces), np.concatenate(owner)


def pred_sequence(cfg, sid, key="pred_key"):
    with np.load(need(cfg, "pred_mesh").format(sid=sid)) as z:
        return z[need(cfg, key)].astype(np.float64)


def gt_sequence(cfg, sid, key="gt_mesh"):
    return np.stack([np.load(need(cfg, key).format(sid=sid, t=t)) for t in range(n_frames(cfg))]).astype(np.float64)


def volume_ml(W, F):
    v0, v1, v2 = W[F[:, 0]], W[F[:, 1]], W[F[:, 2]]
    return float(abs(np.einsum("ij,ij->i", v0, np.cross(v1 - v0, v2 - v0)).sum() / 6) / 1000)


def ed_es(cfg, tpl, G):
    vidx, F = tpl.surfaces["LV"]
    volume = np.array([volume_ml(X[vidx], F) for X in G])
    ed = int(np.argmax(volume)) if cfg.get("ed_rule", "first") == "max_volume" else 0
    return ed, int(np.argmin(volume))


def voxelize(W, F, origin, dims, spacing):
    import vtk
    from vtk.util import numpy_support as ns
    points = vtk.vtkPoints()
    points.SetData(ns.numpy_to_vtk(np.ascontiguousarray(W, np.float64), deep=True))
    conn = np.hstack([np.full((len(F), 1), 3, np.int64), F.astype(np.int64)]).ravel()
    cells = vtk.vtkCellArray()
    cells.SetCells(len(F), ns.numpy_to_vtkIdTypeArray(conn, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    image = vtk.vtkImageData()
    image.SetOrigin(*origin)
    image.SetSpacing(spacing, spacing, spacing)
    image.SetDimensions(*dims)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(1)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly)
    stencil.SetOutputOrigin(*origin)
    stencil.SetOutputSpacing(spacing, spacing, spacing)
    stencil.SetOutputWholeExtent(image.GetExtent())
    stencil.Update()
    apply = vtk.vtkImageStencil()
    apply.SetInputData(image)
    apply.SetStencilConnection(stencil.GetOutputPort())
    apply.ReverseStencilOff()
    apply.SetBackgroundValue(0)
    apply.Update()
    return ns.vtk_to_numpy(apply.GetOutput().GetPointData().GetScalars()).astype(bool)


def grid(A, B, spacing):
    lo = np.minimum(A.min(0), B.min(0)) - 2.0
    hi = np.maximum(A.max(0), B.max(0)) + 2.0
    return lo, tuple(int(np.ceil(x)) + 1 for x in (hi - lo) / spacing)


def dice(a, b):
    return float(2.0 * (a & b).sum() / (a.sum() + b.sum()))


def sax_files(cfg, key, sid, t):
    pattern = re.compile(need(cfg, "sax_depth_regex"))
    files = {}
    for f in glob.glob(need(cfg, key).format(sid=sid, t=t)):
        m = pattern.search(f)
        if m:
            files[int(m.group(1))] = f
    return dict(sorted(files.items()))


def summarize(df, columns, by):
    per_subject = df.groupby(list(by) + ["sid"])[columns].mean()
    out = per_subject.groupby(list(by)).agg(["mean", "std", "count"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    return per_subject.reset_index(), out.reset_index()
