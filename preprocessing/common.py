import glob
import re

import numpy as np
import pandas as pd
import yaml

LAX_VIEWS = ("2ch", "3ch", "4ch")
STRUCTURES = {"LV_MYO": 1, "LV": 2, "RV": 3, "RA": 4, "LA": 5}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def need(cfg, key):
    value = cfg.get(key)
    if value in (None, ""):
        raise ValueError(f"config: '{key}' is empty")
    return value


def subjects(cfg, split=None):
    df = pd.read_csv(need(cfg, "split_csv"))
    if split is not None:
        df = df[df.split.str.lower() == split]
    return sorted(df.subject.astype(str))


def lax_file(cfg, key, sid, t, view):
    return need(cfg, key).format(sid=sid, t=t, view=view)


def sax_files(cfg, key, sid, t):
    pattern = re.compile(need(cfg, "sax_depth_regex"))
    files = {}
    for f in glob.glob(need(cfg, key).format(sid=sid, t=t)):
        m = pattern.search(f)
        if m:
            files[int(m.group(1))] = f
    return dict(sorted(files.items()))


def vertex_structure(cfg):
    names = np.array(open(need(cfg, "vertex_labels")).read().split())
    return np.array([STRUCTURES[x] for x in names])


def to_world(mesh, frame):
    return mesh @ frame[:3, :3].T + frame[:3, 3]
