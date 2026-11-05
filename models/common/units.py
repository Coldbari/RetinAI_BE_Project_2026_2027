"""Evaluation units per source (PROTOCOL_V2 "Units"). Outputs never carry these keys, only aggregates.

Multi-View = eye (patient, Left/Right) via rop_locked_eval.eye_frame; ROP-VL = eye-examination
(infant, exam date, eye) from the filename; Ostrava = examination session (id, device, serie);
Shenzhen = image (no bagging).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_OST = re.compile(r"/(\d+)_D(\d)_S(\d+)_(\d+)\.jpg$")
_RVL = re.compile(r"/(\d+)_(\d{4}-\d{2}-\d{2})_([LR])_(\d+)\.jpg$")


def unit_keys(df: pd.DataFrame) -> np.ndarray:
    """One string key per row of `df` (columns image_path, patient_id, source), row order kept."""
    keys = np.empty(len(df), dtype=object)
    mv = (df.source == "multiview").values
    if mv.any():
        from scripts.rop_locked_eval import eye_frame        # the repo's own eye definition
        e = eye_frame(df[mv].reset_index(drop=True))
        assert (e.image_path.values == df.image_path.values[mv]).all()
        keys[mv] = [f"multiview|{p}|{x}" for p, x in zip(e.patient_id, e.eye)]
    for i, (src, p, pid) in enumerate(zip(df.source.values, df.image_path.values, df.patient_id.values)):
        if src == "ostrava":
            m = _OST.search(p)
            keys[i] = f"ostrava|{pid}|D{m.group(2)}|S{m.group(3)}"
        elif src == "ropvl":
            m = _RVL.search(p)
            keys[i] = f"ropvl|{pid}|{m.group(2)}|{m.group(3)}"
        elif src == "shenzhen":
            keys[i] = f"shenzhen|{p}"
        elif src != "multiview":
            raise ValueError(f"unknown source {src!r}")
    return keys
