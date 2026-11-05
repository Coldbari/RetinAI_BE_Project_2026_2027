"""Step 1 — bag manifest.  .venv/bin/python experiments/eye_spike/s1_bags.py

Writes (no filenames / dates / patient ids):
  bags.csv               one row per bag: bag_id, patient_hash, source, fold (-1 = dev), label, n_images
  cache/membership.npz   image row -> bag index for the pool and for dev, + frame order
  bags_summary.json      counts per source/fold, size distribution, integrity checks
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import CACHE, CLASSES, HERE, ROOT, load_dev, load_pool, bag_key, opaque


def build(df: pd.DataFrame, sz_eye: dict, prefix: str):
    keys, order = zip(*[bag_key(r, sz_eye) for r in df.itertuples()])
    df = df.assign(_key=list(keys), _order=list(order))
    # integrity: a bag never spans folds, never spans patients
    g = df.groupby("_key")
    assert (g.fold.nunique() == 1).all(), "a bag spans folds"
    assert (g.patient_id.nunique() == 1).all(), "a bag spans patients"
    bags = g.agg(source=("source", "first"), fold=("fold", "first"),
                 patient_id=("patient_id", "first"), label=("label", "max"),
                 n_images=("label", "size"), n_labels=("label", "nunique"),
                 first_row=("label", lambda s: s.index.min())).sort_values("first_row")
    bags["bag_idx"] = np.arange(len(bags))
    bags["bag_id"] = [f"{prefix}{opaque(k)}" for k in bags.index]
    bags["patient_hash"] = [opaque("pt|" + p, 10) for p in bags.patient_id]
    img_bag = df._key.map(bags.bag_idx).values.astype(np.int32)
    return bags, img_bag, np.asarray(df._order, dtype=np.int32)


def main():
    sx = pd.read_excel(ROOT / "data/rop/shenzhen/zip information.xlsx", sheet_name="Sheet1")
    sz_eye = dict(zip(sx.img_name, sx.eye))
    pool, dev = load_pool(), load_dev()
    pb, p_img_bag, p_order = build(pool, sz_eye, "c")
    db, d_img_bag, d_order = build(dev, sz_eye, "d")

    np.savez(CACHE / "membership.npz", pool_img_bag=p_img_bag, pool_order=p_order,
             dev_img_bag=d_img_bag, dev_order=d_order,
             pool_bag_patient=np.array(list(pb.patient_hash), dtype="U16"),
             dev_bag_patient=np.array(list(db.patient_hash), dtype="U16"))
    cols = ["bag_id", "patient_hash", "source", "fold", "label", "n_images"]
    out = pd.concat([pb[cols], db[cols]], ignore_index=True)
    out["label_name"] = out.label.map(dict(enumerate(CLASSES)))
    out.to_csv(HERE / "bags.csv", index=False)

    def size_dist(b):
        n = b.n_images
        return {"n_bags": int(len(b)), "n_images": int(n.sum()),
                "single_image_bags": int((n == 1).sum()),
                "median": float(n.median()), "p90": float(n.quantile(.9)), "max": int(n.max()),
                "bags_gt16": int((n > 16).sum()),
                "mixed_label_bags": int((b.n_labels > 1).sum())}

    summ = {"pool": {}, "dev": size_dist(db)}
    for s, b in pb.groupby("source"):
        summ["pool"][s] = size_dist(b) | {
            "per_fold": b.fold.value_counts().sort_index().to_dict(),
            "label_counts": {CLASSES[c]: int(v) for c, v in b.label.value_counts().sort_index().items()},
            "median_size_by_label": {CLASSES[c]: float(v) for c, v in
                                     b.groupby("label").n_images.median().items()}}
    summ["pool"]["ALL"] = size_dist(pb) | {
        "per_fold": pb.fold.value_counts().sort_index().to_dict(),
        "label_counts": {CLASSES[c]: int(v) for c, v in pb.label.value_counts().sort_index().items()},
        "label_by_fold": {int(k): {CLASSES[c]: int(v) for c, v in g.label.value_counts().sort_index().items()}
                          for k, g in pb.groupby("fold")},
        "multi_image_diseased_bags": int(((pb.n_images > 1) & (pb.label > 0)).sum())}
    summ["dev"]["label_counts"] = {CLASSES[c]: int(v) for c, v in db.label.value_counts().sort_index().items()}
    summ["checks"] = {"bags_span_folds": 0, "bags_span_patients": 0,
                      "patients_span_folds": int((pool.groupby('patient_id').fold.nunique() > 1).sum()),
                      "dev_patients_in_pool": len(set(dev.patient_id) & set(pool.patient_id))}
    (HERE / "bags_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
