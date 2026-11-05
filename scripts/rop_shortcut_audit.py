"""Shortcut audit for the 6-class ROP staging model. Run BEFORE trusting any number.

    .venv/bin/python scripts/rop_shortcut_audit.py
    .venv/bin/python scripts/rop_shortcut_audit.py --checkpoint results/experiment_.../best.pt

Five gates. Any FAIL means the headline metric is measuring something other than ROP.

  G1  patient leakage      no patient_id appears on both sides of any split
  G2  duplicate leakage    no image in the held-out hospital is a near-copy of a train image
  G3  site decodability    how well (W,H,filesize) alone identify the source hospital
  G4  metadata baseline    a classifier that never sees a pixel — the BAR the model must beat
  G5  linear site probe    is the hospital linearly decodable from penultimate embeddings

WHY G4 REPLACED THE ORIGINAL GATE. The plan first said "a dimensions-only classifier must score
near chance". That gate is vacuous: it can only fail, never inform, because we already KNOW
dimensions carry site (the binary ROP screener scored AUC 0.911 on dimensions alone). The
useful question is not "does the shortcut exist" — it does — but "does the image model beat
it". So G4 does not pass or fail on its own; it emits a number that later gets compared against
the trained model's score. A model that cannot beat its own metadata has learned nothing.

WHY G5 RUNS ON CV FOLDS, NOT ON test_dev. The held-out hospital is a SINGLE site, so "predict
the source" there is a constant function and trivially perfect — the probe would be measuring
nothing. Source is only decodable-or-not where several sources coexist, which is the training
pool. G5 therefore probes a CV val fold (Ostrava + Shenzhen + ROP-VL).

test_locked is never read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results" / "prepared" / "rop_staging"
OUT = ROOT / "results" / "rop" / "staging_shortcut_audit.json"
CLASSES = ["normal", "stage_1", "stage_2", "stage_3", "stage_4_5", "arop"]

# 256-bit dHash, the same function and the same calibrated threshold used to de-duplicate
# Shenzhen against ROP-VL in build_rop_staging_manifest.py. Calibration there: byte-identical
# pairs sit at d=0 and genuinely different images never came closer than 6, so 4 separates
# them with margin. The 64-bit variant is NOT safe here — it matched a ROP-VL "Stage 2" to a
# Shenzhen "Normal" at distance 0.
POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int32)
DHASH_THR = 4


def dhash256(path, size=16):
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def meta_features(paths):
    """Native width/height/aspect/filesize — everything obtainable WITHOUT decoding pixels."""
    rows = []
    for p in paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
            size = Path(p).stat().st_size
        except Exception:
            w = h = size = 0
        rows.append((w, h, w / h if h else 0.0, size))
    return np.asarray(rows, dtype=np.float64)


# ── G1 ────────────────────────────────────────────────────────────────────────
def gate_patient_leakage(report):
    ok = True
    print("\nG1  patient leakage")
    for f in range(5):
        df = pd.read_csv(D / f"fold_{f}.csv")
        tr = set(df[df.split == "train"].patient_id)
        va = set(df[df.split == "val"].patient_id)
        bad = tr & va
        ok &= not bad
        print(f"      fold_{f}  train {len(tr):>4} pts  val {len(va):>4} pts  overlap {len(bad)}")

    pool = pd.concat([pd.read_csv(D / f"fold_{f}.csv") for f in range(5)])
    pool_pts = set(pool.patient_id)
    for name in ("test_dev", "test_locked"):
        # NOTE: reading only the patient_id column of test_locked — no labels, no images.
        te = set(pd.read_csv(D / f"{name}.csv").patient_id)
        bad = pool_pts & te
        ok &= not bad
        print(f"      train-pool vs {name:<12} overlap {len(bad)}")
    report["G1_patient_leakage"] = {"pass": bool(ok)}
    print(f"      -> {'PASS' if ok else 'FAIL'}")
    return ok


# ── G2 ────────────────────────────────────────────────────────────────────────
def gate_duplicate_leakage(report):
    print("\nG2  duplicate leakage (train pool vs held-out hospital)")
    pool = pd.concat([pd.read_csv(D / f"fold_{f}.csv") for f in range(5)])
    tr = sorted(set(pool.image_path))
    te = sorted(set(pd.read_csv(D / "test_dev.csv").image_path))

    tr_md5 = {md5(p) for p in tr}
    exact = sum(md5(p) in tr_md5 for p in te)
    print(f"      exact md5 matches            {exact}")

    HA = np.stack([dhash256(p) for p in tr])
    HB = np.stack([dhash256(p) for p in te])
    near, worst = 0, 256
    for i in range(0, len(HB), 128):
        d = POP[np.bitwise_xor(HB[i:i + 128, None, :], HA[None, :, :])].sum(-1).min(axis=1)
        near += int((d <= DHASH_THR).sum())
        worst = min(worst, int(d.min()))
    print(f"      dHash-256 pairs at d<={DHASH_THR}         {near}   (closest pair d={worst})")
    ok = exact == 0 and near == 0
    report["G2_duplicate_leakage"] = {"pass": bool(ok), "exact_md5": exact,
                                      "near_dupes": near, "closest_distance": worst}
    print(f"      -> {'PASS' if ok else 'FAIL'}")
    return ok


# ── G3 / G4 ───────────────────────────────────────────────────────────────────
def gate_metadata(report):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    print("\nG3  site decodability from metadata alone (diagnostic, not pass/fail)")
    pool = pd.concat([pd.read_csv(D / f"fold_{f}.csv").assign(fold=f) for f in range(5)])
    # reset_index: the concat leaves duplicate index labels, which crosstab refuses to reindex.
    pool = pool.drop_duplicates("image_path").reset_index(drop=True)
    Xp = meta_features(pool.image_path.tolist())

    # Grouped by patient so a hospital is not identified by having memorised one infant.
    from sklearn.model_selection import GroupKFold
    src = pool.source.values
    pred = np.empty_like(src)
    for tr_i, te_i in GroupKFold(n_splits=5).split(Xp, src, groups=pool.patient_id.values):
        sc = StandardScaler().fit(Xp[tr_i])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(Xp[tr_i]), src[tr_i])
        pred[te_i] = clf.predict(sc.transform(Xp[te_i]))
    site_bacc = balanced_accuracy_score(src, pred)
    chance = 1.0 / len(set(src))
    print(f"      sources: {sorted(set(src))}")
    print(f"      balanced acc {site_bacc:.3f}  (chance {chance:.3f})")
    print("      ^ this is the size of the shortcut the model must not take")
    # CLASS-SOURCE CONFOUNDING. Even a model that ignores dimensions can pick up site if a
    # class only ever appears at one hospital: "recognise ROP-VL" then scores stage_4_5
    # perfectly without learning a retinal detachment. This is not fixable by augmentation —
    # it is a property of what was collected — so it is recorded as a claim limit, not a gate.
    ct = pd.crosstab(pool.cls6, pool.source)
    single = {c: ct.columns[ct.loc[c] > 0].tolist()
              for c in ct.index if int((ct.loc[c] > 0).sum()) == 1}
    if single:
        print("      class-source confounding — classes present at only ONE site:")
        for c, s in single.items():
            print(f"        {c:<10} {int(ct.loc[c].sum()):>4} images, all from {s[0]}")
        print("        ^ performance on these may be site recognition; do not headline it")
    report["G3_site_decodability"] = {
        "balanced_accuracy": round(float(site_bacc), 4), "chance": round(chance, 4),
        "n_sources": len(set(src)),
        "single_source_classes": {c: s[0] for c, s in single.items()},
    }

    print("\nG4  metadata-only staging baseline — the BAR (no pixels are ever decoded)")
    # TWO learners, and the bar is the STRONGER of them. A linear model on four features
    # cannot express "width==1600 and height==1200 => this camera => mostly stage 2", but a
    # tree can, and that carving is exactly how the binary ROP screener reached AUC 0.911 on
    # dimensions alone. Reporting only the linear number would set an artificially low bar and
    # let a model that learned nothing clear it.
    from sklearn.ensemble import RandomForestClassifier

    def learners():
        return {"linear": ("scale", LogisticRegression(max_iter=2000, class_weight="balanced")),
                "tree": ("raw", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                       class_weight="balanced", random_state=0,
                                                       n_jobs=-1))}

    def fit_predict(mode, clf, Xtr, ytr, Xte):
        if mode == "scale":
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        return clf.fit(Xtr, ytr).predict(Xte)

    y = pool.cls6.values
    dev = pd.read_csv(D / "test_dev.csv")
    Xd = meta_features(dev.image_path.tolist())
    scores = {}
    for tag, (mode, clf) in learners().items():
        oof = np.empty_like(y)
        for f in range(5):
            m = pool.fold.values == f
            oof[m] = fit_predict(mode, clf, Xp[~m], y[~m], Xp[m])
        dev_pred = fit_predict(mode, clf, Xp, y, Xd)
        scores[tag] = {
            "cv_macro_f1": float(f1_score(y, oof, average="macro", labels=CLASSES,
                                          zero_division=0)),
            "cv_balanced_accuracy": float(balanced_accuracy_score(y, oof)),
            "dev_macro_f1": float(f1_score(dev.cls6.values, dev_pred, average="macro",
                                           labels=CLASSES, zero_division=0)),
            "dev_balanced_accuracy": float(balanced_accuracy_score(dev.cls6.values, dev_pred)),
            "dev_distinct_predictions": int(len(set(dev_pred))),
        }
        s = scores[tag]
        print(f"      {tag:<7} CV macro-F1 {s['cv_macro_f1']:.3f}  bAcc "
              f"{s['cv_balanced_accuracy']:.3f}   |   held-out macro-F1 "
              f"{s['dev_macro_f1']:.3f}  bAcc {s['dev_balanced_accuracy']:.3f}  "
              f"({s['dev_distinct_predictions']} distinct pred)")

    bar_cv = max(s["cv_macro_f1"] for s in scores.values())
    bar_dev = max(s["dev_macro_f1"] for s in scores.values())
    print(f"      BAR: image model must beat CV macro-F1 {bar_cv:.3f} "
          f"and held-out macro-F1 {bar_dev:.3f}")
    report["G4_metadata_baseline"] = {
        "learners": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                         for kk, vv in v.items()} for k, v in scores.items()},
        "bar_cv_macro_f1": round(bar_cv, 4), "bar_dev_macro_f1": round(bar_dev, 4),
        "note": "the trained image model must EXCEED these to have learned anything from pixels",
    }
    print("      -> no pass/fail; the image model is compared against these numbers")
    return True


# ── G5 ────────────────────────────────────────────────────────────────────────
def gate_site_probe(report, checkpoint, arch, fold=0):
    """Fit a linear source classifier on the model's penultimate embeddings."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(ROOT))
    from models.common.architectures import build_model
    from models.common.config import load_config
    from models.common.datasets import FundusDataset
    from models.common.preprocessing import build_transforms

    print(f"\nG5  linear site probe on penultimate embeddings (fold_{fold} val)")
    cfg = load_config(ROOT / "configs" / "rop_staging.yaml")
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("model", state)
    if any(k.startswith("backbone.") for k in state):
        # A structured (CORN + AP-ROP + DANN) checkpoint. Probe its BACKBONE features —
        # the same penultimate embedding the flat probe reads, so before/after DANN
        # numbers are measured at the identical point in the network.
        from models.common.structured import StructuredROPModel
        n_sites = state["site_head.3.bias"].shape[0]
        wrapper = StructuredROPModel(arch, n_sites=n_sites, pretrained=False)
        missing, unexpected = wrapper.load_state_dict(state, strict=True), None
        model = wrapper.backbone
        print(f"      structured checkpoint (n_sites={n_sites}) — probing backbone features")
    else:
        model = build_model(arch, int(cfg.data.num_classes), pretrained=False)
        model.load_state_dict(state, strict=False)
        # Replace the classifier with Identity so forward() returns the pooled features.
        for attr in ("classifier", "fc", "head"):
            if hasattr(model, attr):
                setattr(model, attr, torch.nn.Identity())
                break
    model.eval()

    df = pd.read_csv(D / f"fold_{fold}.csv")
    df = df[df.split == "val"].reset_index(drop=True)
    ds = FundusDataset(df, split=None, transform=build_transforms(cfg, split="val"),
                       image_size=int(cfg.preprocess.image_size))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def embed(net):
        net.to(dev).eval()
        feats = []
        with torch.no_grad():
            for x, _ in DataLoader(ds, batch_size=32, num_workers=2):
                feats.append(net(x.to(dev)).flatten(1).cpu().numpy())
        return np.concatenate(feats)

    def probe(X, mask=None):
        """Patient-grouped linear decodability of `source` from embeddings."""
        Xs = X if mask is None else X[mask]
        s = df.source.values if mask is None else df.source.values[mask]
        g = df.patient_id.values if mask is None else df.patient_id.values[mask]
        if len(set(s)) < 2:
            return None, None
        pred = np.empty_like(s)
        for tr_i, te_i in GroupKFold(n_splits=5).split(Xs, s, groups=g):
            sc = StandardScaler().fit(Xs[tr_i])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced")
            clf.fit(sc.transform(Xs[tr_i]), s[tr_i])
            pred[te_i] = clf.predict(sc.transform(Xs[te_i]))
        return balanced_accuracy_score(s, pred), 1.0 / len(set(s))

    X = embed(model)
    bacc, chance = probe(X)

    # CONTROL 1 — hold the DISEASE constant. Prevalence differs by site (stage 4/5 is ROP-VL
    # only; AP-ROP never appears in Shenzhen), so a model that learned nothing but genuine
    # pathology would STILL look site-predictive, purely through that correlation. Restricting
    # to `normal` — the one class present at all three sites — removes that explanation: any
    # decodability left is site appearance, not disease.
    norm_mask = (df.cls6.values == "normal")
    bacc_norm, _ = probe(X, norm_mask)

    # CONTROL 2 — how much site is in the images to begin with? The same probe on the
    # ImageNet-pretrained backbone with NO fine-tuning. Cameras genuinely differ, so some
    # decodability is inherent to the data and not something training introduced. Without
    # this baseline a high number is unreadable: it could mean training LEARNED site, or that
    # training REDUCED site from an even higher starting point.
    base = build_model(arch, int(cfg.data.num_classes), pretrained=True)
    for attr in ("classifier", "fc", "head"):
        if hasattr(base, attr):
            setattr(base, attr, torch.nn.Identity())
            break
    bacc_base, _ = probe(embed(base))
    bacc_base_norm, _ = probe(embed(base), norm_mask)

    print(f"      trained,   all classes   {bacc:.3f}   (chance {chance:.3f})")
    print(f"      trained,   normal only   {bacc_norm:.3f}   <- disease held constant")
    print(f"      untrained, all classes   {bacc_base:.3f}   <- inherent to the images")
    print(f"      untrained, normal only   {bacc_base_norm:.3f}")
    # The gate is on the CONTROLLED number, and only counts training as the culprit if it
    # made things worse than the ImageNet starting point. 0.75 is a judgement call: a
    # representation carrying no site sits near chance, one that is a hospital detector sits
    # near 1.0, so 0.75 is "more than half-way to a hospital detector".
    ok = bacc_norm <= 0.75 or bacc_norm <= bacc_base_norm
    verdict = ("PASS" if ok else "FAIL")
    if bacc_norm > 0.75 and bacc_norm <= bacc_base_norm:
        verdict += " (high, but no higher than the untrained backbone — inherent to the data)"
    report["G5_site_probe"] = {
        "pass": bool(ok), "chance": round(float(chance), 4),
        "trained_all": round(float(bacc), 4), "trained_normal_only": round(float(bacc_norm), 4),
        "untrained_all": round(float(bacc_base), 4),
        "untrained_normal_only": round(float(bacc_base_norm), 4),
        "checkpoint": str(checkpoint), "arch": arch,
        "note": "gate reads trained_normal_only; disease is held constant so the score cannot "
                "be explained by site-varying prevalence",
    }
    print(f"      -> {verdict}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="trained staging checkpoint; enables G5")
    ap.add_argument("--arch", default="convnext_in12k")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()

    report, ok = {}, True
    ok &= gate_patient_leakage(report)
    ok &= gate_duplicate_leakage(report)
    gate_metadata(report)
    if a.checkpoint:
        ok &= gate_site_probe(report, a.checkpoint, a.arch, a.fold)
    else:
        print("\nG5  linear site probe — SKIPPED (no --checkpoint yet)")
        report["G5_site_probe"] = {"pass": None, "reason": "no checkpoint supplied"}

    report["overall_pass"] = bool(ok)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{'ALL RUN GATES PASS' if ok else 'AUDIT FAILED'}  ->  {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
