"""Collect a teammate's three independent ROP packages (rop_ml, rop_cnn, rop_foundation; Nov 2026) into
ONE committed JSON, results/comparison/teammate_packages.json, so paper 3 (the results compendium) can
tabulate them from a file like every other generation.

Read only: their result CSVs, per-run meta.json / report.txt / history.csv / logs, the prepared-data
.npz (its `paths`, `y`, `split` arrays; the pixel arrays are never touched) and the training scripts'
argparse defaults. Nothing from the packages is executed; no checkpoint is opened here (the webapp's
deploy script does that with weights_only=True). Our own split files are joined by photo file name to
audit overlap and infant leakage.

    .venv/bin/python scripts/collect_teammate_packages.py \
        --ml "~/Downloads/rop_ml - Copy" --cnn "~/Downloads/rop_cnn - Copy (2)" \
        --fm "~/Downloads/rop_foundation - Copy (2)"
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/comparison/teammate_packages.json"
OUR_SPLITS = ROOT / "results/prepared/rop_staging/splits.csv"
CLASSES = ["NORMAL", "STAGE1", "STAGE2", "STAGE3"]
FM_RUNS = ["p_mlp", "p_mlp_r32", "p_mlp_dp1", "p_mlp_lr6e4", "p_mlp_img280", "p_mlp_ord", "p_mlp_combo",
           "final_base_s42", "final_mlp_s42", "final_mlp_img280_s42", "final_mlp_img280_s1", "final_mlp_img280_s2"]
FM_CHANGE = {"p_mlp": "LoRA on attention + MLP (probe base)", "p_mlp_r32": "rank 32, alpha 64",
             "p_mlp_dp1": "drop-path 0.1", "p_mlp_lr6e4": "lr 6e-4", "p_mlp_img280": "280 px input",
             "p_mlp_ord": "ordinal auxiliary loss 0.3", "p_mlp_combo": "r32 + dp 0.1 + lr 6e-4 + 280 px",
             "final_base_s42": "attention-only LoRA, 30 ep", "final_mlp_s42": "attention + MLP, 224 px, 30 ep",
             "final_mlp_img280_s42": "chosen config, seed 42", "final_mlp_img280_s1": "chosen config, seed 1",
             "final_mlp_img280_s2": "chosen config, seed 2"}


def rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _num(v):
    try:
        return int(v) if re.fullmatch(r"-?\d+", v) else float(v)
    except (TypeError, ValueError):
        return v


def report(path: Path) -> dict:
    """sklearn classification_report text -> {class: {precision, recall, f1, support}, accuracy, n}."""
    out = {}
    for ln in path.read_text().splitlines():
        m = re.match(r"\s*(NORMAL|STAGE1|STAGE2|STAGE3|macro avg)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", ln)
        if m:
            out[m.group(1)] = {"precision": float(m.group(2)), "recall": float(m.group(3)),
                               "f1": float(m.group(4)), "support": int(m.group(5))}
        m = re.match(r"\s*accuracy\s+([\d.]+)\s+(\d+)", ln)
        if m:
            out["accuracy"], out["n"] = float(m.group(1)), int(m.group(2))
    return out


def epochs_run(path: Path) -> int:
    return max(0, sum(1 for _ in open(path)) - 1)


def argparse_defaults(script: Path, names) -> dict:
    t = script.read_text()
    out = {}
    for n in names:
        m = re.search(rf'add_argument\("--{n}"[^)]*?default=([^,)]+)', t)
        if m:
            out[n] = _num(m.group(1).strip().strip('"'))
    return out


def data_audit(npz: Path, ml_manifest: Path | None) -> dict:
    z = np.load(npz, allow_pickle=False)
    theirs = pd.DataFrame({"name": [p.replace("\\", "/").rsplit("/", 1)[-1] for p in z["paths"]],
                           "y": z["y"], "split": z["split"]})
    per_class = {s: {CLASSES[c]: int(((theirs.split == s) & (theirs.y == c)).sum()) for c in range(4)}
                 for s in ("train", "val", "test")}
    ours = pd.read_csv(OUR_SPLITS)
    ours["name"] = ours.image_path.str.rsplit("/", n=1).str[1]
    j = theirs.merge(ours[["name", "split", "source", "patient_id"]].rename(columns={"split": "our_split"}),
                     on="name", how="left")
    tab = pd.crosstab(j.split, j.our_split.fillna("not in our splits"))
    train_inf = set(j[(j.split == "train") & j.patient_id.notna()].patient_id)
    leak = {}
    for s in ("val", "test"):
        t = j[(j.split == s) & j.patient_id.notna()]
        leak[s] = {"matched": int(len(t)), "same_infant_as_train": int(t.patient_id.isin(train_inf).sum()),
                   "by_source": {src: {"n": int((t.source == src).sum()),
                                       "same_infant_as_train": int(t[t.source == src].patient_id.isin(train_inf).sum())}
                                 for src in sorted(t.source.dropna().unique())}}
    out = {"n_photos": int(len(theirs)), "classes": CLASSES, "per_class_by_split": per_class,
           "matched_to_our_data_by_name": int(j.our_split.notna().sum()),
           "sources_by_name": {k: int(v) for k, v in j.source.value_counts().items()},
           "their_split_vs_our_split": {r: {c: int(v) for c, v in row.items()} for r, row in tab.to_dict("index").items()},
           "infant_leakage": leak}
    if ml_manifest and ml_manifest.exists():
        m = pd.read_csv(ml_manifest)
        out["ml_split"] = {s: {c: int(((m.split == s) & (m.label == c)).sum()) for c in CLASSES} for s in ("train", "test")}
    return out


def test_clusters(npz: Path) -> tuple[np.ndarray, np.ndarray]:
    """(y_test, cluster id per test row): our infant id by file name; an unmatched photo is its own cluster."""
    z = np.load(npz, allow_pickle=False)
    names = np.array([p.replace("\\", "/").rsplit("/", 1)[-1] for p in z["paths"]])
    ours = pd.read_csv(OUR_SPLITS)
    pid = dict(zip(ours.image_path.str.rsplit("/", n=1).str[1], ours.patient_id))
    m = z["split"] == "test"
    clusters = np.array([pid.get(n, f"unmatched:{n}") for n in names[m]])
    return z["y"][m], clusters


def clustered_acc(probs: np.ndarray, y: np.ndarray, clusters: np.ndarray, reps: int = 2000, seed: int = 42) -> dict:
    """Test accuracy with a 95% percentile bootstrap over infants (clusters), 2,000 replicates."""
    correct = (probs.argmax(1) == y).astype(float)
    ids = np.unique(clusters)
    groups = [np.where(clusters == c)[0] for c in ids]
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        accs.append(correct[idx].mean())
    return {"accuracy": float(correct.mean()), "ci95_infant": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "n_clusters": int(len(ids)), "replicates": reps, "seed": seed}


def collect_ml(d: Path) -> dict:
    res = rows(d / "outputs/results.csv")
    best = res[0]["Model"]
    z = np.load(d / "outputs/features.npz", allow_pickle=False)
    return {"package": "rop_ml", "n_features": int(z["Xtr"].shape[1]), "train_rows_after_augmentation": int(z["Xtr"].shape[0]),
            "test_rows": int(z["Xte"].shape[0]), "cv_folds": 5, "selection": "5-fold CV accuracy on the training rows; "
            "ensembles of the top 3; table sorted by test accuracy", "models": res, "best": best,
            "best_report": report(d / "outputs/best_model_report.txt")}


def collect_cnn(d: Path) -> dict:
    import timm
    recipe = argparse_defaults(d / "02_train.py", ["epochs", "warmup", "bs", "img", "lr", "wd", "patience"])
    y, cl = test_clusters(d / "outputs/data_clahe.npz")
    models = []
    for meta in sorted((d / "outputs").glob("*/meta.json")):
        m = json.loads(meta.read_text())
        net = timm.create_model(m["model"], pretrained=False, num_classes=4)
        models.append({"name": m["model"], "params": int(sum(p.numel() for p in net.parameters())),
                       "epochs_run": epochs_run(meta.parent / "history.csv"), "val": m["val"], "test": m["test"],
                       "report": report(meta.parent / "report.txt"),
                       "test_acc_clustered": clustered_acc(np.load(meta.parent / "test_probs.npy"), y, cl),
                       "checkpoint_bytes": (meta.parent / "best.pt").stat().st_size})
    res = rows(d / "outputs/cnn_results.csv")
    ens = next(r for r in res if r["Model"].startswith("Ensemble"))
    members = re.search(r"Ensemble of \[(.*?)\]", (d / "outputs/ensemble_report.txt").read_text()).group(1)
    ens_probs = np.load(d / "outputs/ensemble_test_probs.npy")
    return {"package": "rop_cnn", "recipe": {**recipe, "pretrained": "ImageNet (timm)", "dropout": 0.3,
                                             "loss": "class-weighted CE, label smoothing 0.1", "tta": "3-view flip"},
            "models": models, "results_table": res,
            "ensemble": {"members": [s.strip().strip("'") for s in members.split(",")], "val_f1": ens["Val_F1"],
                         "test": {k: ens[k] for k in ("Accuracy", "BalancedAcc", "MacroF1", "QWK", "AUC_OvR")},
                         "test_acc_clustered": clustered_acc(ens_probs, y, cl),
                         "report": report(d / "outputs/ensemble_report.txt")}}


def collect_fm(d: Path, cnn_dir: Path) -> dict:
    y, cl = test_clusters(d / "outputs/data_clahe_320.npz")
    probs = {}
    runs = []
    for tag in FM_RUNS:
        rd = d / "outputs" / f"retfound_dinov2_lora_{tag}"
        m = json.loads((rd / "meta.json").read_text())
        probs[f"retfound_dinov2_lora_{tag}"] = np.load(rd / "test_probs.npy")
        log = d / "outputs/logs" / f"retfound_dinov2_lora_{tag}.log"
        tp = re.search(r"trainable params ([\d.]+)M / total ([\d.]+)M", log.read_text()) if log.exists() else None
        a = m["args"]
        runs.append({"tag": tag, "change": FM_CHANGE[tag], "seed": a["seed"], "img": a["img"], "epochs_max": a["epochs"],
                     "epochs_run": epochs_run(rd / "history.csv"), "lora_r": a["lora_r"], "lora_alpha": a["lora_alpha"],
                     "lora_mlp": a["lora_mlp"], "lr": a["lr"], "drop_path": a["drop_path"], "ordinal": a["ordinal"],
                     "trainable_M": float(tp.group(1)) if tp else None, "total_M": float(tp.group(2)) if tp else None,
                     "val": m["val"], "test": m["test"], "report": report(rd / "report.txt"),
                     "test_acc_clustered": clustered_acc(probs[f"retfound_dinov2_lora_{tag}"], y, cl)})
    seeds = rows(d / "outputs/fm_results_seeds.csv")
    with_cnn = rows(d / "outputs/fm_results_with_cnn.csv")
    ens = {}
    for key, tab, rep in (("seeds", seeds, "ensemble_report_seeds.txt"), ("with_cnn", with_cnn, "ensemble_report_with_cnn.txt")):
        e = next(r for r in tab if r["Model"].startswith("ENSEMBLE"))
        members = re.search(r"Ensemble of \[(.*?)\]", (d / "outputs" / rep).read_text()).group(1)
        mem = [s.strip().strip("'") for s in members.split(",")]
        mp = []
        for name in mem:
            if name.startswith("cnn:"):
                mp.append(np.load(cnn_dir / "outputs" / name[4:] / "test_probs.npy"))
            else:
                mp.append(probs[name])
        ens[key] = {"members": mem, "val_f1": e["Val_F1"],
                    "test": {k: e[k] for k in ("Accuracy", "BalancedAcc", "MacroF1", "QWK", "AUC_OvR")},
                    "test_acc_clustered": clustered_acc(np.mean(mp, 0), y, cl),
                    "report": report(d / "outputs" / rep)}
    recipe = {"backbone": "RETFound-DINOv2 ViT-L/14 (frozen)", "weights_file": "RETFound_dinov2_meh.pth",
              "adapters": "LoRA on attn.qkv, attn.proj, mlp.fc1, mlp.fc2 of all 24 blocks (96 layers); rank 16, alpha 32",
              "head": "linear, 4 classes, on the class token", "optimizer": "AdamW, lr 4e-4 (head x3), wd 0.05, warm-up 3 + cosine",
              "batch": "4 x 4 accumulation", "loss": "class-weighted CE, label smoothing 0.1", "precision": "bf16",
              "selection": "best validation macro-F1, patience 10; stage-2 probe chosen on validation macro-F1",
              "tta": "4-view flip"}
    return {"package": "rop_foundation", "recipe": recipe, "runs": runs, "ensembles": ens}


def main():
    ap = argparse.ArgumentParser()
    dl = Path.home() / "Downloads"
    ap.add_argument("--ml", default=str(dl / "rop_ml - Copy"))
    ap.add_argument("--cnn", default=str(dl / "rop_cnn - Copy (2)"))
    ap.add_argument("--fm", default=str(dl / "rop_foundation - Copy (2)"))
    a = ap.parse_args()
    ml, cnn, fm = (Path(p).expanduser() for p in (a.ml, a.cnn, a.fm))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"generated_on": date.today().isoformat(), "generated_by": "scripts/collect_teammate_packages.py", "commit": commit,
           "author": "a RetinAI team member (independent packages, 3-5 Nov 2026, one Windows GPU)",
           "note": "Photo-level 80/10/10 split (rop_ml: 80/20) of 3,141 project photographs, four classes; outside PROTOCOL_V2; "
                   "not comparable with arms A-I; trained on development-half and locked-half photographs.",
           "sources": {"ml": str(ml), "cnn": str(cnn), "fm": str(fm)},
           "data": data_audit(fm / "outputs/data_clahe.npz", ml / "outputs/manifest.csv"),
           "ml": collect_ml(ml), "cnn": collect_cnn(cnn), "foundation": collect_fm(fm, cnn)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(OUT.relative_to(ROOT), f"{OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
