"""T11b — is the glaucoma 0.973 measuring disease, or measuring which dataset an image is from?

The SMDG source probe (results/glaucoma/smdg_source_probe.json) found that SMDG-19 is a
META-dataset of 15 constituent collections, and that their label composition is extreme:

    EyePACS   3,269 images   3,269 glaucoma      0 non-glaucoma   <- 69% of ALL positives
    OIA       4,442 images     291 glaucoma  4,151 non-glaucoma   <- 6.6% prevalence
    ...

Two collections are 63% of the corpus and sit at opposite ends of prevalence, and the single
largest contributor of positives contains NO negatives at all. That is the ROP device
confound again, one level up: a model can score well by learning "this looks like an EyePACS
capture" rather than "this optic disc is glaucomatous", and a pooled AUC cannot tell the
difference.

This kernel does NOT retrain. It scores the DEPLOYED checkpoint and asks the same question
T8 asked of ROP:

  * per-constituent-source AUC, so discrimination is measured WITHIN a source where the
    source identity carries no label information
  * a source-controlled mean over the collections that carry both classes
  * how much of the pooled figure disappears when source is held constant

If the source-controlled AUC is close to pooled, the 0.973 is real. If it collapses the way
ROP's 0.927 -> 0.684 did, then the headline is largely dataset recognition.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import yaml

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
print("torch %.0fs" % (time.time() - t0), flush=True)

CODE = os.path.dirname(glob.glob("/kaggle/input/**/evaluate.py", recursive=True)[0])
smdg_csv = glob.glob("/kaggle/input/**/metadata*.csv", recursive=True)[0]
smdg_img = glob.glob("/kaggle/input/**/full-fundus/full-fundus", recursive=True)[0]
g1020_csv = glob.glob("/kaggle/input/**/G1020/G1020.csv", recursive=True)[0]
g1020_img = glob.glob("/kaggle/input/**/G1020/Images", recursive=True)[0]
# EXACT name. A previous glaucoma run globbed *.pth and silently loaded an SMDG-only
# checkpoint uploaded before multi-source training finished.
cands = glob.glob("/kaggle/input/**/weights_multisource.pth", recursive=True)
assert cands, "weights_multisource.pth not found — refusing to fall back to another .pth"
WEIGHTS = cands[0]
print("CODE:", CODE, "\nweights:", WEIGHTS, flush=True)

WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

cfg_path = "configs/glaucoma_multisource.yaml"
d = yaml.safe_load(open(cfg_path))
for s in d["data"]["sources"]:
    if s["name"] == "smdg":
        s["images_dir"] = smdg_img; s["csv"] = smdg_csv
    elif s["name"] in ("g1020", "G1020"):
        s["images_dir"] = g1020_img; s["csv"] = g1020_csv
yaml.safe_dump(d, open(cfg_path, "w"), sort_keys=False)
subprocess.run([sys.executable, "prepare_data.py", "--config", cfg_path], check=True)

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import torch                                                  # noqa: E402
from pathlib import Path                                      # noqa: E402
from PIL import Image                                         # noqa: E402
from sklearn.metrics import roc_auc_score                     # noqa: E402
from models.common.config import load_config                  # noqa: E402
from models.common.architectures import build_from_cfg        # noqa: E402
from models.common.preprocessing import build_tta_transforms  # noqa: E402
from models.common.train_utils import predict_with_tta        # noqa: E402

cfg = load_config(cfg_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_from_cfg(cfg, pretrained=False).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()
tta = build_tta_transforms(cfg)

man = pd.read_csv(cfg.data.manifest)
val = man[man.split == "val"].reset_index(drop=True)
print(f"val: {len(val)} images x 5 TTA views", flush=True)


def src_of(p):
    """SMDG names are '<COLLECTION>-<id>'; G1020 arrives from its own source."""
    stem = Path(p).stem
    if "-" in stem:
        head = stem.split("-")[0]
        # OIA-ODIR-... and REFUGE1-... etc: first token is the collection
        return head
    m = re.match(r"^([A-Za-z][A-Za-z0-9]*?)(?=[_\d])", stem)
    return m.group(1) if m else "unknown"


P, Y, S = [], [], []
t = time.time()
for i, r in enumerate(val.itertuples()):
    P.append(predict_with_tta(model, Image.open(r.image_path).convert("RGB"),
                              tta, device, "classification").numpy()[1])
    Y.append(int(r.label))
    S.append(src_of(r.image_path))
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(val)} ({time.time()-t:.0f}s)", flush=True)
df = pd.DataFrame({"y": Y, "p": P, "src": S})

pooled = roc_auc_score(df.y, df.p)
print("\n" + "=" * 78)
print("T11b — GLAUCOMA, AUC WITH SOURCE HELD CONSTANT")
print("=" * 78)
print(f"pooled val AUC {pooled:.4f}   (n={len(df)})\n")
print(f"{'source':<12}{'n':>7}{'glauc':>7}{'prev':>8}{'AUC':>9}")

per_src, measurable = {}, []
for s, g in df.groupby("src"):
    prev = g.y.mean()
    if g.y.nunique() < 2:
        per_src[s] = {"n": int(len(g)), "positives": int(g.y.sum()),
                      "prevalence": float(prev), "auc": None, "note": "SINGLE CLASS"}
        print(f"{s:<12}{len(g):>7}{int(g.y.sum()):>7}{prev:>8.3f}{'n/a':>9}  SINGLE CLASS")
        continue
    a = float(roc_auc_score(g.y, g.p))
    per_src[s] = {"n": int(len(g)), "positives": int(g.y.sum()),
                  "prevalence": float(prev), "auc": a}
    measurable.append(a)
    print(f"{s:<12}{len(g):>7}{int(g.y.sum()):>7}{prev:>8.3f}{a:>9.4f}")

ctrl = float(np.mean(measurable)) if measurable else None
single = [s for s, v in per_src.items() if v["auc"] is None]
pos_in_single = sum(per_src[s]["positives"] for s in single)
print(f"\nmeasurable sources: {len(measurable)}/{len(per_src)}")
if single:
    print(f"single-class sources: {single}")
    print(f"  they contribute {pos_in_single}/{int(df.y.sum())} "
          f"({100*pos_in_single/max(int(df.y.sum()),1):.1f}%) of all positives in val")
if ctrl is not None:
    print(f"\nsource-controlled AUC (mean over sources) {ctrl:.4f}")
    print(f"pooled AUC                                {pooled:.4f}   ({ctrl - pooled:+.4f})")
    print("\nFor scale: the same audit on ROP moved 0.9271 -> 0.6843 once every device")
    print("carried both classes. A comparable collapse here would mean the glaucoma")
    print("headline is largely dataset recognition, not disease discrimination.")

json.dump({"pooled_auc": float(pooled), "n_val": int(len(df)),
           "per_source": per_src, "measurable_sources": len(measurable),
           "source_controlled_auc": ctrl,
           "single_class_sources": single,
           "positives_from_single_class_sources": int(pos_in_single),
           "total_positives": int(df.y.sum())},
          open("/kaggle/working/glaucoma_source_audit.json", "w"), indent=2, default=float)
np.savez("/kaggle/working/glaucoma_val_preds_by_source.npz",
         prob=np.array(P), y_true=np.array(Y), source=np.array(S, dtype=object))
print("\nDONE", flush=True)
