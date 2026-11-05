"""Structured ROP staging model (ordinal CORN + AP-ROP branch + DANN) on Kaggle.

Runs train_structured.py on the SAME folds as the five-backbone benchmark, so its numbers
drop straight into the comparison table. Change FOLDS/SEED per session exactly like the
benchmark kernel.

The preamble (GPU capability fallback, weight-cache staging, path remap) is duplicated
from rop_staging_benchmark.py on purpose: Kaggle kernels are standalone single files, and
each of these blocks exists because its absence already cost a failed session — P100s
cannot run the preinstalled torch, /kaggle/input is read-only and huggingface_hub writes
locks into HF_HOME, and silently dropped manifest rows would shrink a class.

REQUIRED KAGGLE INPUTS: the code dataset, the ROP image dataset, the weights dataset.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

# ── knobs ────────────────────────────────────────────────────────────────────
ARCH = "effnetv2s"           # cross-site leader in the flat benchmark; keep for like-for-like
EPOCHS = 20
SEED = 42
FOLDS = [0, 1, 2, 3, 4]      # the full CV — fold-0 probe validated the design (v2)
ALSO_SITE_FOLD = True
# Adversary strength, passed through to train.w_site. 0.5 is the main arm; the ablation
# arm runs 2.0 on fold 0 only, to test whether a stronger adversary moves the
# disease-CONTROLLED site probe (at 0.5 it moved only the naive one).
W_SITE = "0.5"
NUM_WORKERS = 4              # measured: these runs are CPU-bound behind 2 workers

# v2: NATURAL sampling. v1 ran the balanced sampler and the CORN head miscalibrated
# catastrophically — stage_4_5 precision 0.043 (~420 false predictions of a 19-support
# class), Normal recall 0.456, QWK 0.121 vs the flat baseline's 0.594. CORN's conditional
# thresholds calibrate to the prevalence they are TRAINED on; the sampler feeds it a world
# where stage_4_5 is ~9x its real share, and on natural-prevalence val data every high-rank
# conditional fires far too often. Flat softmax tolerates the same sampler because argmax
# is rank-free; CORN's cascade is not. The CORN paper itself trains at natural prevalence.
WEIGHTED_SAMPLER = "false"

# Head variant: "full" (the paper's model), "no_ordinal" (softmax stages + AP-ROP branch),
# "no_branch" (CORN over 6 forced ranks, AP-ROP as rank 5). Ablation arms only.
VARIANT = "full"

t0 = time.time()


def _compute_capability():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,name",
                              "--format=csv,noheader"], capture_output=True, text=True)
        cap, name = out.stdout.strip().splitlines()[0].split(",")
        return float(cap), name.strip()
    except Exception:
        return None, "unknown"


cap, gpu_name = _compute_capability()
print(f"GPU: {gpu_name} (compute capability {cap})", flush=True)
if cap is not None and cap < 7.0:
    print("  sm_60 GPU: installing a cu121 torch build (has sm_60 kernels).", flush=True)
    # +cu121 local versions exist ONLY on the pytorch index (so torch cannot silently
    # resolve to a PyPI sm_70+ build), while the extra index lets pip fetch dependencies
    # the pytorch index stopped carrying — nvidia-cudnn-cu12==9.1.0.70 vanished from it
    # on 2026-08-13 and killed a seed run at boot.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                    "--extra-index-url", "https://pypi.org/simple"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0"], check=False)

import torch  # noqa: E402

print("deps %.0fs | torch %s | cuda %s" % (time.time() - t0, torch.__version__,
                                           torch.cuda.is_available()), flush=True)
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
        raise SystemExit(f"torch has no kernels for sm_{major}{minor} — re-run for a "
                         f"different GPU or enable internet for the cu121 fallback.")

# ── weight caches -> writable /tmp (read-only /kaggle/input breaks hf .locks) ─
WEIGHTS = "/tmp/_weights"
for cache, env in (("hf", "HF_HOME"), ("torch", "TORCH_HOME")):
    hits = glob.glob(f"/kaggle/input/**/{cache}/hub", recursive=True)
    if hits:
        src, dst = os.path.dirname(hits[0]), os.path.join(WEIGHTS, cache)
        if not os.path.isdir(dst):
            shutil.copytree(src, dst)
        os.environ[env] = dst
        print(f"{env}={dst}", flush=True)

# ── locate code + images ─────────────────────────────────────────────────────
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train_structured.py", recursive=True)[0])
WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

SOURCES = ["ostrava", "shenzhen", "rop_vl", "multiview"]
hits = (glob.glob("/kaggle/input/**/rop/ostrava", recursive=True)
        or glob.glob("/kaggle/input/**/data/rop/ostrava", recursive=True))
assert hits, "ROP image dataset not found under /kaggle/input"
ROP_BASE = os.path.dirname(hits[0])
missing = [s for s in SOURCES if not os.path.isdir(os.path.join(ROP_BASE, s))]
assert not missing, f"ROP_BASE={ROP_BASE} missing {missing}"
print("ROP_BASE:", ROP_BASE, flush=True)

PREP = "results/prepared/rop_staging"


def remap(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    def fix(p):
        i = p.find("data/rop/")
        return os.path.join(ROP_BASE, p[i + len("data/rop/"):]) if i >= 0 else p
    df["image_path"] = df.image_path.map(fix)
    bad = int((~df.image_path.map(os.path.exists)).sum())
    if bad:
        raise SystemExit(f"{bad} images from {os.path.basename(csv_in)} not found")
    df.to_csv(csv_out, index=False)
    return df


def run_fold(name, src_csv):
    out_csv = f"{PREP}/_kaggle_{name}.csv"
    df = remap(src_csv, out_csv)
    print(f"\n=== structured {ARCH} | {name} | train {int((df.split=='train').sum())} "
          f"val {int((df.split=='val').sum())}", flush=True)
    tag = f"structured_{ARCH}_{name}_s{SEED}" + ("" if VARIANT == "full" else f"_{VARIANT}")
    r = subprocess.run([sys.executable, "train_structured.py",
                        "--config", "configs/rop_staging_structured.yaml", "--tag", tag,
                        "--set", f"data.manifest={out_csv}", f"model.arch={ARCH}",
                        f"model.variant={VARIANT}",
                        f"train.epochs={EPOCHS}", f"seed={SEED}",
                        f"train.num_workers={NUM_WORKERS}",
                        f"train.weighted_sampler={WEIGHTED_SAMPLER}",
                        f"train.w_site={W_SITE}"])
    if r.returncode != 0:
        print(f"!! {name} FAILED (rc={r.returncode})", flush=True)
        return None
    exp = sorted(glob.glob(f"results/experiment_*{tag}*"), key=os.path.getmtime)
    return exp[-1] if exp else None


BUDGET_S = 8.0 * 3600
results, done = {}, 0
for f in FOLDS:
    if done and (time.time() - t0) / done + (time.time() - t0) > BUDGET_S:
        print(f"!! budget stop before fold_{f}", flush=True)
        break
    results[f"fold_{f}"] = run_fold(f"fold{f}", f"{PREP}/fold_{f}.csv")
    done += 1
if ALSO_SITE_FOLD:
    results["fold_site"] = run_fold("foldsite", f"{PREP}/fold_site.csv")

summary = {"model": "structured", "variant": VARIANT, "arch": ARCH, "seed": SEED,
           "epochs": EPOCHS, "w_site": W_SITE, "weighted_sampler": WEIGHTED_SAMPLER,
           "folds_requested": len(FOLDS), "folds_completed": done, "folds": {}}
for name, exp in results.items():
    if not exp:
        summary["folds"][name] = {"status": "failed"}
        continue
    mfile = os.path.join(exp, "metrics.json")
    if os.path.exists(mfile):
        m = json.load(open(mfile))
        val = m.get("val", m)
        summary["folds"][name] = {
            "macro_f1": val.get("macro_f1"), "accuracy": val.get("accuracy"),
            "balanced_accuracy": val.get("macro_recall"), "qwk": val.get("qwk"),
            "auc": val.get("auc"), "per_class": val.get("per_class"), "dir": exp,
        }

out = f"/kaggle/working/structured_{ARCH}_s{SEED}.json"
json.dump(summary, open(out, "w"), indent=2)
print("\n" + "=" * 70)
for k, v in summary["folds"].items():
    pc = (v.get("per_class") or {}).get("AP-ROP", {})
    print(f"  {k:<10} macroF1 {v.get('macro_f1')}  QWK {v.get('qwk')}  "
          f"AP-ROP recall {pc.get('recall')} precision {pc.get('precision')}")
print(f"-> {out}\nTOTAL %.0f min" % ((time.time() - t0) / 60), flush=True)
