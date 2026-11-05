"""ROP 6-class STAGING — five-backbone benchmark, patient-grouped 5-fold CV.

Runs ONE backbone across all five folds and writes its pooled out-of-fold predictions.
Run it five times, changing ARCH, to complete the comparison.

WHY ONE BACKBONE PER SESSION: five backbones x five folds is 25 runs at ~30-45 min each,
which is 12-18h — well past Kaggle's ~9h session cap. One backbone x five folds is ~3h and
fits comfortably, with headroom for the 3-seed repeat that makes the ranking meaningful.

    ARCH = "convnext_in12k"   ->  timm:convnext_tiny.in12k_ft_in1k_384   28M   modern CNN
           "caformer"         ->  timm:caformer_s18...                   26M   conv+attn hybrid
           "effnetv2s"        ->  efficientnet_v2_s                      20M   the anchor
           "deit3"            ->  timm:deit3_small_patch16_384...        22M   pure ViT
           "effnetb0"         ->  efficientnet_b0                         4M   capacity floor

NO EARLY STOPPING, BY DESIGN. Each fold's val split carries only 16-19 stage_4_5 images, so
epoch-to-epoch swing on that class exceeds any real signal and "best val epoch" would be noise
selection. Every fold trains for a FIXED epoch budget and the last epoch is kept. Model
selection happens afterwards, on the pooled out-of-fold metric across all five folds.

THE HELD-OUT HOSPITAL IS NOT TOUCHED HERE. Multi-View (test_dev / test_locked) is never loaded
by this kernel. Cross-site behaviour is estimated from fold_site.csv (Shenzhen held out),
which lives inside the training sites. test_locked is opened once, at the very end, by a
separate script — not by a benchmark loop that runs five times.

REQUIRED KAGGLE INPUTS
  1. the code dataset (contains train.py, models/, configs/, results/prepared/)
  2. the ROP image dataset, mirroring data/rop/<source>/... from the repo
  3. (timm arms only) a dataset holding the HF checkpoints, since kernels may have no internet
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
ARCH = "convnext_in12k"      # <- change this per session
#
# 20, not 30. MEASURED on the first real run: EfficientNet-B0 took 212 min for two folds on a
# P100, i.e. ~3.5 min/epoch, so 30 epochs x 6 folds is ~10.5h against Kaggle's ~9h cap — my
# earlier "~3h for five folds" estimate was out by roughly 3x. 20 epochs x 6 folds is ~7h.
# It also costs nothing measurable: on fold 0 the val macro-F1 reached 0.50 by epoch 4 and
# then oscillated between 0.45 and 0.55 for the remaining 26 epochs while train loss fell from
# 0.74 to 0.34 — that stretch is memorisation, not learning. OneCycleLR rescales to whatever
# budget it is given, so this is a complete schedule rather than a truncated one.
EPOCHS = 20
SEED = 42                    # repeat with 1337 and 2024 for the 3-seed ranking

# MEASURED, not guessed. EfficientNet-B0 (4.0M params) and CAFormer-S18 (24.3M) both ran at
# 3.5 min/epoch on the same P100 — a 6x parameter difference with identical wall-clock, which
# means the GPU is not the bottleneck. At ~4,165 images/epoch that is ~20 img/s, far below
# what a P100 does with either model, so the limit is CPU-side preprocessing: CLAHE (a LAB
# round-trip per image), letterbox, and resolution jitter, all at 384px behind 2 workers.
# Kaggle GPU instances give 4 vCPUs. Raising this is the single cheapest speedup available and
# it changes nothing about the experiment — same transforms, same order, same seed.
NUM_WORKERS = 4
FOLDS = [0, 1, 2, 3, 4]
ALSO_SITE_FOLD = True        # additionally run fold_site.csv (Shenzhen held out)

# The config's default is true (all benchmark runs so far trained weighted). The "natural"
# ablation arm flips this to close the flat-weighted vs structured-natural confound from
# BOTH sides — the structured+sampler arm already showed CORN cannot train weighted.
WEIGHTED_SAMPLER = "true"

t0 = time.time()

# GPU COMPATIBILITY. Kaggle hands out either a Tesla T4 (sm_75) or a Tesla P100 (sm_60), and
# the API has no field to request one — enable_gpu is the only knob. The preinstalled torch
# (2.10+cu128) ships kernels for sm_70 and up ONLY, so on a P100 every forward pass dies with
# "no kernel image is available for execution on the device". Not a model problem: nothing at
# all can run. cu121 wheels still carry sm_60, hence the pin below.
#
# The capability is probed with nvidia-smi rather than torch, because reinstalling torch after
# importing it would leave a half-swapped module in memory.
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
    print(f"  {gpu_name} is sm_{int(cap*10)}; the preinstalled torch needs sm_70+. Installing "
          f"a cu121 build, which still carries sm_60 kernels. Takes a few minutes.", flush=True)
    # +cu121 local pins force the pytorch index for torch itself; the extra index covers
    # dependencies it stopped carrying (nvidia-cudnn-cu12==9.1.0.70, gone 2026-08-13).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                    "--extra-index-url", "https://pypi.org/simple"], check=False)

# timm ships with the Kaggle image; this is a convenience for a stale one. check=False so a
# pip failure never aborts the session before a single epoch.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0"], check=False)

import torch  # noqa: E402

print("deps %.0fs | torch %s | cuda %s" % (time.time() - t0, torch.__version__,
                                           torch.cuda.is_available()), flush=True)
# Fail LOUDLY and immediately rather than after five folds of stack traces.
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    supported = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" not in supported:
        raise SystemExit(f"torch {torch.__version__} has no kernels for sm_{major}{minor} "
                         f"(builds: {supported}). Re-run to get a different GPU, or enable "
                         f"internet so the cu121 fallback above can install.")

# OFFLINE WEIGHTS. timm fetches from HuggingFace and torchvision from download.pytorch.org on
# first use; with internet off both raise. praharshnagpure/rop-staging-backbones carries the
# caches for all five backbones, so point the loaders at it when it is attached.
#
# COPY THE CACHE SOMEWHERE WRITABLE FIRST. Pointing HF_HOME straight at /kaggle/input does
# not work: that mount is read-only, and huggingface_hub creates a `.locks/` directory inside
# the cache before it reads anything, so every timm backbone died with
#   OSError: [Errno 30] Read-only file system: '.../hf/hub/.locks'
# torchvision happens to survive a read-only TORCH_HOME because it stats the checkpoint before
# writing, but relying on that difference between two loaders is not worth the ten seconds the
# copy costs — both are staged the same way.
# /tmp, NOT /kaggle/working: everything under /kaggle/working is captured as kernel OUTPUT,
# so staging caches there attaches 436 MB of ImageNet weights to every run's results.
WEIGHTS = "/tmp/_weights"
for cache, env in (("hf", "HF_HOME"), ("torch", "TORCH_HOME")):
    hits = glob.glob(f"/kaggle/input/**/{cache}/hub", recursive=True)
    if not hits:
        print(f"WARNING: no {cache} weight cache found — {env} unset, pretrained weights will "
              f"need internet", flush=True)
        continue
    src, dst = os.path.dirname(hits[0]), os.path.join(WEIGHTS, cache)
    if not os.path.isdir(dst):
        shutil.copytree(src, dst)
    os.environ[env] = dst
    print(f"{env}={dst}  (copied from {src})", flush=True)

# ── locate code + images ─────────────────────────────────────────────────────
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)
print("CODE:", CODE, flush=True)

# The fold manifests were written on a laptop and carry absolute paths like
#   /Users/.../RetinAI/data/rop/<source>/...
# Kaggle mounts the same tree somewhere else, so remap on the "data/rop/" pivot. Anything that
# fails to resolve is a hard error: silently dropping unreadable rows would quietly shrink a
# class and change the experiment.
#
# Anchor on a SOURCE FOLDER, not on the "data/rop" prefix. Kaggle strips the top level of an
# uploaded archive when it extracts, so a tree uploaded as data/rop/<source>/ mounts as
# <dataset>/rop/<source>/ — matching on the prefix found nothing and killed the run before
# epoch 1. A source directory that must exist is a stable anchor under either layout.
SOURCES = ["ostrava", "shenzhen", "rop_vl", "multiview"]
hits = (glob.glob("/kaggle/input/**/rop/ostrava", recursive=True)
        or glob.glob("/kaggle/input/**/data/rop/ostrava", recursive=True))
assert hits, ("ROP image dataset not found under /kaggle/input — is rop-staging-corpus "
              "attached to this kernel?")
ROP_BASE = os.path.dirname(hits[0])
missing = [s for s in SOURCES if not os.path.isdir(os.path.join(ROP_BASE, s))]
assert not missing, f"ROP_BASE={ROP_BASE} is missing source(s) {missing}"
print("ROP_BASE:", ROP_BASE, flush=True)

PREP = "results/prepared/rop_staging"


def remap(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    def fix(p):
        i = p.find("data/rop/")
        return os.path.join(ROP_BASE, p[i + len("data/rop/"):]) if i >= 0 else p
    df["image_path"] = df.image_path.map(fix)
    missing = int((~df.image_path.map(os.path.exists)).sum())
    if missing:
        bad = df[~df.image_path.map(os.path.exists)].image_path.head(3).tolist()
        raise SystemExit(f"{missing} images from {os.path.basename(csv_in)} not found. e.g. {bad}")
    df.to_csv(csv_out, index=False)
    return df


# ── run one fold ─────────────────────────────────────────────────────────────
def run_fold(name, src_csv):
    out_csv = f"{PREP}/_kaggle_{name}.csv"
    df = remap(src_csv, out_csv)
    counts = df[df.split == "train"].cls6.value_counts().to_dict()
    print(f"\n=== {ARCH} | {name} | train {int((df.split=='train').sum())} "
          f"val {int((df.split=='val').sum())} | {counts}", flush=True)
    tag = (f"staging_{ARCH}_{name}_s{SEED}"
           + ("" if WEIGHTED_SAMPLER == "true" else "_natural"))
    cmd = [sys.executable, "train.py", "--config", "configs/rop_staging.yaml", "--tag", tag,
           "--set", f"data.manifest={out_csv}", f"model.arch={ARCH}",
           f"train.epochs={EPOCHS}", f"seed={SEED}", f"train.num_workers={NUM_WORKERS}",
           f"train.weighted_sampler={WEIGHTED_SAMPLER}"]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"!! {name} FAILED (rc={r.returncode})", flush=True)
        return None
    exp = sorted(glob.glob(f"results/experiment_*{tag}*"), key=os.path.getmtime)
    return exp[-1] if exp else None


# WALL-CLOCK GUARD. Kaggle kills the session at ~9h with no warning and no artefacts, so a
# 5-fold run that overruns loses the folds that already finished too. Before starting each
# fold, check whether it can plausibly finish: if the time already spent per completed fold
# projects past the budget, stop cleanly and write out what we have. Partial results from
# three folds are usable; a killed session is not.
BUDGET_S = 8.0 * 3600
results, done = {}, 0


def budget_ok(label):
    if done == 0:
        return True
    per_fold = (time.time() - t0) / done
    if time.time() - t0 + per_fold > BUDGET_S:
        print(f"\n!! stopping before {label}: {per_fold/60:.0f} min/fold would exceed the "
              f"{BUDGET_S/3600:.0f}h budget. {done} fold(s) completed and saved.", flush=True)
        return False
    return True


for f in FOLDS:
    if not budget_ok(f"fold_{f}"):
        break
    results[f"fold_{f}"] = run_fold(f"fold{f}", f"{PREP}/fold_{f}.csv")
    done += 1
if ALSO_SITE_FOLD and budget_ok("fold_site"):
    results["fold_site"] = run_fold("foldsite", f"{PREP}/fold_site.csv")

# ── pool the per-fold metrics ────────────────────────────────────────────────
# folds_requested vs folds_completed: a budget-truncated run must not be mistakable for a
# full one when the JSONs are compared later. Skipped folds are absent from `results`
# entirely, so without this the output would look complete.
summary = {"arch": ARCH, "seed": SEED, "epochs": EPOCHS,
           "folds_requested": len(FOLDS), "folds_completed": done, "folds": {}}
for name, exp in results.items():
    if not exp:
        summary["folds"][name] = {"status": "failed"}
        continue
    mfile = os.path.join(exp, "metrics.json")
    if os.path.exists(mfile):
        m = json.load(open(mfile))
        val = m.get("val", m)
        # Key names must match models/common/metrics.py exactly. They did not: this asked for
        # "balanced_accuracy" and "quadratic_weighted_kappa", which do not exist, so the table
        # printed None for both while the training log showed QWK 0.515 right there. dict.get
        # fails silently, so a wrong key looks like a missing measurement rather than a typo.
        # macro_recall IS balanced accuracy — the mean of per-class recalls, by definition.
        summary["folds"][name] = {
            "macro_f1": val.get("macro_f1"), "accuracy": val.get("accuracy"),
            "balanced_accuracy": val.get("macro_recall"),
            "qwk": val.get("qwk"), "auc": val.get("auc"),
            "per_class": val.get("per_class"), "dir": exp,
        }

cv = [v for k, v in summary["folds"].items()
      if k.startswith("fold_") and k != "fold_site" and v.get("macro_f1") is not None]
if cv:
    import statistics as st
    for metric in ("macro_f1", "balanced_accuracy", "qwk"):
        vals = [v[metric] for v in cv if v.get(metric) is not None]
        if vals:
            summary[f"cv_{metric}_mean"] = round(st.mean(vals), 4)
            summary[f"cv_{metric}_sd"] = round(st.stdev(vals), 4) if len(vals) > 1 else 0.0

out = f"/kaggle/working/staging_{ARCH}_s{SEED}.json"
json.dump(summary, open(out, "w"), indent=2)

print("\n" + "=" * 70)
print(f"{ARCH}  seed {SEED}")
print("=" * 70)
for k, v in summary["folds"].items():
    print(f"  {k:<10} macroF1 {v.get('macro_f1')}  bAcc {v.get('balanced_accuracy')}  "
          f"QWK {v.get('qwk')}")
if cv:
    print(f"\n  CV macro-F1 {summary.get('cv_macro_f1_mean')} "
          f"+- {summary.get('cv_macro_f1_sd')}  (n={len(cv)} folds)")
print(f"\n-> {out}")
print("REMEMBER: rank backbones on CV mean +- SD across 3 seeds. A gap smaller than the SD")
print("is not a result. test_locked stays closed until the winner is chosen.")
print("TOTAL %.0f min" % ((time.time() - t0) / 60), flush=True)
