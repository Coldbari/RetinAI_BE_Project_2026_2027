"""ROP 6-class staging, v3: ViT-S/14 arms (RETFound-Green, DINOv2) on the preprocessed cache.

One arm x one seed per kernel. Folds run sequentially, each a `train.py` subprocess with a
FIXED epoch budget and no early stopping (the last epoch is kept). Every non-probe fold also
scores the Multi-View dev half from the cache and saves dev_predictions.npz, so no checkpoint
has to be downloaded later. test_locked is never read here.

VARIANT "bag" is arm D: the same kernel on configs/rop_staging_dinov2_bag.yaml, whose
train.bag_loss block pools each multi-frame Ostrava session's frame logits by log-sum-exp
(models/common/bag_loss.py). Evaluation and dev scoring are per image, as for every arm.
VARIANT "r518" is arm G (PROTOCOL_V2 Amendment 4a): B on configs/rop_staging_dinov2_518.yaml,
DINOv2's native 518 px. VARIANT "uda" is arm F (Amendment 4b): B on
configs/rop_staging_dinov2_uda.yaml, whose train.uda block adds self-ensembling on the UNLABELLED
dev photos (models/common/uda.py: train.py reads only their image_path column, never a label);
weights.pth, val_predictions.npz and dev_predictions.npz are then the EMA teacher's. Arms G and F
train the five CV folds only (no site fold), split over two kernels (folds 0,1,2 and 3,4).

PROBE "inner" is the learning-rate probe: fold_0's TRAIN rows only, split by patient 80/20
(seed 0); the 20 % is the val split. Fold_0's own val rows are never used.

POOL (PROTOCOL_V2 Amendment 5a) picks the fold files: "v3" reads results/prepared/rop_staging/,
"v4" reads its v4/ subdirectory (the quality-filtered pool of arms H and I). The dev half is the
same for every pool and is always read from results/prepared/rop_staging/test_dev.csv. LOSO
(Amendment 5a.5) appends the two leave-one-source-out folds after the site fold; with FOLDS = []
and no site fold it is arm B's LOSO-only launch (models/common/v3_launch.py: job_list, fold_file).

ATTACHED DATASETS (and nothing else)
  retinal-framework-code   train.py, models/, configs/, results/prepared/rop_staging/{,v4/}*.csv
  rop-staging-cache768     preprocessed images + CACHE_INFO.json
  rop-v3-weights           retfoundgreen_statedict.pth and hf/hub (DINOv2 timm cache)
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

# ── knobs (rewritten by scripts/push_v3_kernels.py) ──────────────────────────
ARCH = "retfound_green"
SEED = 42
FOLDS = [0, 1, 2, 3, 4]
ALSO_SITE_FOLD = True
LR = "1e-4"
PROBE = ""
VARIANT = ""          # "bag" = arm D (Amendment 3c), "r518" = G, "uda" = F (Amendment 4); CONFIG matches
CONFIG = "configs/rop_staging_rfg.yaml"
EPOCHS = 20
NUM_WORKERS = 4
# fold files: "v3" = results/prepared/rop_staging/, "v4" = its v4/ subdirectory (Amendment 5a)
POOL = "v3"
# True = also the two leave-one-source-out folds, after the site fold (Amendment 5a.5)
LOSO = False

t0 = time.time()


# GPU compatibility: Kaggle hands out a T4 (sm_75) or a P100 (sm_60) at random; the preinstalled
# torch has no sm_60 kernels, so on a P100 install a cu121 build (needs internet).
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
    print(f"  {gpu_name} is sm_{int(cap*10)}; installing a cu121 torch build.", flush=True)
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
        raise SystemExit(f"torch {torch.__version__} has no kernels for sm_{major}{minor} "
                         f"({torch.cuda.get_arch_list()}). Re-run for another GPU.")


def find(rel):
    """Shallow search for `rel` under /kaggle/input (mounts are <ds>/ or datasets/<user>/<ds>/).

    A recursive `**` glob walked every cached image: 5 minutes per call on the first smoke run.
    """
    return sorted(g for depth in range(1, 5)
                  for g in glob.glob("/kaggle/input/" + "*/" * (depth - 1) + rel))


# OFFLINE WEIGHTS. /kaggle/input is read-only and huggingface_hub writes .locks/ into the cache,
# so every hf/hub found is MERGED into a writable copy under /tmp (not /kaggle/working, which
# would be captured as output).
WEIGHTS = "/tmp/_weights"
hub_dst = os.path.join(WEIGHTS, "hf", "hub")
hits = find("hf/hub")
for h in hits:
    shutil.copytree(h, hub_dst, dirs_exist_ok=True)
if hits:
    os.environ["HF_HOME"] = os.path.join(WEIGHTS, "hf")
    print(f"HF_HOME={os.environ['HF_HOME']} (merged {len(hits)} hub dir(s))", flush=True)
else:
    print("WARNING: no hf/hub found under /kaggle/input", flush=True)

rfg = find("retfoundgreen_statedict.pth")
CACHE_INFO = find("CACHE_INFO.json")
assert CACHE_INFO, "rop-staging-cache768 not attached (no CACHE_INFO.json)"
CACHE_ROOT = os.path.dirname(CACHE_INFO[0])
if ARCH == "retfound_green":
    assert rfg, "rop-v3-weights not attached (no retfoundgreen_statedict.pth)"

# ── code ─────────────────────────────────────────────────────────────────────
CODE = os.path.dirname(find("train.py")[0])
WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)
print("CODE:", CODE, "| CACHE_ROOT:", CACHE_ROOT, flush=True)

from models.common.config import load_config  # noqa: E402
from models.common.v3_launch import (FLAGS, check_cache_flags, check_cached_files,  # noqa: E402
                                     check_loso, check_pool, check_site_fold, check_variant,
                                     drop_duplicate_train_rows, fold_file, inner_probe_manifest,
                                     job_list, score_manifest)

PREP = "results/prepared/rop_staging"
cfg0 = load_config(CONFIG)
check_cache_flags(json.load(open(CACHE_INFO[0])), {k: cfg0.preprocess.get(k) for k in FLAGS})
print("launch check: cache flags match", {k: cfg0.preprocess.get(k) for k in FLAGS}, flush=True)
check_variant(VARIANT, cfg0)
check_site_fold(VARIANT, ALSO_SITE_FOLD)
check_loso(VARIANT, LOSO)
check_pool(POOL)
print(f"launch check: variant {VARIANT!r} matches {CONFIG} "
      f"(train.bag_loss={cfg0.train.get('bag_loss') and cfg0.train.bag_loss.to_dict()}, "
      f"train.uda={cfg0.train.get('uda') and cfg0.train.uda.to_dict()}, "
      f"preprocess.image_size={cfg0.preprocess.image_size}, "
      f"train.batch_size={cfg0.train.batch_size}, train.grad_accum={cfg0.train.get('grad_accum', 1)})",
      flush=True)
print(f"launch check: pool {POOL!r}, loso {LOSO}", flush=True)

# Manifests this launch will read, loaded once; every image must have its cached file.
# The dev half is the same for every pool: always PREP/test_dev.csv. Fold files (CV, site and
# LOSO) come through fold_file(): PREP for POOL "v3", PREP/v4 for "v4".
dev_df = pd.read_csv(f"{PREP}/test_dev.csv")
jobs = []                    # (name, manifest DataFrame)
if PROBE == "inner":
    jobs.append(("inner",
                 inner_probe_manifest(pd.read_csv(fold_file("fold0", PREP, POOL)), seed=0)))
else:
    # Protocol v2 "Deduplication": the 14 in-pool exact duplicates never reach training, in every
    # fold (CV, site, LOSO) of either pool. (The lr probes predate this: 12 duplicates sat in
    # fold-0 train, 7 in the inner train and 5 in the inner val; PROTOCOL_V2 Amendment 2.)
    dups = pd.read_csv(f"{PREP}/exclude_duplicates.csv")
    jobs += [(name, drop_duplicate_train_rows(pd.read_csv(fold_file(name, PREP, POOL)), dups))
             for name in job_list(FOLDS, ALSO_SITE_FOLD, LOSO)]
    print("launch check: fold files", {n: fold_file(n, PREP, POOL) for n, _ in jobs}, flush=True)
for name, df in jobs:
    check_cached_files(df, CACHE_ROOT)
check_cached_files(dev_df, CACHE_ROOT)
print(f"launch check: all cached files present for {[n for n, _ in jobs]} + dev", flush=True)

# LOCAL COPY. The first smoke run read the mount lazily: ~2.5 img/s over the first 50 batches.
# Copy just the files this launch uses to local disk with many threads, then read from there.
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from models.common.datasets import cached_path  # noqa: E402

LOCAL = "/tmp/_cache768"
need = sorted({str(cached_path(p, CACHE_ROOT)) for d in [dev_df] + [d for _, d in jobs]
               for p in d.image_path})


def _cp(src):
    dst = os.path.join(LOCAL, os.path.relpath(src, CACHE_ROOT))
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)


tc = time.time()
with ThreadPoolExecutor(32) as ex:
    list(ex.map(_cp, need))
shutil.copyfile(CACHE_INFO[0], os.path.join(LOCAL, "CACHE_INFO.json"))
print(f"cache copy: {len(need)} files -> {LOCAL} in {time.time() - tc:.0f}s", flush=True)
CACHE_ROOT = LOCAL
for name, df in jobs:
    check_cached_files(df, CACHE_ROOT)
check_cached_files(dev_df, CACHE_ROOT)


DEV_ERR = {}      # fold name -> error string, for folds whose dev scoring failed


def run_fold(name, df):
    out_csv = f"{PREP}/_v3_{name}.csv"
    df.to_csv(out_csv, index=False)
    counts = df[df.split == "train"].cls6.value_counts().to_dict()
    arm = f"{ARCH}_{VARIANT}" if VARIANT else ARCH
    print(f"\n=== {arm} | {name} | lr {LR} | train {int((df.split=='train').sum())} "
          f"val {int((df.split=='val').sum())} | {counts}", flush=True)
    tag = f"v3_{arm}_{name}_s{SEED}" + (f"_probe_lr{LR}" if PROBE else "")
    sets = [f"data.manifest={out_csv}", f"train.epochs={EPOCHS}", f"seed={SEED}",
            f"train.lr={LR}", f"train.num_workers={NUM_WORKERS}",
            "preprocess.cached=true", f"preprocess.cache_root={CACHE_ROOT}"]
    if ARCH == "retfound_green":
        sets.append(f"model.weights_path={rfg[0]}")
    r = subprocess.run([sys.executable, "train.py", "--config", CONFIG, "--tag", tag,
                        "--set", *sets])
    if r.returncode != 0:
        print(f"!! {name} FAILED (rc={r.returncode})", flush=True)
        return None
    exp = sorted(glob.glob(f"results/experiment_*{tag}*"), key=os.path.getmtime)
    if not exp:
        return None
    try:
        p = score_manifest(exp[-1], dev_df, CACHE_ROOT, "dev_predictions.npz",
                           num_workers=NUM_WORKERS)
        print(f"dev scored -> {p}", flush=True)
    except Exception as e:      # a dev-scoring failure must not discard a trained fold
        DEV_ERR[name] = repr(e)
        print(f"!! dev scoring failed for {name}: {e!r}", flush=True)
    return exp[-1]


# WALL-CLOCK GUARD: Kaggle kills the session at ~9h with no artefacts.
BUDGET_S = 8.0 * 3600
results, done = {}, 0
for name, df in jobs:
    if done and time.time() - t0 + (time.time() - t0) / done > BUDGET_S:
        print(f"!! stopping before {name}: budget would be exceeded; {done} done", flush=True)
        break
    results[name] = run_fold(name, df)
    done += 1

summary = {"arch": ARCH, "variant": VARIANT, "seed": SEED, "epochs": EPOCHS, "lr": LR,
           "probe": PROBE, "pool": POOL, "loso": LOSO, "gpu": gpu_name,
           "jobs_requested": len(jobs), "jobs_completed": done, "folds": {}}
for name, exp in results.items():
    mfile = os.path.join(exp, "metrics.json") if exp else None
    if not mfile or not os.path.exists(mfile):
        summary["folds"][name] = {"status": "failed"}
        continue
    m = json.load(open(mfile))
    val = m.get("val", m)
    summary["folds"][name] = {
        "macro_f1": val.get("macro_f1"), "accuracy": val.get("accuracy"),
        "balanced_accuracy": val.get("macro_recall"), "qwk": val.get("qwk"),
        "auc": val.get("auc"), "per_class": val.get("per_class"), "dir": exp}
    dev_ok = os.path.exists(os.path.join(exp, "dev_predictions.npz"))
    summary["folds"][name]["dev_scored"] = dev_ok
    if not dev_ok:
        summary["folds"][name]["dev_error"] = DEV_ERR.get(name, "dev_predictions.npz missing")
# the CV folds only (fold0..fold4): never the site or LOSO folds
cv = [v["macro_f1"] for k, v in summary["folds"].items()
      if k.removeprefix("fold").isdigit() and v.get("macro_f1") is not None]
if cv:
    import statistics as st
    summary["cv_macro_f1_mean"] = round(st.mean(cv), 4)
    summary["cv_macro_f1_sd"] = round(st.stdev(cv), 4) if len(cv) > 1 else 0.0
json.dump(summary, open("/kaggle/working/summary.json", "w"), indent=2)
print("\n" + json.dumps({k: v for k, v in summary.items() if k != "folds"}), flush=True)
for k, v in summary["folds"].items():
    print(f"  {k:<10} macroF1 {v.get('macro_f1')}  QWK {v.get('qwk')}")
print("TOTAL %.0f min" % ((time.time() - t0) / 60), flush=True)

# Fail the kernel (after summary.json is safely written, training outputs kept) when a run
# that should have produced dev predictions did not, or when nothing trained at all.
failed_dev = [] if PROBE else [k for k, v in summary["folds"].items()
                               if v.get("dev_scored") is False]
trained = [k for k, v in summary["folds"].items() if v.get("macro_f1") is not None]
if failed_dev:
    print(f"!! DEV SCORING MISSING for: {failed_dev}", flush=True)
if not trained:
    print("!! NO JOB COMPLETED TRAINING", flush=True)
if failed_dev or not trained:
    sys.exit(1)
