"""Kaggle kernel script — ROP training (validated).

Push from a machine with the Kaggle CLI configured:
    kaggle kernels push -p kaggle_kernels/rop_kernel/      # metadata + this file

Attach as inputs (Kaggle mounts them at /kaggle/input/datasets/<owner>/<slug>):
    - praharshnagpure/retinal-framework-code   (this repo, uploaded as a dataset)
    - praharshnagpure/rop-retinal-dataset      (dataset_split/{train,val,test}/{DG0,DG1})

Notes baked in from debugging:
  * Kaggle's default image ships torch cu128 which DROPS the P100's sm_60 arch, so we
    pin torch 2.5.1+cu121 (works on P100 and T4). Adds ~2.5 min; also deterministic.
  * Datasets mount under /kaggle/input/datasets/<owner>/<slug>/... — discover by glob.
"""
import os, subprocess, sys, time, shutil, glob

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
print("torch install %.0fs" % (time.time() - t0), flush=True)

CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
rop_train = glob.glob("/kaggle/input/**/dataset_split/train", recursive=True)[0]
ROP_BASE = os.path.dirname(os.path.dirname(rop_train))
print("CODE:", CODE, "| ROP_BASE:", ROP_BASE, flush=True)

WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK); sys.path.insert(0, WORK)

cfg = "configs/rop.yaml"
s = open(cfg).read().replace("dataset_split/", ROP_BASE + "/dataset_split/")
open(cfg, "w").write(s)

subprocess.run([sys.executable, "prepare_data.py", "--config", cfg], check=True)
# full run uses config default (30 epochs); use --set train.epochs=3 for a quick check
subprocess.run([sys.executable, "train.py", "--config", cfg,
                "--set", "train.num_workers=2"], check=True)
for w in glob.glob("results/experiment_*/weights.pth"):
    print("WEIGHTS:", w, flush=True)
print("DONE", flush=True)
