import os, subprocess, sys, time, shutil, glob, yaml
import importlib

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
print("torch install %.0fs" % (time.time() - t0), flush=True)

import pandas as pd
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
csv = glob.glob("/kaggle/input/**/metadata - standardized.csv", recursive=True)[0]
root = os.path.dirname(csv)
# discover the real fundus image dir by locating the first labelled image
n0 = str(pd.read_csv(csv)["names"].dropna().iloc[0])
hit = glob.glob(root + f"/**/{n0}.png", recursive=True)
IMG = os.path.dirname(hit[0])
print("CODE:", CODE, "| CSV:", csv, "| IMG:", IMG, flush=True)

WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK); sys.path.insert(0, WORK)

cfg = "configs/glaucoma.yaml"
d = yaml.safe_load(open(cfg))
src = d["data"]["sources"][0]
src["csv"] = csv; src["images_dir"] = IMG; src["ext"] = [".png"]
src.pop("disc_mask_dir", None); src.pop("kagglehub", None)
d["preprocess"]["optic_disc_roi"] = False
yaml.safe_dump(d, open(cfg, "w"), sort_keys=False)

subprocess.run([sys.executable, "prepare_data.py", "--config", cfg], check=True)
subprocess.run([sys.executable, "train.py", "--config", cfg,
                "--set", "train.epochs=3", "train.num_workers=2"], check=True)
for w in glob.glob("results/experiment_*/weights.pth"):
    print("WEIGHTS:", w, flush=True)
print("DONE", flush=True)
