import os, subprocess, sys, time, shutil, glob, yaml

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
print("torch install %.0fs" % (time.time() - t0), flush=True)

CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
DRC = glob.glob("/kaggle/input/**/diabetic-retinopathy-detection", recursive=True)[0]
aptos_csv = glob.glob("/kaggle/input/**/aptos*/train.csv", recursive=True)[0]
aptos_img = glob.glob("/kaggle/input/**/aptos*/train_images", recursive=True)[0]
print("CODE:", CODE, "| EyePACS:", DRC, "| APTOS:", aptos_img, flush=True)

# extract EyePACS (split 7z archive -> /tmp)
if not os.path.exists("/tmp/train/train"):
    if not (shutil.which("7z") or shutil.which("7za")):
        os.system("apt-get install -y p7zip-full >/dev/null 2>&1")
    sevenz = shutil.which("7z") or shutil.which("7za") or "7z"
    os.makedirs("/tmp/train", exist_ok=True)
    print("extracting EyePACS (~20 min)...", flush=True)
    os.system(f"{sevenz} x {DRC}/train.zip.001 -o/tmp/train/ -y >/dev/null")
if not os.path.exists("/tmp/trainLabels.csv"):
    os.system(f"unzip -o -q {DRC}/trainLabels.csv.zip -d /tmp/")
print("EyePACS images:", len(glob.glob('/tmp/train/train/*')), flush=True)

WORK = "/kaggle/working/code"
shutil.copytree(CODE, WORK, dirs_exist_ok=True)
os.chdir(WORK); sys.path.insert(0, WORK)

cfg = "configs/dr.yaml"
d = yaml.safe_load(open(cfg))
for s in d["data"]["sources"]:
    if s["name"] == "eyepacs":
        s["images_dir"] = "/tmp/train/train"; s["csv"] = "/tmp/trainLabels.csv"
    elif s["name"] == "aptos":
        s["images_dir"] = aptos_img; s["csv"] = aptos_csv
yaml.safe_dump(d, open(cfg, "w"), sort_keys=False)

subprocess.run([sys.executable, "prepare_data.py", "--config", cfg], check=True)
subprocess.run([sys.executable, "train.py", "--config", cfg,
                "--set", "train.epochs=3", "train.num_workers=2"], check=True)
for w in glob.glob("results/experiment_*/weights.pth"):
    print("WEIGHTS:", w, flush=True)
print("DONE", flush=True)
