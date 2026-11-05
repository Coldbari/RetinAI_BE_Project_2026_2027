"""Export a SECOND batch of labelled test images — 10 more per folder, no overlap.

Same rule as batch 1: the label comes from the originating dataset's ground-truth CSV, never
from a filename or a caption. An image whose correct answer we only think we know cannot
test anything.

Stricter than batch 1 on certainty, because these are being used to score the models:

  * DR positives are grade 4 (proliferative) first, then grade 3 (severe). Grades 1-2 are
    where human graders disagree most — that disagreement is the documented ceiling on this
    dataset's QWK — so a wrong prediction there would be ambiguous evidence, not a failure.
  * Glaucoma positives come only from clinically graded collections (ORIGA, REFUGE1, G1020,
    PAPILA, DRISHTI, BEH). EyePACS-AIROGS is excluded: it contains no healthy eyes at all,
    so its label carries no discriminative information.
  * "Normal" is APTOS grade 0. Stated plainly in the manifest: that certifies NO DIABETIC
    RETINOPATHY. It does not certify a healthy eye — nobody examined those images for
    glaucoma, and claiming otherwise would put an unverifiable label in a scoring folder.

Excludes every id already exported in batch 1, so the combined set has no duplicates.
"""
import glob
import json
import os
import shutil

import pandas as pd
from PIL import Image

OUT = "/kaggle/working/test_images_v2"
N = 10
MAXPX = 1600
SEED = 1234                      # different from batch 1's 42/7

ALREADY = set()
for p in glob.glob("/kaggle/input/**/GROUND_TRUTH.json", recursive=True):
    try:
        for r in json.load(open(p)):
            ALREADY.add(r["file"].rsplit("_", 1)[-1].replace(".jpg", ""))
    except Exception:
        pass
print(f"excluding {len(ALREADY)} ids already exported in batch 1")

for sub in ("normal", "dr", "glaucoma"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
manifest = []


def save(src, folder, name):
    try:
        im = Image.open(src).convert("RGB")
    except Exception as e:
        print("  skip", src, e)
        return False
    if max(im.size) > MAXPX:
        im.thumbnail((MAXPX, MAXPX))
    im.save(f"{OUT}/{folder}/{name}.jpg", quality=94)
    return True


# ── DR ───────────────────────────────────────────────────────────────────────
ap_csv = glob.glob("/kaggle/input/**/aptos*/train.csv", recursive=True)[0]
ap_img = glob.glob("/kaggle/input/**/aptos*/train_images", recursive=True)[0]
ap = pd.read_csv(ap_csv)
ap = ap[~ap.id_code.isin(ALREADY)]
print("APTOS available after exclusions:", len(ap),
      ap.diagnosis.value_counts().sort_index().to_dict())

# grade 4 first (least ambiguous), top up with grade 3
pool = pd.concat([ap[ap.diagnosis == 4].sample(frac=1, random_state=SEED),
                  ap[ap.diagnosis == 3].sample(frac=1, random_state=SEED)]).head(N)
for _, r in pool.iterrows():
    nm = f"dr_grade{r.diagnosis}_aptos_{r.id_code}"
    if save(f"{ap_img}/{r.id_code}.png", "dr", nm):
        manifest.append({"folder": "dr", "file": nm + ".jpg", "truth": "referable DR",
                         "grade": int(r.diagnosis), "source": "APTOS 2019",
                         "label_from": "train.csv diagnosis column",
                         "certainty": "grade 4 = proliferative, grade 3 = severe; "
                                      "unambiguous referable disease"})

norm = ap[ap.diagnosis == 0].sample(N, random_state=SEED)
for _, r in norm.iterrows():
    nm = f"normal_noDR_aptos_{r.id_code}"
    if save(f"{ap_img}/{r.id_code}.png", "normal", nm):
        manifest.append({"folder": "normal", "file": nm + ".jpg",
                         "truth": "no diabetic retinopathy", "grade": 0,
                         "source": "APTOS 2019", "label_from": "train.csv diagnosis column",
                         "certainty": "certifies NO DR only — these eyes were never "
                                      "examined for glaucoma, so a glaucoma finding on them "
                                      "is unverifiable, not wrong"})

# ── Glaucoma ─────────────────────────────────────────────────────────────────
sm_csv = glob.glob("/kaggle/input/**/metadata*.csv", recursive=True)[0]
sm_img = glob.glob("/kaggle/input/**/full-fundus/full-fundus", recursive=True)[0]
sm = pd.read_csv(sm_csv)
sm = sm[sm.types != -1].copy()
sm["src"] = sm["names"].astype(str).str.split("-").str[0]
GOOD = ["ORIGA", "REFUGE1", "G1020", "PAPILA", "DRISHTI", "BEH"]
pool = sm[(sm.types == 1) & (sm.src.isin(GOOD)) & (~sm["names"].isin(ALREADY))]
print("glaucoma pool after exclusions:", len(pool), pool.src.value_counts().to_dict())
for _, r in pool.sample(min(N, len(pool)), random_state=SEED).iterrows():
    hits = glob.glob(f"{sm_img}/{r['names']}.*")
    nm = f"glaucoma_pos_{r['src']}_{r['names']}"
    if hits and save(hits[0], "glaucoma", nm):
        manifest.append({"folder": "glaucoma", "file": nm + ".jpg", "truth": "glaucoma",
                         "source": f"SMDG / {r['src']}",
                         "label_from": "SMDG metadata `types` column (1 = glaucoma)",
                         "certainty": "clinically graded collection; EyePACS-AIROGS "
                                      "deliberately excluded (contains no healthy eyes)"})

json.dump(manifest, open(f"{OUT}/GROUND_TRUTH_V2.json", "w"), indent=2)
shutil.make_archive("/kaggle/working/test_images_v2", "zip", OUT)
print(f"\nexported {len(manifest)}")
for f in ("normal", "dr", "glaucoma"):
    print(f"  {f:<10} {len(os.listdir(f'{OUT}/{f}'))}")
print("DONE", flush=True)
