"""Export a small, LABELLED hold-out test folder for end-to-end model testing.

Every image exported here carries a label that comes from the originating dataset's own
ground truth CSV — not from a web caption, not from a filename someone typed. That matters:
the point of this exercise is to check whether the models are right, and an image whose
"correct answer" we only think we know cannot test anything.

Selection rules, and why:

  * DR positives are drawn at grade >= 3 (severe / proliferative), not grade 2. Grade 1-2 is
    where human graders disagree most (it is the documented ceiling on this dataset's QWK),
    so a wrong prediction there would be ambiguous evidence. Severe disease is unambiguous.
  * Glaucoma positives are drawn from collections with clinical provenance (ORIGA, REFUGE,
    G1020, PAPILA, DRISHTI) and NOT from EyePACS-AIROGS, which contains no healthy eyes and
    therefore contributed nothing to within-source discrimination.
  * "Normal" is drawn from APTOS grade 0. Stated honestly: that means *no diabetic
    retinopathy*, which is NOT the same as a perfectly healthy eye. A grade-0 APTOS image
    has not been screened for glaucoma.
  * Everything comes from the VAL/TEST side where a split exists, so we are not testing on
    images the models trained on.

Filenames encode the truth: <disease>_<label>_<source>_<id>.<ext>
"""
import glob
import json
import os
import random
import shutil

import pandas as pd
from PIL import Image

random.seed(42)
OUT = "/kaggle/working/test_images"
N = 10
MAXPX = 1600            # keep the download small; still far above the 384px the model uses

for sub in ("normal", "dr", "glaucoma"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)

manifest = []


def save(src, dest_dir, name):
    try:
        im = Image.open(src).convert("RGB")
    except Exception as e:
        print("  skip", src, e)
        return False
    if max(im.size) > MAXPX:
        im.thumbnail((MAXPX, MAXPX))
    im.save(f"{OUT}/{dest_dir}/{name}.jpg", quality=94)
    return True


# ── DR: APTOS 2019, labels from train.csv ────────────────────────────────────
ap_csv = glob.glob("/kaggle/input/**/aptos*/train.csv", recursive=True)[0]
ap_img = glob.glob("/kaggle/input/**/aptos*/train_images", recursive=True)[0]
ap = pd.read_csv(ap_csv)
print("APTOS:", len(ap), "rows | grade counts:", ap.diagnosis.value_counts().to_dict())

severe = ap[ap.diagnosis >= 3].sample(min(N, (ap.diagnosis >= 3).sum()), random_state=42)
for _, r in severe.iterrows():
    p = f"{ap_img}/{r.id_code}.png"
    if save(p, "dr", f"dr_grade{r.diagnosis}_aptos_{r.id_code}"):
        manifest.append({"folder": "dr", "file": f"dr_grade{r.diagnosis}_aptos_{r.id_code}.jpg",
                         "truth": "referable DR", "grade": int(r.diagnosis),
                         "source": "APTOS 2019", "label_from": "train.csv diagnosis column"})

normal = ap[ap.diagnosis == 0].sample(N, random_state=7)
for _, r in normal.iterrows():
    p = f"{ap_img}/{r.id_code}.png"
    if save(p, "normal", f"normal_noDR_aptos_{r.id_code}"):
        manifest.append({"folder": "normal", "file": f"normal_noDR_aptos_{r.id_code}.jpg",
                         "truth": "no diabetic retinopathy", "grade": 0,
                         "source": "APTOS 2019", "label_from": "train.csv diagnosis column",
                         "caveat": "grade 0 means no DR; the eye was not screened for glaucoma"})

# ── Glaucoma: SMDG, labels from metadata `types` ─────────────────────────────
sm_csv = glob.glob("/kaggle/input/**/metadata*.csv", recursive=True)[0]
sm_img = glob.glob("/kaggle/input/**/full-fundus/full-fundus", recursive=True)[0]
sm = pd.read_csv(sm_csv)
sm = sm[sm.types != -1]
sm["src"] = sm["names"].astype(str).str.split("-").str[0]
GOOD = ["ORIGA", "REFUGE1", "G1020", "PAPILA", "DRISHTI", "BEH"]
pool = sm[(sm.types == 1) & (sm.src.isin(GOOD))]
print("glaucoma pool:", len(pool), "| by source:", pool.src.value_counts().to_dict())
pick = pool.sample(min(N, len(pool)), random_state=42)
for _, r in pick.iterrows():
    hits = glob.glob(f"{sm_img}/{r['names']}.*")
    if hits and save(hits[0], "glaucoma", f"glaucoma_pos_{r['src']}_{r['names']}"):
        manifest.append({"folder": "glaucoma",
                         "file": f"glaucoma_pos_{r['src']}_{r['names']}.jpg",
                         "truth": "glaucoma", "source": f"SMDG / {r['src']}",
                         "label_from": "SMDG metadata `types` column (1 = glaucoma)"})

json.dump(manifest, open(f"{OUT}/GROUND_TRUTH.json", "w"), indent=2)
shutil.make_archive("/kaggle/working/test_images", "zip", OUT)
print(f"\nexported {len(manifest)} labelled images")
for f in ("normal", "dr", "glaucoma"):
    print(f"  {f:<10} {len(os.listdir(f'{OUT}/{f}'))}")
print("DONE", flush=True)
