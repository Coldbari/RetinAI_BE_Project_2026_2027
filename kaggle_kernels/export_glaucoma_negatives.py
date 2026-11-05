"""Export glaucoma-NEGATIVE images so a glaucoma confusion matrix is possible at all.

Every folder so far certifies a disease is PRESENT, or that DR is absent. Nothing certifies
glaucoma is ABSENT — so glaucoma sensitivity was measurable and specificity was not, and a
confusion matrix with no true-negative row is not a confusion matrix.

SMDG `types == 0` is an explicit non-glaucoma label from a clinically graded collection.
Drawn from the same collections as the positives (ORIGA, REFUGE1, G1020, PAPILA, DRISHTI,
BEH) so the negative set is not trivially separable by source — pulling negatives from a
different collection than the positives would let the model score well by recognising the
dataset, which is the exact confound measured in T11b.
"""
import glob, json, os, shutil
import pandas as pd
from PIL import Image

OUT = "/kaggle/working/test_images_gneg"
os.makedirs(f"{OUT}/glaucoma_negative", exist_ok=True)
N, MAXPX, SEED = 20, 1600, 99

sm_csv = glob.glob("/kaggle/input/**/metadata*.csv", recursive=True)[0]
sm_img = glob.glob("/kaggle/input/**/full-fundus/full-fundus", recursive=True)[0]
sm = pd.read_csv(sm_csv)
sm = sm[sm.types != -1].copy()
sm["src"] = sm["names"].astype(str).str.split("-").str[0]
GOOD = ["ORIGA", "REFUGE1", "G1020", "PAPILA", "DRISHTI", "BEH"]
pool = sm[(sm.types == 0) & (sm.src.isin(GOOD))]
print("glaucoma-negative pool:", len(pool), pool.src.value_counts().to_dict())

man = []
# stratify across collections so the negatives mirror the positives' source mix
per = max(1, N // len(GOOD))
picks = pd.concat([g.sample(min(per, len(g)), random_state=SEED)
                   for _, g in pool.groupby("src")]).head(N)
for _, r in picks.iterrows():
    hits = glob.glob(f"{sm_img}/{r['names']}.*")
    nm = f"glaucomaNeg_{r['src']}_{r['names']}"
    if not hits:
        continue
    im = Image.open(hits[0]).convert("RGB")
    if max(im.size) > MAXPX:
        im.thumbnail((MAXPX, MAXPX))
    im.save(f"{OUT}/glaucoma_negative/{nm}.jpg", quality=94)
    man.append({"folder": "glaucoma_negative", "file": nm + ".jpg",
                "truth": "no glaucoma", "source": f"SMDG / {r['src']}",
                "label_from": "SMDG metadata `types` column (0 = non-glaucoma)"})
json.dump(man, open(f"{OUT}/GROUND_TRUTH_GNEG.json", "w"), indent=2)
shutil.make_archive("/kaggle/working/test_images_gneg", "zip", OUT)
print("exported", len(man), "| by source:",
      pd.Series([m["source"] for m in man]).value_counts().to_dict())
print("DONE", flush=True)
