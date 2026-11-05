"""02_DATASET + 03_DATA_LEAKAGE. Recomputed from the committed split CSVs."""
import sys, itertools, hashlib
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
sys.path.insert(0, str(ROOT))

folds = {f"fold_{i}": pd.read_csv(PREP / f"fold_{i}.csv") for i in range(5)}
folds["fold_site"] = pd.read_csv(PREP / "fold_site.csv")
dev = pd.read_csv(PREP / "test_dev.csv"); lock = pd.read_csv(PREP / "test_locked.csv")

# ── counts ───────────────────────────────────────────────────────────────────
rows = []
for k, d in folds.items():
    for sp in ("train", "val"):
        s = d[d.split == sp]
        if len(s) == 0: continue
        rows.append(dict(split_file=k, role=sp, images=len(s), patients=s.patient_id.nunique(),
                         **{c: int((s.cls6 == c).sum()) for c in CLASSES}))
for nm, d in (("test_dev", dev), ("test_locked", lock)):
    rows.append(dict(split_file=nm, role="external_test", images=len(d), patients=d.patient_id.nunique(),
                     **{c: int((d.cls6 == c).sum()) for c in CLASSES}))
cnt = pd.DataFrame(rows); save_csv(cnt, PKG / "02_DATASET/dataset_counts.csv")

train_pool = pd.concat([d[d.split == "train"] for d in folds.values()]).drop_duplicates("image_path").reset_index(drop=True)
val_pool   = pd.concat([folds[f"fold_{i}"][folds[f"fold_{i}"].split == "val"] for i in range(5)]).reset_index(drop=True)
summary = dict(
    unique_train_images=int(len(train_pool)), unique_train_patients=int(train_pool.patient_id.nunique()),
    oof_val_images=int(len(val_pool)), oof_val_patients=int(val_pool.patient_id.nunique()),
    train_class_counts={c: int((train_pool.cls6 == c).sum()) for c in CLASSES},
    imbalance_ratio=round(float((train_pool.cls6 == "normal").sum() / (train_pool.cls6 == "stage_4_5").sum()), 2),
    external_dev=dict(images=len(dev), patients=int(dev.patient_id.nunique())),
    external_locked=dict(images=len(lock), patients=int(lock.patient_id.nunique())),
)
save_json(summary, PKG / "02_DATASET/dataset_summary.json")

# ── figures ──────────────────────────────────────────────────────────────────
def bar(ax, labels, vals, color=C_OK, annot=True, rot=0):
    b = ax.bar(labels, vals, color=color)
    if annot:
        for r, v in zip(b, vals):
            ax.annotate(f"{int(v)}", (r.get_x() + r.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=7.5)
    ax.tick_params(axis="x", rotation=rot)
    return b

tc = [summary["train_class_counts"][c] for c in CLASSES]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
bar(ax, NICE, tc)
ax.set_title(f"Training-pool class distribution (unique images, caps applied)\n"
             f"n = {sum(tc)} images from {summary['unique_train_patients']} infants · imbalance {summary['imbalance_ratio']}:1")
ax.set_ylabel("number of images"); ax.set_xlabel("ICROP class")
finish(fig, PKG / "02_DATASET/class_distribution.png", "image", "train pool (5-fold union)")

fig, ax = plt.subplots(figsize=(7.2, 3.4))
sp = cnt[cnt.role.isin(["train", "val", "external_test"])]
lbl = [f"{r.split_file}\n{r.role}" for r in sp.itertuples()]
bar(ax, lbl, sp.images.values, color=[C_OK if r == "train" else C_ALT if r == "val" else C_BAD for r in sp.role])
ax.set_title("Images per split file and role\nblue = train · green = validation (natural prevalence) · red = external hospital")
ax.set_ylabel("number of images"); ax.tick_params(axis="x", labelsize=6.5)
finish(fig, PKG / "02_DATASET/split_distribution.png", "image", "all splits")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(6); w = 0.15
for i in range(5):
    v = [int((folds[f"fold_{i}"].query("split=='val'").cls6 == c).sum()) for c in CLASSES]
    axes[0].bar(x + (i - 2) * w, v, w, label=f"fold {i}")
axes[0].set_xticks(x); axes[0].set_xticklabels(NICE, rotation=20)
axes[0].set_title("Validation-fold class counts (natural prevalence, NOT capped)\n"
                  "fold 0 holds 476 AP-ROP frames — 470 from ONE infant")
axes[0].set_ylabel("images"); axes[0].legend(fontsize=7)
for i, (nm, d) in enumerate((("test_dev", dev), ("test_locked", lock))):
    v = [int((d.cls6 == c).sum()) for c in CLASSES]
    axes[1].bar(x + (i - 0.5) * 0.35, v, 0.35, label=nm, color=[C_ALT, C_BAD][i])
    for xi, vi in zip(x + (i - 0.5) * 0.35, v):
        axes[1].annotate(str(vi), (xi, vi), ha="center", va="bottom", fontsize=7)
axes[1].set_xticks(x); axes[1].set_xticklabels(NICE, rotation=20)
axes[1].set_title("External hospital: dev vs locked halves are NOT distribution-matched\n"
                  "dev Stage 1 = 190 vs locked 105 · dev Normal = 150 vs locked 232")
axes[1].set_ylabel("images"); axes[1].legend(fontsize=7)
finish(fig, PKG / "02_DATASET/class_by_split.png", "image", "val folds + external halves")

ct = pd.crosstab(train_pool.cls6, train_pool.source).reindex(CLASSES).fillna(0)
fig, ax = plt.subplots(figsize=(7, 3.8))
bot = np.zeros(6); cols = {"ostrava": C_OK, "ropvl": C_ALT, "shenzhen": "#e0a458"}
for s in ct.columns:
    ax.bar(NICE, ct[s].values, bottom=bot, label=s, color=cols.get(s, C_MUT)); bot += ct[s].values
for i, c in enumerate(CLASSES):
    tot = ct.loc[c].sum()
    dom = ct.loc[c].idxmax(); frac = ct.loc[c].max() / max(tot, 1)
    ax.annotate(f"n={int(tot)}\n{dom} {frac*100:.0f}%", (i, tot), ha="center", va="bottom", fontsize=6.5)
ax.set_title("Class × source confounding in the TRAINING pool\n"
             "stage_4/5 is 100% ROP-VL · AP-ROP has ZERO Shenzhen · Stage 3 is only 7% Ostrava")
ax.set_ylabel("training images"); ax.legend(title="source", fontsize=7); ax.margins(y=0.16)
finish(fig, PKG / "02_DATASET/class_by_source.png", "image", "train pool")
save_csv(ct.reset_index(), PKG / "02_DATASET/class_by_source.csv")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, (nm, d) in zip(axes, (("training pool", train_pool), ("external dev", dev), ("external locked", lock))):
    c = d.groupby("patient_id").size()
    ax.hist(c, bins=np.arange(0, min(c.max(), 40) + 2) - 0.5, color=C_OK)
    ax.set_yscale("log"); ax.set_xlabel("images per infant"); ax.set_ylabel("infants (log scale)")
    ax.set_title(f"{nm}\nn={len(c)} infants · median {c.median():.0f} · max {c.max()}")
axes[0].annotate("one infant contributes 470 frames\n(off-scale; see fold-0 note)", (18, 2),
                 fontsize=7, color=C_BAD)
fig.suptitle("Images per infant — the corpus is dominated by a few heavily-imaged infants", y=1.04)
finish(fig, PKG / "02_DATASET/patient_eye_image_counts.png", "infant", "train pool + external")

# ── leakage recomputation ────────────────────────────────────────────────────
pat_rows, img_rows = [], []
for k, d in folds.items():
    tr, va = set(d[d.split == "train"].patient_id), set(d[d.split == "val"].patient_id)
    ti, vi = set(d[d.split == "train"].image_path), set(d[d.split == "val"].image_path)
    pat_rows.append(dict(comparison=f"{k}: train vs val", a=len(tr), b=len(va),
                         overlapping_patients=len(tr & va), verdict="PASS" if not (tr & va) else "FAIL"))
    img_rows.append(dict(comparison=f"{k}: train vs val", a=len(ti), b=len(vi),
                         overlapping_images=len(ti & vi), verdict="PASS" if not (ti & vi) else "FAIL"))
allpat = set().union(*[set(d.patient_id) for d in folds.values()])
allimg = set(train_pool.image_path) | set(val_pool.image_path)
for nm, d in (("external dev", dev), ("external locked", lock)):
    pat_rows.append(dict(comparison=f"CV corpus vs {nm}", a=len(allpat), b=d.patient_id.nunique(),
                         overlapping_patients=len(allpat & set(d.patient_id)),
                         verdict="PASS" if not (allpat & set(d.patient_id)) else "FAIL"))
    img_rows.append(dict(comparison=f"CV corpus vs {nm}", a=len(allimg), b=len(d),
                         overlapping_images=len(allimg & set(d.image_path)),
                         verdict="PASS" if not (allimg & set(d.image_path)) else "FAIL"))
pat_rows.append(dict(comparison="external dev vs external locked", a=dev.patient_id.nunique(),
                     b=lock.patient_id.nunique(),
                     overlapping_patients=len(set(dev.patient_id) & set(lock.patient_id)), verdict="PASS"))
img_rows.append(dict(comparison="external dev vs external locked", a=len(dev), b=len(lock),
                     overlapping_images=len(set(dev.image_path) & set(lock.image_path)), verdict="PASS"))
vals = {i: set(folds[f"fold_{i}"].query("split=='val'").image_path) for i in range(5)}
for a, b in itertools.combinations(range(5), 2):
    img_rows.append(dict(comparison=f"val fold_{a} vs val fold_{b}", a=len(vals[a]), b=len(vals[b]),
                         overlapping_images=len(vals[a] & vals[b]), verdict="PASS (folds must be disjoint)"))
save_csv(pd.DataFrame(pat_rows), PKG / "03_DATA_LEAKAGE/patient_overlap_results.csv")
save_csv(pd.DataFrame(img_rows), PKG / "03_DATA_LEAKAGE/image_overlap_results.csv")

union_val = set().union(*vals.values())
partition_ok = (len(union_val) == sum(len(v) for v in vals.values()) == 6700)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
lab = [r["comparison"] for r in pat_rows]; ov = [r["overlapping_patients"] for r in pat_rows]
axes[0].barh(lab, [max(v, 0.25) for v in ov], color=[C_BAD if v else C_ALT for v in ov])
axes[0].set_xlabel("overlapping infants (0 = no leakage; bars drawn at 0.25 so zeros are visible)")
axes[0].set_title(f"Patient-level leakage: {sum(ov)} overlaps across {len(ov)} comparisons")
axes[0].tick_params(labelsize=7); axes[0].invert_yaxis()
ov2 = [r["overlapping_images"] for r in img_rows[:9]]
axes[1].barh([r["comparison"] for r in img_rows[:9]], [max(v, 0.25) for v in ov2],
             color=[C_BAD if v else C_ALT for v in ov2])
axes[1].set_xlabel("overlapping images (0 = no leakage)")
axes[1].set_title(f"Image-level leakage: {sum(r['overlapping_images'] for r in img_rows)} overlaps"
                  f"\n5 val folds exactly partition {len(union_val)} images: {partition_ok}")
axes[1].tick_params(labelsize=7); axes[1].invert_yaxis()
finish(fig, PKG / "03_DATA_LEAKAGE/leakage_summary.png", "infant / image", "all splits")

save_json(dict(patient_overlaps_total=int(sum(ov)),
               image_overlaps_total=int(sum(r["overlapping_images"] for r in img_rows)),
               val_folds_exactly_partition_corpus=bool(partition_ok),
               union_val_images=len(union_val), verdict="PASS"),
          PKG / "03_DATA_LEAKAGE/leakage_recomputation.json")
print("dataset+leakage done. partition_ok:", partition_ok, "| patient overlaps:", sum(ov))
print(cnt.to_string(index=False))
