"""17_USER_FACING — the ten-figure story, written for a non-ML reader."""
import sys, shutil, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
sys.path.insert(0, str(ROOT))
from sklearn.metrics import confusion_matrix
U = PKG / "17_USER_FACING"

oof = np.load(PKG / "07_TEST_RESULTS/current_model_oof_predictions.npz", allow_pickle=True)
y, p, pr = oof["y_true"], oof["y_pred"], oof["prob6"].astype(float)
lk = load_npz(ROOT / "results/rop/locked_eval/structured_s42_ckpt0.npz"); ly, lp, lpr = lk[0], lk[1], lk[2].astype(float)
hist = pd.read_csv(PKG / "05_TRAINING/training_history.csv")
cal  = pd.read_csv(PKG / "10_CONFIDENCE_CALIBRATION/calibration_metrics.csv")
rel  = pd.read_csv(PKG / "10_CONFIDENCE_CALIBRATION/reliability_bins.csv")
agg  = pd.read_csv(PKG / "11_AGGREGATION_AND_VIEW_ANALYSIS/aggregation_results.csv")
reg  = pd.read_csv(PKG / "11_AGGREGATION_AND_VIEW_ANALYSIS/region_results.csv")
cnt  = pd.read_csv(PKG / "02_DATASET/dataset_counts.csv")
pcm  = pd.read_csv(PKG / "07_TEST_RESULTS/per_class_metrics/per_class_metrics.csv")
ecat = pd.read_csv(PKG / "09_ERROR_ANALYSIS/error_categories.csv")

# 01 — the data
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
tr = cnt[(cnt.split_file == "fold_0") & (cnt.role == "train")]
tot = [int(cnt[(cnt.role == "train")][c].max()) for c in CLASSES]
sumjson = json.loads((PKG/"02_DATASET/dataset_summary.json").read_text())
vals = [sumjson["train_class_counts"][c] for c in CLASSES]
b = axes[0].bar(NICE, vals, color=[C_ALT] + [C_OK]*4 + [C_BAD])
for r, v in zip(b, vals): axes[0].annotate(f"{v}", (r.get_x()+r.get_width()/2, v), ha="center", va="bottom", fontsize=8)
axes[0].set_ylabel("number of retinal photographs used for TRAINING")
axes[0].tick_params(axis="x", rotation=18)
axes[0].set_title("What the model was taught from\n"
                  f"{sum(vals):,} photographs of {sumjson['unique_train_patients']:,} premature babies, 3 hospitals\n"
                  "The rarest disease category has only 89 photographs.")
lbl = ["Training hospitals\n(3 sites, used to build the model)", "New hospital\n(never seen during training)"]
axes[1].bar(lbl, [6700, 661], color=[C_OK, C_BAD])
for i, v in enumerate([6700, 661]): axes[1].annotate(f"{v:,} photographs", (i, v), ha="center", va="bottom", fontsize=9)
axes[1].set_ylabel("photographs used for testing")
axes[1].set_title("How the model was tested\nNo baby appears in both groups — verified, zero overlap")
finish(fig, U / "01_dataset.png", "image", "training pool + test sets")

# 02 — pipeline
shutil.copy(PKG / "04_PREPROCESSING/preprocessing_pipeline.png", U / "02_model_pipeline.png")

# 03 — training behaviour
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
for f in range(5):
    h = hist[hist.run == f"fold{f}"]
    axes[0].plot(h.epoch, h.train_loss_total, lw=1.3, label=f"training run {f+1}")
    axes[1].plot(h.epoch, h.val_macro_f1*100, "o-", ms=3, lw=1.2, label=f"training run {f+1}")
axes[0].set_xlabel("training pass over the data (epoch)"); axes[0].set_ylabel("training error (lower = better)")
axes[0].legend(fontsize=7)
axes[0].set_title("The model IS learning\nTraining error falls smoothly and never diverges")
axes[1].set_xlabel("training pass over the data (epoch)")
axes[1].set_ylabel("score on held-out photos (macro-F1, %)")
axes[1].set_ylim(0, 100)
axes[1].set_title("But its score swings wildly between passes\nUp to 31 points between two consecutive epochs —\n"
                  "so small score differences between models mean little")
finish(fig, U / "03_training_behavior.png", "image", "training + validation")

# 04 — overall performance
fig, ax = plt.subplots(figsize=(9, 4.2))
im = full_metrics(y, p, pr); ex = full_metrics(ly, lp, lpr)
mets = [("Overall accuracy", im["accuracy"], ex["accuracy"]),
        ("Balanced accuracy\n(fair across rare classes)", im["balanced_accuracy"], ex["balanced_accuracy"]),
        ("Agreement with the grader\n(QWK, 0=chance 1=perfect)", im["qwk_ordinal_5class"], ex["qwk_ordinal_5class"]),
        ("Spotting ANY disease\n(AUC of the screening score)", im["any_rop_auc"], ex["any_rop_auc"])]
x = np.arange(len(mets)); w = .38
ax.bar(x-w/2, [m[1] for m in mets], w, label="Training hospitals (6,700 photos)", color=C_OK)
ax.bar(x+w/2, [m[2] for m in mets], w, label="NEW hospital (661 photos)", color=C_BAD)
for i, m in enumerate(mets):
    ax.annotate(f"{m[1]:.2f}", (i-w/2, m[1]), ha="center", va="bottom", fontsize=8)
    ax.annotate(f"{m[2]:.2f}", (i+w/2, m[2]), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in mets], fontsize=7.5)
ax.set_ylim(0, 1.1); ax.set_ylabel("score (0 = useless, 1 = perfect)"); ax.legend(fontsize=8)
ax.set_title("How well does it do? — one photograph at a time\n"
             "Every measure drops at a hospital the model has never seen")
finish(fig, U / "04_overall_performance.png", "image", "internal 5-fold OOF vs external locked")

# 05 — confusion matrix, plain language
cm = confusion_matrix(y, p, labels=range(6)); M = cm/np.maximum(cm.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(7, 5.6))
imh = ax.imshow(M, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{M[i,j]*100:.0f}%\n{cm[i,j]} photos", ha="center", va="center", fontsize=7,
                color="white" if M[i, j] > .55 else "black")
ax.set_xticks(range(6)); ax.set_xticklabels(NICE, rotation=25, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels([f"{n}\n({s} photos)" for n, s in zip(NICE, cm.sum(1))], fontsize=7.5)
ax.set_xlabel("What the model SAID"); ax.set_ylabel("What the photograph ACTUALLY was")
ax.set_title("Where the model agrees and disagrees with the human grade\n"
             "Each row adds to 100%. The diagonal is 'correct'.\n"
             "Training hospitals, 6,700 photographs, one photo at a time.")
fig.colorbar(imh, ax=ax, shrink=.8, label="share of that true class")
finish(fig, U / "05_confusion_matrix.png", "image", "internal 5-fold OOF")

# 06 — per class
fig, ax = plt.subplots(figsize=(9.4, 4.4))
x = np.arange(6); w = .38
ax.bar(x-w/2, pcm.recall, w, yerr=[pcm.recall-pcm.recall_lo, pcm.recall_hi-pcm.recall], capsize=3,
       label="Of the photos that really ARE this class, how many did it find? (sensitivity)", color=C_OK)
ax.bar(x+w/2, pcm.precision, w, yerr=[pcm.precision-pcm.precision_lo, pcm.precision_hi-pcm.precision], capsize=3,
       label="When it SAYS this class, how often is it right? (precision)", color=C_ALT)
for i in x: ax.annotate(f"{int(pcm.support[i])} photos", (i, 1.06), ha="center", fontsize=7, color="#444")
ax.set_xticks(x); ax.set_xticklabels(NICE, rotation=18); ax.set_ylim(0, 1.18)
ax.set_ylabel("score (0–1)"); ax.legend(fontsize=7.5, loc="lower left")
ax.set_title("Performance for every category — bars show 95% uncertainty\n"
             "Normal is reliable. Stage 1 is the weakest: when the model says 'Stage 1' it is right only 38% of the time.")
finish(fig, U / "06_per_class_performance.png", "image", "internal 5-fold OOF")

# 08 — confidence
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
conf = pr.max(1); ok = y == p
axes[0].hist(conf[ok], bins=np.linspace(0, 1, 31), color=C_ALT, alpha=.85, label=f"model was RIGHT ({int(ok.sum()):,})")
axes[0].hist(conf[~ok], bins=np.linspace(0, 1, 31), color=C_BAD, alpha=.85, label=f"model was WRONG ({int((~ok).sum()):,})")
axes[0].set_yscale("log"); axes[0].set_xlabel("how sure the model said it was (0–1)")
axes[0].set_ylabel("number of photographs (log scale)"); axes[0].legend(fontsize=8)
axes[0].set_title("The model is often very sure when it is wrong\n"
                  f"359 photographs were called with >90% certainty AND were wrong")
b = ecat.sort_values("n", ascending=True)
cols = [C_BAD if "CRITICAL" in c else C_MUT for c in b.error_category]
axes[1].barh(range(len(b)), b.mean_confidence, color=cols)
axes[1].set_yticks(range(len(b)))
axes[1].set_yticklabels([c.replace("CLINICALLY CRITICAL: ", "⚠ ").replace(" ", "\n", 0)[:46] for c in b.error_category], fontsize=6.5)
for i, r in enumerate(b.itertuples()):
    axes[1].annotate(f"{r.mean_confidence:.2f}  (n={r.n})", (r.mean_confidence, i), va="center",
                     fontsize=7, xytext=(3, 0), textcoords="offset points")
axes[1].set_xlim(0, 1.15); axes[1].set_xlabel("average certainty when making this kind of mistake")
axes[1].set_title("The most DANGEROUS mistake is the one it is most sure about\n"
                  "'a diseased eye called Normal' has the highest average certainty of all error types")
finish(fig, U / "08_confidence.png", "image", "internal 5-fold OOF")

# 09 — calibration
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
for nm, c, lab in ((f"CURRENT structured (s42) | internal 5-fold OOF", C_OK, "Training hospitals"),
                   (f"CURRENT structured (served fold-0 ckpt) | external locked", C_BAD, "NEW hospital")):
    b = rel[rel.series == nm].dropna(subset=["conf"])
    axes[0].plot(b.conf, b.acc, "o-", color=c, ms=4, label=lab)
axes[0].plot([0, 1], [0, 1], "--", color=C_MUT, label="what a trustworthy 'certainty' looks like")
axes[0].set_xlabel("certainty the model reported"); axes[0].set_ylabel("how often it was actually right")
axes[0].legend(fontsize=7.5); axes[0].set_xlim(0, 1.02); axes[0].set_ylim(0, 1.02)
axes[0].set_title("Can you trust the percentage the app shows?\n"
                  "Points below the dashed line = the model is over-confident.\nAt the new hospital it is far below.")
c1 = cal[cal.name == "CURRENT structured (s42)"].iloc[0]
c2 = cal[cal.name == "CURRENT structured (served fold-0 ckpt)"].iloc[0]
c3 = cal[cal.name == "flat softmax baseline (s42)"].iloc[0]
lbls = ["CURRENT model\nTraining hospitals", "CURRENT model\nNEW hospital", "simpler baseline model\nTraining hospitals"]
v = [c1.ece_15bin, c2.ece_15bin, c3.ece_15bin]
axes[1].bar(lbls, v, color=[C_OK, C_BAD, C_MUT])
for i, vv in enumerate(v): axes[1].annotate(f"{vv:.3f}", (i, vv), ha="center", va="bottom", fontsize=9)
axes[1].set_ylabel("calibration error (0 = perfectly trustworthy certainty)")
axes[1].tick_params(axis="x", labelsize=7.5)
axes[1].set_title("Calibration error\nThe simpler baseline gives MORE trustworthy percentages\nthan the current model, at the same accuracy.")
finish(fig, U / "09_calibration.png", "image", "internal + external")

# 10 — external validation + the view story
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
cmp_ = pd.read_csv(PKG / "13_EXTERNAL_GENERALIZATION/external_metrics.csv")
x = np.arange(6); w = .38
axes[0].bar(x-w/2, cmp_.internal_recall, w, label="Training hospitals", color=C_OK)
axes[0].bar(x+w/2, cmp_.external_recall, w, label="NEW hospital", color=C_BAD)
for i in x:
    axes[0].annotate(f"{int(cmp_.external_support[i])}", (i+w/2, cmp_.external_recall[i]+.03), ha="center",
                     fontsize=6.5, color=C_BAD if cmp_.external_support[i] < 30 else "#444")
axes[0].set_xticks(x); axes[0].set_xticklabels(NICE, rotation=18); axes[0].set_ylim(0, 1.15)
axes[0].set_ylabel("how many of this class it finds (sensitivity)"); axes[0].legend(fontsize=7.5)
axes[0].set_title("At a new hospital the model still spots healthy eyes —\nbut finds only ~1 in 4 diseased eyes, per photograph.\n"
                  "(numbers above red bars = photographs available; <30 is too few to judge)")
r = reg[reg.model_head == "CURRENT structured"].set_index("region")
ordr = ["posterior pole", "nasal", "temporal"]
vals = [r.loc[o, "qwk_mean"] for o in ordr]
eye = float(agg.query("model_head=='CURRENT structured' and rule=='max (protocol rule)'").qwk_mean.iloc[0])
allv = float(r.loc["ALL REGIONS", "qwk_mean"])
axes[1].bar(range(3), vals, color=[C_BAD, C_BAD, C_ALT])
axes[1].axhline(allv, ls=":", color=C_MUT); axes[1].axhline(eye, ls="--", color=C_OK)
axes[1].annotate(f"all photos pooled = {allv:.2f}", (-0.4, allv+.02), fontsize=7, color="#555")
axes[1].annotate(f"combining an eye's photos = {eye:.2f}", (-0.4, eye+.02), fontsize=7, color=C_OK)
for i, v in enumerate(vals):
    axes[1].annotate(f"{v:.2f}\n(n={int(r.loc[ordr[i],'n_images'])})", (i, v+.03), ha="center", fontsize=8)
axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["back of the eye\n(posterior pole)", "nose side\n(nasal)", "outer side\n(temporal)"])
axes[1].set_ylabel("agreement with the human grade (QWK)"); axes[1].set_ylim(0, 1)
axes[1].set_title("WHY the per-photograph score looks bad\nROP grows on the OUTER edge of the retina. Photos of the\n"
                  "outer side score as well as combining all of an eye's photos.")
finish(fig, U / "10_external_validation.png", "image / eye", "external locked")
print("user-facing figures written")
