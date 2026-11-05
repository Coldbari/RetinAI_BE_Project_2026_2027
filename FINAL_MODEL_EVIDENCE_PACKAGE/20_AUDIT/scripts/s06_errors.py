"""09_ERROR_ANALYSIS — error atlas + de-identified image galleries. CURRENT MODEL only."""
import sys, hashlib
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
sys.path.insert(0, str(ROOT))
from PIL import Image
from sklearn.metrics import confusion_matrix

SALT = "retinai-evidence-package-v1"
def anon(s):   # stable, non-reversible case id — no filename, no infant id, no site code
    return "CASE-" + hashlib.sha256((SALT + str(s)).encode()).hexdigest()[:10].upper()

val = pd.concat([pd.read_csv(PREP / f"fold_{f}.csv").query("split=='val'").assign(fold=f)
                 for f in range(5)]).reset_index(drop=True)
oof = np.load(PKG / "07_TEST_RESULTS/current_model_oof_predictions.npz", allow_pickle=True)
y, p, pr = oof["y_true"], oof["y_pred"], oof["prob6"].astype(float)
assert (y == val.label.values).all()

conf = pr.max(1)
atlas = pd.DataFrame(dict(
    case_id=[anon(v) for v in val.image_path], fold=val.fold.values, source_site=val.source.values,
    true_class=[CLASSES[i] for i in y], predicted_class=[CLASSES[i] for i in p],
    correct=(y == p), confidence=conf,
    prob_normal=pr[:, 0], prob_stage1=pr[:, 1], prob_stage2=pr[:, 2],
    prob_stage3=pr[:, 3], prob_stage45=pr[:, 4], prob_arop=pr[:, 5],
    any_rop_score=pr[:, 1:].sum(1)))
def cat(t, q, c):
    if t == q: return "correct"
    if q == 0 and t in (1, 2, 3): return "CLINICALLY CRITICAL: disease stage called Normal"
    if q == 0 and t == 5: return "CLINICALLY CRITICAL: AP-ROP called Normal"
    if q == 0 and t == 4: return "CLINICALLY CRITICAL: stage 4/5 called Normal"
    if t == 0: return "false positive: Normal called diseased"
    if abs(t - q) == 1 and t < 5 and q < 5: return "adjacent-stage confusion"
    if t == 5 or q == 5: return "AP-ROP / stage confusion"
    return "non-adjacent stage error"
atlas["error_category"] = [cat(int(a), int(b), c) for a, b, c in zip(y, p, conf)]
atlas["high_confidence"] = atlas.confidence > 0.90
atlas = atlas.sort_values(["correct", "confidence"], ascending=[True, False]).reset_index(drop=True)
save_csv(atlas, PKG / "09_ERROR_ANALYSIS/error_atlas_all_predictions.csv")
save_csv(atlas[~atlas.correct], PKG / "09_ERROR_ANALYSIS/error_summary.csv")
hc = atlas[(~atlas.correct) & atlas.high_confidence]
save_csv(hc, PKG / "09_ERROR_ANALYSIS/high_confidence_errors.csv")
save_csv(hc, PKG / "10_CONFIDENCE_CALIBRATION/high_confidence_error_analysis.csv")

byc = (atlas[~atlas.correct].groupby("error_category")
       .agg(n=("case_id", "size"), mean_confidence=("confidence", "mean"),
            n_high_confidence=("high_confidence", "sum")).reset_index()
       .sort_values("n", ascending=False))
byc["pct_of_all_errors"] = byc.n / len(atlas[~atlas.correct]) * 100
save_csv(byc, PKG / "09_ERROR_ANALYSIS/error_categories.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
axes[0].barh(byc.error_category, byc.n, color=[C_BAD if "CRITICAL" in c else C_MUT for c in byc.error_category])
for i, r in enumerate(byc.itertuples()):
    axes[0].annotate(f"{r.n}  ({r.pct_of_all_errors:.0f}% of errors, {r.n_high_confidence} at conf>0.90)",
                     (r.n, i), va="center", fontsize=7, xytext=(4, 0), textcoords="offset points")
axes[0].set_xlabel(f"number of errors (total errors = {len(atlas[~atlas.correct])} of {len(atlas)} images)")
axes[0].tick_params(labelsize=7); axes[0].invert_yaxis(); axes[0].margins(x=0.32)
axes[0].set_title("Error taxonomy, CURRENT MODEL, pooled 5-fold out-of-fold\nred = a diseased eye called Normal")
cmw = confusion_matrix(y[(y != p) & (conf > .9)], p[(y != p) & (conf > .9)], labels=range(6))
axes[1].imshow(cmw, cmap="Reds"); axes[1].grid(False)
for i in range(6):
    for j in range(6):
        if cmw[i, j]: axes[1].text(j, i, cmw[i, j], ha="center", va="center", fontsize=8,
                                   color="white" if cmw[i, j] > cmw.max()*.55 else "black")
axes[1].set_xticks(range(6)); axes[1].set_xticklabels(NICE, rotation=25, ha="right", fontsize=7.5)
axes[1].set_yticks(range(6)); axes[1].set_yticklabels(NICE, fontsize=7.5)
axes[1].set_xlabel("PREDICTED (wrongly)"); axes[1].set_ylabel("TRUE")
axes[1].set_title(f"HIGH-CONFIDENCE errors only (confidence > 0.90), n={int(cmw.sum())}\n"
                  "72 Stage 1 and 39 AP-ROP eyes confidently called Normal")
finish(fig, PKG / "09_ERROR_ANALYSIS/confusion_error_matrix.png", "image", "5-fold OOF")

# ── de-identified galleries ──────────────────────────────────────────────────
def gallery(sub, outdir, title, n=12):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    sub = sub.head(n)
    if not len(sub): return
    idx = [atlas.index.get_loc(i) for i in sub.index]
    paths = [val.image_path.values[atlas.index[i]] for i in range(len(atlas))]
    rows = int(np.ceil(len(sub) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(11, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    man = []
    for ax, (_, r) in zip(axes, sub.iterrows()):
        src = ROOT / paths[atlas.index.get_loc(_)]
        img = Image.open(src).convert("RGB"); img.thumbnail((420, 420))
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"{r.case_id}\ntrue {NICE[CLASSES.index(r.true_class)]} → pred "
                     f"{NICE[CLASSES.index(r.predicted_class)]}\nconfidence {r.confidence:.3f}",
                     fontsize=7.5, color=C_BAD if not r.correct else C_ALT)
        img.save(outdir / f"{r.case_id}.png")
        man.append(dict(case_id=r.case_id, file=f"{r.case_id}.png", true_class=r.true_class,
                        predicted_class=r.predicted_class, confidence=round(float(r.confidence), 4),
                        error_category=r.error_category, source_site=r.source_site))
    for ax in axes[len(sub):]: ax.axis("off")
    fig.suptitle(title, y=1.0, fontsize=10)
    finish(fig, outdir / "_contact_sheet.png", "image", "5-fold OOF")
    save_csv(pd.DataFrame(man), outdir / "manifest.csv")

gallery(atlas[(~atlas.correct) & atlas.high_confidence & atlas.error_category.str.contains("CRITICAL")],
        PKG / "09_ERROR_ANALYSIS/false_negative_cases",
        "FALSE NEGATIVES — diseased eyes confidently called Normal (CURRENT MODEL, internal OOF)")
gallery(atlas[(~atlas.correct) & (atlas.true_class == "normal")].sort_values("confidence", ascending=False),
        PKG / "09_ERROR_ANALYSIS/false_positive_cases",
        "FALSE POSITIVES — healthy eyes called diseased (CURRENT MODEL, internal OOF)")
gallery(atlas[(~atlas.correct) & atlas.high_confidence],
        PKG / "09_ERROR_ANALYSIS/high_confidence_errors",
        "HIGH-CONFIDENCE ERRORS — the model was >0.90 sure and wrong (CURRENT MODEL, internal OOF)")
gallery(atlas[atlas.correct].sort_values("confidence", ascending=False),
        PKG / "09_ERROR_ANALYSIS/correct_predictions",
        "CORRECT, high-confidence predictions (CURRENT MODEL, internal OOF)")
gallery(atlas[~atlas.correct].sort_values("confidence"),
        PKG / "09_ERROR_ANALYSIS/incorrect_predictions",
        "LOW-CONFIDENCE errors — the model was unsure and wrong (CURRENT MODEL, internal OOF)")
import shutil
shutil.copytree(PKG / "09_ERROR_ANALYSIS/high_confidence_errors",
                PKG / "17_USER_FACING/07_error_examples", dirs_exist_ok=True)
print(byc.to_string(index=False))
print("\nhigh-conf errors:", len(hc), "| galleries written")
