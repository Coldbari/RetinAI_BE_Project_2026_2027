"""06_VALIDATION + 07_TEST_RESULTS (internal CV) for the CURRENT MODEL.
Every number recomputed from the committed raw per-sample predictions.
QWK is recomputed under BOTH definitions and the correction is recorded, never silently applied."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
sys.path.insert(0, str(ROOT))
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

ARMS = {"CURRENT structured s42": CUR["cv_dir"], "structured s1337 (seed replicate)": SEED_B,
        "flat softmax s42 (baseline)": FLAT, "flat softmax s1337 (baseline)": FLAT_B}
valdf = {i: pd.read_csv(PREP / f"fold_{i}.csv").query("split=='val'").reset_index(drop=True) for i in range(5)}

# ── per-fold, per-arm ────────────────────────────────────────────────────────
rows, pooled = [], {}
for arm, d in ARMS.items():
    yy, pp, bb = [], [], []
    for f in range(5):
        y, p, pr = load_npz(Path(d) / f"fold_{f}.npz")
        assert len(y) == len(valdf[f]), f"{arm} fold{f}: {len(y)} vs manifest {len(valdf[f])}"
        assert (y == valdf[f].label.values).all(), f"{arm} fold{f}: label order mismatch vs manifest"
        m = full_metrics(y, p, pr)
        rows.append(dict(arm=arm, fold=f, **{k: v for k, v in m.items() if k != "per_class"}))
        yy.append(y); pp.append(p); bb.append(pr)
    pooled[arm] = (np.concatenate(yy), np.concatenate(pp), np.concatenate(bb))
fold_df = pd.DataFrame(rows)
save_csv(fold_df, PKG / "06_VALIDATION/fold_results.csv")

agg = fold_df.groupby("arm").agg(["mean", "std"])
summ = []
for arm in ARMS:
    s = fold_df[fold_df.arm == arm]
    y, p, pr = pooled[arm]
    pm = full_metrics(y, p, pr)
    summ.append(dict(arm=arm,
        macro_f1_all6_mean=s.macro_f1_all6.mean(), macro_f1_all6_sd_over_folds=s.macro_f1_all6.std(ddof=1),
        qwk_ordinal_mean=s.qwk_ordinal_5class.mean(), qwk_ordinal_sd_over_folds=s.qwk_ordinal_5class.std(ddof=1),
        qwk_6class_legacy_mean=s.qwk_6class_legacy.mean(), qwk_6class_legacy_sd_over_folds=s.qwk_6class_legacy.std(ddof=1),
        accuracy_mean=s.accuracy.mean(), balanced_accuracy_mean=s.balanced_accuracy.mean(),
        pooled_macro_f1_all6=pm["macro_f1_all6"], pooled_qwk_ordinal=pm["qwk_ordinal_5class"],
        pooled_qwk_6class_legacy=pm["qwk_6class_legacy"], pooled_kappa_unweighted=pm["kappa_unweighted"],
        pooled_accuracy=pm["accuracy"], pooled_balanced_accuracy=pm["balanced_accuracy"],
        pooled_any_rop_auc=pm.get("any_rop_auc"), pooled_any_rop_ap=pm.get("any_rop_ap"), n=pm["n"]))
summ_df = pd.DataFrame(summ)
save_csv(summ_df, PKG / "06_VALIDATION/validation_metrics.csv")
save_csv(summ_df, PKG / "07_TEST_RESULTS/metrics_summary.csv")

# ── seed comparison ──────────────────────────────────────────────────────────
sc = []
for head, a, b in (("structured", "CURRENT structured s42", "structured s1337 (seed replicate)"),
                   ("flat", "flat softmax s42 (baseline)", "flat softmax s1337 (baseline)")):
    x = fold_df[fold_df.arm == a].set_index("fold"); z = fold_df[fold_df.arm == b].set_index("fold")
    for f in range(5):
        sc.append(dict(head=head, fold=f, seed42=x.loc[f, "macro_f1_all6"], seed1337=z.loc[f, "macro_f1_all6"],
                       abs_diff=abs(x.loc[f, "macro_f1_all6"] - z.loc[f, "macro_f1_all6"])))
save_csv(pd.DataFrame(sc), PKG / "06_VALIDATION/seed_comparison.csv")

# ── QWK correction ledger ────────────────────────────────────────────────────
ql = []
for arm in ARMS:
    s = fold_df[fold_df.arm == arm]
    for f in range(5):
        r = s[s.fold == f].iloc[0]
        ql.append(dict(arm=arm, scope=f"CV fold {f}", n=int(r.n),
                       previously_reported_qwk_6class=r.qwk_6class_legacy,
                       corrected_qwk_ordinal_5class=r.qwk_ordinal_5class,
                       delta=r.qwk_ordinal_5class - r.qwk_6class_legacy,
                       n_pairs_used_after_excluding_AP_ROP=int(r.qwk_ordinal_n_pairs)))
    ql.append(dict(arm=arm, scope="CV mean over 5 folds", n=int(s.n.sum()),
                   previously_reported_qwk_6class=s.qwk_6class_legacy.mean(),
                   corrected_qwk_ordinal_5class=s.qwk_ordinal_5class.mean(),
                   delta=s.qwk_ordinal_5class.mean() - s.qwk_6class_legacy.mean(),
                   n_pairs_used_after_excluding_AP_ROP=int(s.qwk_ordinal_n_pairs.sum())))
save_csv(pd.DataFrame(ql), PKG / "07_TEST_RESULTS/qwk_analysis/qwk_correction_ledger.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
s = fold_df[fold_df.arm == "CURRENT structured s42"]
x = np.arange(5); w = 0.38
axes[0].bar(x - w/2, s.qwk_6class_legacy, w, label="as previously computed (AP-ROP as ordinal rank 5)", color=C_MUT)
axes[0].bar(x + w/2, s.qwk_ordinal_5class, w, label="CORRECTED (5 ordinal stages only, as the config declares)", color=C_OK)
for i in x:
    axes[0].annotate(f"{s.qwk_6class_legacy.iloc[i]:.3f}", (i-w/2, s.qwk_6class_legacy.iloc[i]), ha="center", va="bottom", fontsize=7)
    axes[0].annotate(f"{s.qwk_ordinal_5class.iloc[i]:.3f}", (i+w/2, s.qwk_ordinal_5class.iloc[i]), ha="center", va="bottom", fontsize=7)
axes[0].set_xticks(x); axes[0].set_xticklabels([f"fold {i}\nn={int(s.n.iloc[i])}" for i in x])
axes[0].set_ylabel("quadratic-weighted kappa"); axes[0].set_ylim(0, 1.05)
axes[0].set_title("QWK definition correction, per cross-validation fold\n"
                  "fold 0's apparent collapse was AP-ROP being scored as a stage")
axes[0].legend(fontsize=6.5, loc="lower right")
for arm, c in zip(ARMS, [C_OK, "#4f9ecf", C_BAD, "#e08a6a"]):
    s2 = fold_df[fold_df.arm == arm]
    axes[1].errorbar([arm.split(" (")[0]], [s2.qwk_ordinal_5class.mean()],
                     yerr=[s2.qwk_ordinal_5class.std(ddof=1)], fmt="o", color=c, capsize=4)
    axes[1].errorbar([arm.split(" (")[0]], [s2.qwk_6class_legacy.mean()],
                     yerr=[s2.qwk_6class_legacy.std(ddof=1)], fmt="s", color=c, alpha=0.4, capsize=4)
axes[1].set_ylabel("QWK, mean ± SD over 5 folds"); axes[1].tick_params(axis="x", rotation=18, labelsize=7)
axes[1].set_title("Circles = corrected 5-class QWK · faded squares = legacy 6-class\n"
                  "Correction removes ~5× of the fold-to-fold spread")
axes[1].set_ylim(0, 1.05)
finish(fig, PKG / "07_TEST_RESULTS/qwk_analysis/qwk_correction.png", "image", "5-fold out-of-fold validation")

# ── pooled confusion + per-class, CURRENT MODEL ──────────────────────────────
y, p, pr = pooled["CURRENT structured s42"]
groups = np.concatenate([valdf[f].patient_id.values for f in range(5)])
np.savez_compressed(PKG / "07_TEST_RESULTS/current_model_oof_predictions.npz",
                    y_true=y, y_pred=p, prob6=pr, patient_id=groups)
cm = confusion_matrix(y, p, labels=range(6))

def plot_cm(cm, path, title, normalize=None, unit="image", split="5-fold OOF"):
    M = cm.astype(float)
    if normalize == "true": M = M / np.maximum(M.sum(1, keepdims=True), 1)
    if normalize == "pred": M = M / np.maximum(M.sum(0, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() if M.max() else 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i,j]}" if normalize is None else f"{M[i,j]*100:.1f}%\n({cm[i,j]})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] > M.max() * 0.55 else "black")
    sup = cm.sum(1)
    ax.set_xticks(range(6)); ax.set_xticklabels(NICE, rotation=25, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels([f"{n}\n(n={s})" for n, s in zip(NICE, sup)], fontsize=7.5)
    ax.set_xlabel("PREDICTED class"); ax.set_ylabel("TRUE class (with support)")
    ax.set_title(title); ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.8)
    return finish(fig, path, unit, split)

plot_cm(cm, PKG / "07_TEST_RESULTS/confusion_matrix/confusion_matrix_counts.png",
        f"Confusion matrix — counts\nCURRENT MODEL, pooled 5-fold out-of-fold, n={len(y)} images")
plot_cm(cm, PKG / "07_TEST_RESULTS/confusion_matrix/confusion_matrix_row_normalized.png",
        f"Confusion matrix — row-normalised (recall per true class)\nCURRENT MODEL, pooled 5-fold out-of-fold, n={len(y)} images",
        normalize="true")
plot_cm(cm, PKG / "07_TEST_RESULTS/confusion_matrix/confusion_matrix_col_normalized.png",
        f"Confusion matrix — column-normalised (precision per predicted class)\nCURRENT MODEL, pooled 5-fold OOF, n={len(y)}",
        normalize="pred")
np.savetxt(PKG / "07_TEST_RESULTS/confusion_matrix/confusion_matrix_counts.csv", cm, fmt="%d",
           delimiter=",", header=",".join(CLASSES), comments="")

pm = full_metrics(y, p, pr)
pc = pd.DataFrame([dict(cls=CLASSES[i], nice=NICE[i], **pm["per_class"][CLASSES[i]]) for i in range(6)])
# patient-clustered CI per class recall
ci = []
for i in range(6):
    f_rec = lambda yy, pp, i=i: float((pp[yy == i] == i).mean()) if (yy == i).any() else np.nan
    pt, lo, hi = cluster_bootstrap(y, p, groups, f_rec, n=1000)
    f_pre = lambda yy, pp, i=i: float((yy[pp == i] == i).mean()) if (pp == i).any() else np.nan
    pt2, lo2, hi2 = cluster_bootstrap(y, p, groups, f_pre, n=1000)
    ci.append(dict(cls=CLASSES[i], recall_lo=lo, recall_hi=hi,
                   precision_lo=lo2, precision_hi=hi2))
pc = pc.merge(pd.DataFrame(ci), on="cls")
save_csv(pc, PKG / "07_TEST_RESULTS/per_class_metrics/per_class_metrics.csv")

fig, ax = plt.subplots(figsize=(8.2, 4.2))
x = np.arange(6); w = 0.27
for off, col, lab, lo, hi in ((-w, C_OK, "precision", "precision_lo", "precision_hi"),
                              (0, C_ALT, "recall", "recall_lo", "recall_hi")):
    v = pc[lab].values
    ax.bar(x + off, v, w, label=lab, color=col,
           yerr=[v - pc[lo].values, pc[hi].values - v], capsize=3, error_kw=dict(lw=0.8))
ax.bar(x + w, pc.f1.values, w, label="F1", color=C_MUT)
for i in x:
    ax.annotate(f"n={int(pc.support.iloc[i])}", (i, 1.04), ha="center", fontsize=7, color="#444")
ax.set_xticks(x); ax.set_xticklabels(NICE, rotation=18); ax.set_ylim(0, 1.15)
ax.set_ylabel("score (0–1)"); ax.legend(fontsize=8, ncol=3, loc="lower left")
ax.set_title("Per-class performance, CURRENT MODEL — pooled 5-fold out-of-fold\n"
             "error bars = 95% patient-clustered bootstrap · Stage 1 precision 0.38 is the weakest cell")
finish(fig, PKG / "07_TEST_RESULTS/per_class_metrics/per_class_performance.png", "image", "5-fold OOF")

# ── statistical uncertainty ──────────────────────────────────────────────────
st = []
for nm, fn in (("macro_f1_all6", lambda a, b: f1_score(a, b, labels=list(range(6)), average="macro", zero_division=0)),
               ("qwk_ordinal_5class", lambda a, b: qwk_ordinal(a, b)[0]),
               ("accuracy", lambda a, b: float((a == b).mean())),
               ("balanced_accuracy", lambda a, b: float(np.mean([(b[a == c] == c).mean() for c in np.unique(a)])))):
    pt, lo, hi = cluster_bootstrap(y, p, groups, fn, n=2000)
    s = fold_df[fold_df.arm == "CURRENT structured s42"]
    col = {"macro_f1_all6": "macro_f1_all6", "qwk_ordinal_5class": "qwk_ordinal_5class",
           "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"}[nm]
    st.append(dict(metric=nm, unit="image (pooled OOF)", point=pt, ci95_lo=lo, ci95_hi=hi,
                   ci_method="patient-clustered percentile bootstrap, 2000 reps, seed 42",
                   fold_mean=s[col].mean(), fold_sd=s[col].std(ddof=1),
                   fold_sem=s[col].std(ddof=1)/np.sqrt(5), n_folds=5, n_images=int(len(y)),
                   n_patients=int(len(np.unique(groups)))))
save_csv(pd.DataFrame(st), PKG / "07_TEST_RESULTS/statistical_uncertainty/uncertainty_image_level.csv")
save_csv(pd.DataFrame(st), PKG / "16_STATISTICAL_ANALYSIS/confidence_intervals.csv")

# ── paired tests, current vs flat baseline ───────────────────────────────────
from scipy import stats
pt_rows = []
for seed, a, b in ((42, "CURRENT structured s42", "flat softmax s42 (baseline)"),
                   (1337, "structured s1337 (seed replicate)", "flat softmax s1337 (baseline)")):
    A = fold_df[fold_df.arm == a].sort_values("fold"); B = fold_df[fold_df.arm == b].sort_values("fold")
    for met in ("macro_f1_all6", "qwk_ordinal_5class", "accuracy"):
        d = A[met].values - B[met].values
        t = stats.ttest_rel(A[met].values, B[met].values)
        w = stats.wilcoxon(A[met].values, B[met].values)
        pt_rows.append(dict(seed=seed, metric=met, structured_mean=A[met].mean(), flat_mean=B[met].mean(),
                            mean_paired_diff=d.mean(), sd_paired_diff=d.std(ddof=1),
                            paired_t_p=t.pvalue, wilcoxon_p=w.pvalue, n_folds=5,
                            note="5 folds is a small n for a paired test; report alongside the seed flip"))
save_csv(pd.DataFrame(pt_rows), PKG / "16_STATISTICAL_ANALYSIS/paired_tests.csv")

# ── CV summary figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for arm, c, mk in zip(ARMS, [C_OK, "#4f9ecf", C_BAD, "#e08a6a"], ["o", "o", "s", "s"]):
    s = fold_df[fold_df.arm == arm].sort_values("fold")
    axes[0].plot(s.fold, s.macro_f1_all6, mk + "-", color=c, label=arm, ms=5, lw=1.2)
    axes[1].plot(s.fold, s.qwk_ordinal_5class, mk + "-", color=c, ms=5, lw=1.2)
axes[0].set_xlabel("cross-validation fold"); axes[0].set_ylabel("macro-F1 (6 classes)")
axes[0].set_title("Macro-F1 per fold\nfold 0 is the atypical fold in every arm"); axes[0].legend(fontsize=6.5)
axes[0].set_xticks(range(5))
axes[1].set_xlabel("cross-validation fold"); axes[1].set_ylabel("QWK (corrected, 5 ordinal classes)")
axes[1].set_title("Corrected QWK per fold\nfold 0 is unremarkable once AP-ROP is off the scale")
axes[1].set_xticks(range(5)); axes[1].set_ylim(0.8, 0.96)
lbl = [a.split(" (")[0] for a in ARMS]
mn = [fold_df[fold_df.arm == a].macro_f1_all6.mean() for a in ARMS]
sd = [fold_df[fold_df.arm == a].macro_f1_all6.std(ddof=1) for a in ARMS]
axes[2].bar(lbl, mn, yerr=sd, capsize=4, color=[C_OK, "#4f9ecf", C_BAD, "#e08a6a"])
for i, (m, s_) in enumerate(zip(mn, sd)): axes[2].annotate(f"{m:.3f}\n±{s_:.3f}", (i, m), ha="center", va="bottom", fontsize=7)
axes[2].set_ylabel("macro-F1, mean ± SD over 5 folds"); axes[2].tick_params(axis="x", rotation=20, labelsize=6.5)
axes[2].set_title("Head × seed: the difference reverses sign between seeds\n(SD is fold-to-fold spread, NOT a confidence interval)")
axes[2].set_ylim(0, 0.95)
finish(fig, PKG / "06_VALIDATION/cross_validation_summary.png", "image", "5-fold OOF, all four arms")
print(summ_df.to_string(index=False))
print("\nPAIRED TESTS\n", pd.DataFrame(pt_rows).to_string(index=False))
