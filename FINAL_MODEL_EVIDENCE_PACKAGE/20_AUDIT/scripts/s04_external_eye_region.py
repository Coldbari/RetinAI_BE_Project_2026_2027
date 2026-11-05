"""13_EXTERNAL_GENERALIZATION + 08_EYE_LEVEL_ANALYSIS + 11_AGGREGATION_AND_VIEW_ANALYSIS
+ 07 ROC/PR. CURRENT MODEL on the held-out hospital."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
sys.path.insert(0, str(ROOT))
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

VIEWS = ["Left temporal", "Left posterior pole", "Left nasal",
         "Right nasal", "Right posterior pole", "Right temporal"]

def eyeframe(csv):
    df = pd.read_csv(PREP / csv)
    av = pd.read_excel(ROOT / "data/rop/multiview/ROP-DATASET/Dataset Acquisition View Table.xlsx")
    av["view"] = av[VIEWS].notna().idxmax(axis=1)
    av["eye"] = av["view"].str.split().str[0]
    av["region"] = av["view"].str.split(n=1).str[1]
    df["Image_name"] = df.image_path.str.rsplit("/", n=1).str[1].str.replace(".jpg", "", regex=False)
    df = df.merge(av[["Image_name", "eye", "region"]], on="Image_name", how="left")
    assert df.eye.notna().all()
    df["eye_id"] = df.patient_id.astype(str) + "|" + df.eye
    return df

locked, dev = eyeframe("test_locked.csv"), eyeframe("test_dev.csv")
CK = {s: [load_npz(ROOT / f"results/rop/locked_eval/{'structured' if s=='structured' else 'flat'}_s42_ckpt{k}.npz")
          for k in range(5)] for s in ("structured", "flat")}
SERVED = 0   # identified in s01: the served weights ARE the fold-0 checkpoint

# ══ 1. IMAGE-LEVEL EXTERNAL, current model ═══════════════════════════════════
rows = []
for k in range(5):
    y, p, pr = CK["structured"][k]
    rows.append(dict(checkpoint=f"fold{k}", is_served_checkpoint=(k == SERVED),
                     **{a: b for a, b in full_metrics(y, p, pr).items() if a != "per_class"}))
ext = pd.DataFrame(rows)
save_csv(ext, PKG / "13_EXTERNAL_GENERALIZATION/external_metrics_per_checkpoint.csv")

y0, p0, pr0 = CK["structured"][SERVED]
served_m = full_metrics(y0, p0, pr0)
ens_prob = np.mean([c[2].astype(float) for c in CK["structured"]], 0)
ens_pred = ens_prob.argmax(1)
ens_m = full_metrics(y0, ens_pred, ens_prob)
save_json(dict(served_checkpoint_fold0=served_m, five_checkpoint_ensemble=ens_m,
               per_checkpoint_mean={c: float(ext[c].mean()) for c in
                                    ("accuracy", "macro_f1_all6", "qwk_ordinal_5class", "qwk_6class_legacy")},
               per_checkpoint_sd_NOT_a_confidence_interval={c: float(ext[c].std(ddof=1)) for c in
                                    ("accuracy", "macro_f1_all6", "qwk_ordinal_5class", "qwk_6class_legacy")}),
          PKG / "13_EXTERNAL_GENERALIZATION/external_image_level.json")

# internal vs external, per class
oof = np.load(PKG / "07_TEST_RESULTS/current_model_oof_predictions.npz", allow_pickle=True)
yi, pi, pri, gi = oof["y_true"], oof["y_pred"], oof["prob6"], oof["patient_id"]
im = full_metrics(yi, pi, pri)
cmp_rows = []
for i, c in enumerate(CLASSES):
    ir = im["per_class"][c]; er = served_m["per_class"][c]
    cmp_rows.append(dict(cls=c, nice=NICE[i], internal_recall=ir["recall"], internal_support=ir["support"],
                         external_recall=er["recall"], external_support=er["support"],
                         recall_drop=ir["recall"] - er["recall"],
                         internal_precision=ir["precision"], external_precision=er["precision"]))
cmpdf = pd.DataFrame(cmp_rows)
save_csv(cmpdf, PKG / "13_EXTERNAL_GENERALIZATION/external_metrics.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(6); w = 0.38
axes[0].bar(x - w/2, cmpdf.internal_recall, w, label="internal (5-fold OOF, 3 training hospitals)", color=C_OK)
axes[0].bar(x + w/2, cmpdf.external_recall, w, label="external (locked held-out hospital)", color=C_BAD)
for i in x:
    axes[0].annotate(f"n={int(cmpdf.internal_support[i])}", (i-w/2, cmpdf.internal_recall[i]+.02), ha="center", fontsize=6.5)
    axes[0].annotate(f"n={int(cmpdf.external_support[i])}", (i+w/2, cmpdf.external_recall[i]+.02), ha="center", fontsize=6.5,
                     color=C_BAD if cmpdf.external_support[i] < 30 else "black")
axes[0].set_xticks(x); axes[0].set_xticklabels(NICE, rotation=18); axes[0].set_ylim(0, 1.12)
axes[0].set_ylabel("recall (sensitivity) for that class")
axes[0].set_title("Per-class recall collapses at a new hospital — except Normal\n"
                  "red n < 30 = NOT externally validated (pre-registered)")
axes[0].legend(fontsize=7, loc="upper center")
mets = ["accuracy", "balanced_accuracy", "macro_f1_all6", "qwk_ordinal_5class"]
lbl = ["accuracy", "balanced\naccuracy", "macro-F1\n(6 cls)", "QWK\n(corrected)"]
axes[1].bar(np.arange(4) - w/2, [im[m] for m in mets], w, label="internal (5-fold OOF, n=6,700)", color=C_OK)
axes[1].bar(np.arange(4) + w/2, [served_m[m] for m in mets], w, label="external locked (n=661)", color=C_BAD)
for i, m in enumerate(mets):
    axes[1].annotate(f"{im[m]:.3f}", (i-w/2, im[m]), ha="center", va="bottom", fontsize=7)
    axes[1].annotate(f"{served_m[m]:.3f}", (i+w/2, served_m[m]), ha="center", va="bottom", fontsize=7)
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(lbl); axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("score (0–1)"); axes[1].legend(fontsize=7)
axes[1].set_title("Internal vs external, image level, served checkpoint (fold 0)")
finish(fig, PKG / "13_EXTERNAL_GENERALIZATION/internal_vs_external.png", "image",
       "internal 5-fold OOF vs external locked")

plot_cm_path = PKG / "13_EXTERNAL_GENERALIZATION/external_confusion_matrix.png"
cme = confusion_matrix(y0, p0, labels=range(6))
fig, ax = plt.subplots(figsize=(6.4, 5.4))
M = cme / np.maximum(cme.sum(1, keepdims=True), 1)
im_ = ax.imshow(M, cmap="Reds", vmin=0, vmax=1)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{M[i,j]*100:.0f}%\n({cme[i,j]})", ha="center", va="center", fontsize=7,
                color="white" if M[i, j] > 0.55 else "black")
ax.set_xticks(range(6)); ax.set_xticklabels(NICE, rotation=25, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels([f"{n}\n(n={s})" for n, s in zip(NICE, cme.sum(1))], fontsize=7.5)
ax.set_xlabel("PREDICTED"); ax.set_ylabel("TRUE (support)"); ax.grid(False)
ax.set_title("External hospital, image level, row-normalised\n"
             "Normal 99% right; every disease stage ~25% right — the model defaults to Normal")
fig.colorbar(im_, ax=ax, shrink=0.8)
finish(fig, plot_cm_path, "image", "external locked (n=661), served fold-0 checkpoint")

# ── site comparison (internal per-source) ────────────────────────────────────
valsrc = pd.concat([pd.read_csv(PREP / f"fold_{f}.csv").query("split=='val'") for f in range(5)]).reset_index(drop=True)
srows = []
for s in sorted(valsrc.source.unique()):
    m = (valsrc.source == s).values
    mm = full_metrics(yi[m], pi[m], pri[m])
    srows.append(dict(site=s, kind="internal (training hospital)", n=int(m.sum()),
                      accuracy=mm["accuracy"], macro_f1_present=mm["macro_f1_present"],
                      qwk_ordinal=mm["qwk_ordinal_5class"], normal_recall=mm["per_class"]["normal"]["recall"]))
srows.append(dict(site="multiview (HELD-OUT hospital)", kind="external", n=len(y0),
                  accuracy=served_m["accuracy"], macro_f1_present=served_m["macro_f1_present"],
                  qwk_ordinal=served_m["qwk_ordinal_5class"], normal_recall=served_m["per_class"]["normal"]["recall"]))
sdf = pd.DataFrame(srows); save_csv(sdf, PKG / "13_EXTERNAL_GENERALIZATION/site_comparison.csv")
fig, ax = plt.subplots(figsize=(7.6, 3.8))
x = np.arange(len(sdf)); w = 0.27
for off, col, lab in ((-w, C_OK, "accuracy"), (0, C_ALT, "macro_f1_present"), (w, C_MUT, "qwk_ordinal")):
    ax.bar(x + off, sdf[lab], w, label=lab, color=col)
for i in x: ax.annotate(f"n={int(sdf.n[i])}", (i, 1.02), ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f"{r.site}\n({r.kind.split(' ')[0]})" for r in sdf.itertuples()], fontsize=7)
ax.set_ylim(0, 1.12); ax.legend(fontsize=7); ax.set_ylabel("score (0–1)")
ax.set_title("Per-site performance — three training hospitals vs the held-out hospital")
finish(fig, PKG / "13_EXTERNAL_GENERALIZATION/site_comparison.png", "image", "internal per-source + external locked")

# ══ 2. ANY-ROP SCREENING ROC / PR ════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for nm, yv, s, c in (("internal 5-fold OOF (n=6,700)", (yi >= 1).astype(int), pri[:, 1:].sum(1), C_OK),
                     ("external locked (n=661)", (y0 >= 1).astype(int), pr0[:, 1:].sum(1).astype(float), C_BAD)):
    fpr, tpr, _ = roc_curve(yv, s); a = roc_auc_score(yv, s)
    axes[0].plot(fpr, tpr, color=c, label=f"{nm} — AUC {a:.3f}")
    pcs, rcs, _ = precision_recall_curve(yv, s); ap = average_precision_score(yv, s)
    axes[1].plot(rcs, pcs, color=c, label=f"{nm} — AP {ap:.3f} (prevalence {yv.mean():.2f})")
axes[0].plot([0, 1], [0, 1], "--", color=C_MUT, lw=0.8, label="chance")
axes[0].set_xlabel("false-positive rate (1 − specificity)"); axes[0].set_ylabel("true-positive rate (sensitivity)")
axes[0].set_title("ROC — binary 'any ROP' = P(stage 1..4/5 or AP-ROP)"); axes[0].legend(fontsize=7, loc="lower right")
axes[1].set_xlabel("recall (sensitivity)"); axes[1].set_ylabel("precision (PPV)")
axes[1].set_title("Precision–Recall — same score"); axes[1].legend(fontsize=7, loc="lower left")
finish(fig, PKG / "07_TEST_RESULTS/roc_pr_analysis/any_rop_roc_pr.png", "image",
       "internal OOF vs external locked")

thr_rows = []
for nm, yv, s in (("internal 5-fold OOF", (yi >= 1).astype(int), pri[:, 1:].sum(1)),
                  ("external locked", (y0 >= 1).astype(int), pr0[:, 1:].sum(1).astype(float))):
    for t in (0.0155, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
        pp = (s > t).astype(int)
        tp = int(((pp == 1) & (yv == 1)).sum()); fp = int(((pp == 1) & (yv == 0)).sum())
        tn = int(((pp == 0) & (yv == 0)).sum()); fn = int(((pp == 0) & (yv == 1)).sum())
        thr_rows.append(dict(split=nm, threshold=t, TP=tp, FP=fp, TN=tn, FN=fn,
                             sensitivity=tp/max(tp+fn,1), specificity=tn/max(tn+fp,1),
                             PPV=tp/max(tp+fp,1), NPV=tn/max(tn+fn,1),
                             note="0.0155 is the SERVED threshold (screening_rebase.json)" if t == 0.0155 else ""))
save_csv(pd.DataFrame(thr_rows), PKG / "07_TEST_RESULTS/roc_pr_analysis/any_rop_threshold_table.csv")

# ══ 3. EYE-LEVEL + AGGREGATION RULES ═════════════════════════════════════════
def agg_eye(df, pred, prob, rule):
    d = df.assign(pred=pred)
    if rule == "max (protocol rule)":
        g = d.groupby("eye_id").agg(y=("label", "max"), p=("pred", "max"), pid=("patient_id", "first"))
    elif rule == "majority vote":
        g = d.groupby("eye_id").agg(y=("label", "max"),
                                    p=("pred", lambda s: s.value_counts().idxmax()), pid=("patient_id", "first"))
    elif rule == "mean probability":
        t = pd.DataFrame(prob.astype(float)); t["eye_id"] = df.eye_id.values
        t["y"] = df.label.values; t["pid"] = df.patient_id.values
        mp = t.groupby("eye_id")[list(range(6))].mean()
        g = pd.DataFrame(dict(y=t.groupby("eye_id").y.max(), p=mp.values.argmax(1),
                              pid=t.groupby("eye_id").pid.first()))
    elif rule == "max probability per class":
        t = pd.DataFrame(prob.astype(float)); t["eye_id"] = df.eye_id.values
        t["y"] = df.label.values; t["pid"] = df.patient_id.values
        mp = t.groupby("eye_id")[list(range(6))].max()
        g = pd.DataFrame(dict(y=t.groupby("eye_id").y.max(), p=mp.values.argmax(1),
                              pid=t.groupby("eye_id").pid.first()))
    elif rule == "one random view":
        s = d.groupby("eye_id", group_keys=False).sample(1, random_state=42)
        g = pd.DataFrame(dict(y=s.label.values, p=s.pred.values, pid=s.patient_id.values), index=s.eye_id)
    elif rule == "temporal view only":
        s = d[d.region == "temporal"].groupby("eye_id").agg(y=("label", "max"), p=("pred", "max"),
                                                            pid=("patient_id", "first"))
        g = s
    return g

RULES = ["max (protocol rule)", "max probability per class", "majority vote", "mean probability",
         "one random view", "temporal view only"]
agg_rows = []
for head in ("structured", "flat"):
    for rule in RULES:
        vals = []
        for k in range(5):
            y, p, pr = CK[head][k]
            g = agg_eye(locked, p, pr, rule)
            vals.append(dict(qwk=qwk_ordinal(g.y.values, g.p.values)[0],
                             acc=float((g.y == g.p).mean()),
                             miss=float((g[g.y.isin([1, 2, 3])].p == 0).mean()), n=len(g)))
        agg_rows.append(dict(model_head=("CURRENT structured" if head == "structured" else "flat baseline"),
                             rule=rule, n_units=vals[0]["n"],
                             qwk_mean=np.mean([v["qwk"] for v in vals]),
                             qwk_sd_over_checkpoints=np.std([v["qwk"] for v in vals], ddof=1),
                             accuracy_mean=np.mean([v["acc"] for v in vals]),
                             stage123_called_normal_mean=np.mean([v["miss"] for v in vals])))
# image level reference
for head in ("structured", "flat"):
    v = [qwk_ordinal(*CK[head][k][:2])[0] for k in range(5)]
    a = [float((CK[head][k][0] == CK[head][k][1]).mean()) for k in range(5)]
    agg_rows.append(dict(model_head=("CURRENT structured" if head == "structured" else "flat baseline"),
                         rule="NO aggregation (image level)", n_units=661,
                         qwk_mean=np.mean(v), qwk_sd_over_checkpoints=np.std(v, ddof=1),
                         accuracy_mean=np.mean(a),
                         stage123_called_normal_mean=np.mean([
                             float((CK[head][k][1][(CK[head][k][0] >= 1) & (CK[head][k][0] <= 3)] == 0).mean())
                             for k in range(5)])))
aggdf = pd.DataFrame(agg_rows)
save_csv(aggdf, PKG / "11_AGGREGATION_AND_VIEW_ANALYSIS/aggregation_results.csv")
save_csv(aggdf[aggdf.model_head == "CURRENT structured"], PKG / "08_EYE_LEVEL_ANALYSIS/eye_qwk.csv")

order = ["NO aggregation (image level)", "one random view", "mean probability",
         "max probability per class", "majority vote", "max (protocol rule)", "temporal view only"]
fig, ax = plt.subplots(figsize=(9.2, 4.4))
cur = aggdf[aggdf.model_head == "CURRENT structured"].set_index("rule").loc[order]
fl = aggdf[aggdf.model_head == "flat baseline"].set_index("rule").loc[order]
x = np.arange(len(order)); w = 0.38
ax.bar(x - w/2, cur.qwk_mean, w, yerr=cur.qwk_sd_over_checkpoints, capsize=3,
       label="CURRENT structured model", color=C_OK)
ax.bar(x + w/2, fl.qwk_mean, w, yerr=fl.qwk_sd_over_checkpoints, capsize=3,
       label="flat softmax baseline", color=C_BAD, alpha=0.75)
for i in x:
    ax.annotate(f"{cur.qwk_mean.iloc[i]:.3f}", (i-w/2, cur.qwk_mean.iloc[i]+.03), ha="center", fontsize=7)
    ax.annotate(f"n={int(cur.n_units.iloc[i])}", (i-w/2, 0.02), ha="center", fontsize=6, color="white")
ax.axhspan(0.45, 0.60, color=C_MUT, alpha=0.12)
ax.annotate("every rule except MAX lands here\n(= the image-level value)", (1.5, 0.63), fontsize=7.5, color="#444")
ax.set_xticks(x); ax.set_xticklabels([o.replace(" (", "\n(") for o in order], fontsize=7)
ax.set_ylabel("QWK (corrected, 5 ordinal classes)"); ax.set_ylim(0, 1.0); ax.legend(fontsize=8)
ax.set_title("The eye-level headline depends on the AGGREGATION OPERATOR, not on using the eye as a unit\n"
             "External locked set · error bars = SD over 5 fold-checkpoints (NOT a confidence interval)")
finish(fig, PKG / "11_AGGREGATION_AND_VIEW_ANALYSIS/aggregation_comparison.png", "eye / image",
       "external locked (n=661 images, 230 eyes)")

# ── eye-level detail for the CURRENT model, protocol (max) rule ──────────────
g = agg_eye(locked, p0, pr0, "max (protocol rule)")
em = full_metrics(g.y.values, g.p.values)
save_json(em, PKG / "08_EYE_LEVEL_ANALYSIS/eye_level_metrics_served_checkpoint.json")
epc = pd.DataFrame([dict(cls=CLASSES[i], nice=NICE[i], **em["per_class"][CLASSES[i]]) for i in range(6)])
save_csv(epc, PKG / "08_EYE_LEVEL_ANALYSIS/eye_metrics.csv")
cmeye = confusion_matrix(g.y.values, g.p.values, labels=range(6))
for norm, fn_ in ((None, "eye_confusion_matrix.png"), ("true", "eye_confusion_matrix_normalized.png")):
    M = cmeye.astype(float)
    if norm == "true": M = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    imh = ax.imshow(M, cmap="Blues"); ax.grid(False)
    for i in range(6):
        for j in range(6):
            t = f"{cmeye[i,j]}" if norm is None else f"{M[i,j]*100:.0f}%\n({cmeye[i,j]})"
            ax.text(j, i, t, ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] > M.max()*0.55 else "black")
    ax.set_xticks(range(6)); ax.set_xticklabels(NICE, rotation=25, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels([f"{n}\n(n={s})" for n, s in zip(NICE, cmeye.sum(1))], fontsize=7.5)
    ax.set_xlabel("PREDICTED eye stage"); ax.set_ylabel("TRUE eye stage (support)")
    ax.set_title(f"EYE-level confusion, max-over-views rule{' (row-normalised)' if norm else ' (counts)'}\n"
                 f"external locked hospital · {len(g)} eyes from {locked.patient_id.nunique()} infants")
    fig.colorbar(imh, ax=ax, shrink=0.8)
    finish(fig, PKG / "08_EYE_LEVEL_ANALYSIS" / fn_, "eye", "external locked (230 eyes)")

fig, ax = plt.subplots(figsize=(7.6, 3.8))
x = np.arange(6); w = 0.27
for off, col, lab in ((-w, C_OK, "precision"), (0, C_ALT, "recall"), (w, C_MUT, "f1")):
    ax.bar(x + off, epc[lab], w, label=lab, color=col)
for i in x: ax.annotate(f"n={int(epc.support[i])}", (i, 1.03), ha="center", fontsize=7,
                        color=C_BAD if epc.support[i] < 10 else "black")
ax.set_xticks(x); ax.set_xticklabels(NICE, rotation=18); ax.set_ylim(0, 1.15); ax.legend(fontsize=7)
ax.set_ylabel("score (0–1)")
ax.set_title("Eye-level per-class performance (max rule), external locked hospital\n"
             "red support < 10 eyes — not interpretable")
finish(fig, PKG / "08_EYE_LEVEL_ANALYSIS/eye_per_class_metrics.png", "eye", "external locked (230 eyes)")

eci = []
for nm, fn_ in (("qwk_ordinal_5class", lambda a, b: qwk_ordinal(a, b)[0]),
                ("accuracy", lambda a, b: float((a == b).mean())),
                ("macro_f1_present", lambda a, b: f1_score(a, b, labels=sorted(set(a.tolist())), average="macro", zero_division=0)),
                ("stage123_eyes_called_normal", lambda a, b: float((b[(a >= 1) & (a <= 3)] == 0).mean()))):
    pt, lo, hi = cluster_bootstrap(g.y.values, g.p.values, g.pid.values, fn_, n=2000)
    eci.append(dict(metric=nm, unit="eye", point=pt, ci95_lo=lo, ci95_hi=hi,
                    ci_method="PATIENT-clustered bootstrap (eyes of one infant move together), 2000 reps",
                    n_eyes=len(g), n_patients=int(g.pid.nunique())))
save_csv(pd.DataFrame(eci), PKG / "08_EYE_LEVEL_ANALYSIS/eye_confidence_intervals.csv")

# ══ 4. REGION / VIEW ANALYSIS — the decisive control ═════════════════════════
reg_rows = []
for head in ("structured", "flat"):
    for reg in ["temporal", "nasal", "posterior pole", "ALL REGIONS"]:
        m = np.ones(len(locked), bool) if reg == "ALL REGIONS" else (locked.region == reg).values
        v = []
        for k in range(5):
            y, p, pr = CK[head][k]
            mm = full_metrics(y[m], p[m], pr[m])
            v.append((mm["qwk_ordinal_5class"], mm["accuracy"], mm["macro_f1_present"],
                      float((p[m][(y[m] >= 1) & (y[m] <= 3)] == 0).mean())))
        v = np.array(v)
        reg_rows.append(dict(model_head=("CURRENT structured" if head == "structured" else "flat baseline"),
                             region=reg, n_images=int(m.sum()),
                             n_stage123=int(((locked.label >= 1) & (locked.label <= 3) & m).sum()),
                             qwk_mean=v[:, 0].mean(), qwk_sd_over_checkpoints=v[:, 0].std(ddof=1),
                             accuracy_mean=v[:, 1].mean(), macro_f1_present_mean=v[:, 2].mean(),
                             stage123_called_normal_mean=v[:, 3].mean()))
regdf = pd.DataFrame(reg_rows)
save_csv(regdf, PKG / "11_AGGREGATION_AND_VIEW_ANALYSIS/region_results.csv")

cur_r = regdf[regdf.model_head == "CURRENT structured"].set_index("region")
eye_max = float(aggdf.query("model_head=='CURRENT structured' and rule=='max (protocol rule)'").qwk_mean.iloc[0])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
ordr = ["ALL REGIONS", "posterior pole", "nasal", "temporal"]
v = cur_r.loc[ordr]
b = axes[0].bar(range(4), v.qwk_mean, yerr=v.qwk_sd_over_checkpoints, capsize=4,
                color=[C_MUT, C_BAD, C_BAD, C_ALT])
axes[0].axhline(eye_max, ls="--", color=C_OK, lw=1.3)
axes[0].annotate(f"eye-level, max-over-views rule = {eye_max:.3f}", (0.05, eye_max + 0.015),
                 fontsize=7.5, color=C_OK)
for i in range(4):
    axes[0].annotate(f"{v.qwk_mean.iloc[i]:.3f}\nn={int(v.n_images.iloc[i])}",
                     (i, v.qwk_mean.iloc[i] + 0.03), ha="center", fontsize=7.5)
axes[0].set_xticks(range(4)); axes[0].set_xticklabels([o.replace(" ", "\n") for o in ordr])
axes[0].set_ylabel("QWK (corrected, 5 ordinal classes)"); axes[0].set_ylim(0, 1.0)
axes[0].set_title("THE CONTROL: temporal frames alone, scored at IMAGE level with NO aggregation,\n"
                  "reach the eye-level number. The gap is anatomical, not an averaging artifact.")
v2 = cur_r.loc[["posterior pole", "nasal", "temporal"]]
axes[1].bar(range(3), v2.stage123_called_normal_mean * 100, color=[C_BAD, C_BAD, C_ALT])
for i in range(3):
    axes[1].annotate(f"{v2.stage123_called_normal_mean.iloc[i]*100:.1f}%\nof n={int(v2.n_stage123.iloc[i])}",
                     (i, v2.stage123_called_normal_mean.iloc[i]*100 + 2), ha="center", fontsize=7.5)
axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["posterior\npole", "nasal", "temporal"])
axes[1].set_ylabel("% of stage 1–3 frames predicted Normal"); axes[1].set_ylim(0, 100)
axes[1].set_title("Mechanism: the ROP ridge lives in the temporal periphery.\n"
                  "Frames that cannot show it are called Normal — and their label came from the eye.")
finish(fig, PKG / "11_AGGREGATION_AND_VIEW_ANALYSIS/region_qwk_comparison.png", "image",
       "external locked, by acquisition region")

for reg, fn_ in (("temporal", "temporal_analysis.png"), ("nasal", "nasal_analysis.png"),
                 ("posterior pole", "posterior_analysis.png")):
    m = (locked.region == reg).values
    cmr = confusion_matrix(y0[m], p0[m], labels=range(6))
    fig, ax = plt.subplots(figsize=(5.8, 4.9))
    M = cmr / np.maximum(cmr.sum(1, keepdims=True), 1)
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(6):
        for j in range(6):
            if cmr[i, j]: ax.text(j, i, f"{cmr[i,j]}", ha="center", va="center", fontsize=8,
                                  color="white" if M[i, j] > 0.55 else "black")
    ax.set_xticks(range(6)); ax.set_xticklabels(NICE, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(6)); ax.set_yticklabels([f"{n} ({s})" for n, s in zip(NICE, cmr.sum(1))], fontsize=7)
    r = cur_r.loc[reg]
    ax.set_title(f"{reg.upper()} frames only — external locked hospital\n"
                 f"n={int(r.n_images)} images · QWK {r.qwk_mean:.3f} · accuracy {r.accuracy_mean:.3f}")
    ax.set_xlabel("PREDICTED"); ax.set_ylabel("TRUE (support)")
    finish(fig, PKG / "11_AGGREGATION_AND_VIEW_ANALYSIS" / fn_, "image", f"external locked, {reg} only")

print(aggdf.to_string(index=False)); print(); print(regdf.to_string(index=False))
print("\nEYE CI\n", pd.DataFrame(eci).to_string(index=False))
