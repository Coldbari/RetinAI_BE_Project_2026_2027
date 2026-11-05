"""10_CONFIDENCE_CALIBRATION for the CURRENT MODEL, with the flat baseline as comparator."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
sys.path.insert(0, str(ROOT))

oof = np.load(PKG / "07_TEST_RESULTS/current_model_oof_predictions.npz", allow_pickle=True)
y, p, pr, grp = oof["y_true"], oof["y_pred"], oof["prob6"].astype(float), oof["patient_id"].astype(str)
fy, fp, fpr = [np.concatenate(a) for a in zip(*[load_npz(FLAT / f"fold_{f}.npz") for f in range(5)])]
fpr = fpr.astype(float)
lk = [load_npz(ROOT / f"results/rop/locked_eval/structured_s42_ckpt{k}.npz") for k in range(5)]
ly, lp, lpr = lk[0][0], lk[0][1], lk[0][2].astype(float)

def pack(y, p, pr, name, unit, split):
    conf = pr.max(1); ok = (y == p).astype(float)
    e, bins = ece_bins(conf, ok)
    return dict(name=name, unit=unit, split=split, n=len(y), ece_15bin=e,
                brier_multiclass=brier_multiclass(y, pr),
                mean_confidence=float(conf.mean()), accuracy=float(ok.mean()),
                overconfidence_gap=float(conf.mean() - ok.mean()),
                mean_conf_when_correct=float(conf[ok == 1].mean()),
                mean_conf_when_wrong=float(conf[ok == 0].mean()),
                separation=float(conf[ok == 1].mean() - conf[ok == 0].mean()),
                frac_conf_over_90=float((conf > .9).mean()),
                n_high_conf=int((conf > .9).sum()),
                n_high_conf_wrong=int(((conf > .9) & (ok == 0)).sum()),
                high_conf_accuracy=float(ok[conf > .9].mean()) if (conf > .9).any() else np.nan,
                argmax_disagrees_with_decode=float((pr.argmax(1) != p).mean())), bins

rows, binsets = [], {}
for args in ((y, p, pr, "CURRENT structured (s42)", "image", "internal 5-fold OOF"),
             (fy, fp, fpr, "flat softmax baseline (s42)", "image", "internal 5-fold OOF"),
             (ly, lp, lpr, "CURRENT structured (served fold-0 ckpt)", "image", "external locked")):
    r, b = pack(*args); rows.append(r); binsets[r["name"] + " | " + r["split"]] = b
for f in range(5):
    yy, pp, pp6 = load_npz(CUR["cv_dir"] / f"fold_{f}.npz")
    r, _ = pack(yy, pp, pp6.astype(float), f"CURRENT structured — fold {f}", "image", f"val fold {f}")
    rows.append(r)
cal = pd.DataFrame(rows)
save_csv(cal, PKG / "10_CONFIDENCE_CALIBRATION/calibration_metrics.csv")

# reliability diagram
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for nm, c, mk in (("CURRENT structured (s42) | internal 5-fold OOF", C_OK, "o"),
                  ("flat softmax baseline (s42) | internal 5-fold OOF", C_BAD, "s")):
    b = binsets[nm].dropna(subset=["conf"])
    axes[0].plot(b.conf, b.acc, mk + "-", color=c, ms=4, lw=1.3,
                 label=f"{nm.split(' |')[0]} — ECE {cal[cal.name==nm.split(' |')[0]].ece_15bin.iloc[0]:.4f}")
    for _, r in b.iterrows():
        if r.n > 40: axes[0].annotate(f"{int(r.n)}", (r.conf, r.acc), fontsize=5.5, color=c,
                                      xytext=(0, -9), textcoords="offset points", ha="center")
axes[0].plot([0, 1], [0, 1], "--", color=C_MUT, lw=1, label="perfect calibration")
axes[0].set_xlabel("predicted confidence (max class probability)")
axes[0].set_ylabel("observed accuracy in that bin")
axes[0].set_title("Reliability diagram, 15 bins — internal 5-fold out-of-fold (n=6,700)\n"
                  "points below the diagonal = over-confident · small numbers = images in the bin")
axes[0].legend(fontsize=7, loc="upper left"); axes[0].set_xlim(0, 1.02); axes[0].set_ylim(0, 1.02)
for nm, c in (("CURRENT structured (s42) | internal 5-fold OOF", C_OK),
              ("flat softmax baseline (s42) | internal 5-fold OOF", C_BAD)):
    b = binsets[nm]
    axes[1].bar(b.lo + 1/30, b.n, width=1/15*0.9, alpha=0.55, color=c, label=nm.split(" |")[0])
axes[1].set_xlabel("predicted confidence"); axes[1].set_ylabel("number of images")
axes[1].set_title("Where the confidence mass sits\n"
                  "the current model puts 69% of predictions above 0.90; the baseline puts 13%")
axes[1].legend(fontsize=7)
finish(fig, PKG / "10_CONFIDENCE_CALIBRATION/reliability_diagram.png", "image", "internal 5-fold OOF")

fig, axes = plt.subplots(1, 2, figsize=(11, 4)); bins = np.linspace(0, 1, 31)
for ax, (yy, pp, pp6, t) in zip(axes, ((y, p, pr, "CURRENT structured model"),
                                       (fy, fp, fpr, "flat softmax baseline (comparator)"))):
    conf = pp6.max(1); ok = yy == pp
    ax.hist(conf[ok], bins=bins, color=C_ALT, alpha=.8, label=f"correct (n={int(ok.sum())})")
    ax.hist(conf[~ok], bins=bins, color=C_BAD, alpha=.8, label=f"incorrect (n={int((~ok).sum())})")
    ax.axvline(0.9, ls="--", color="#333", lw=1)
    ax.annotate("0.90", (0.9, ax.get_ylim()[1]*0.9), fontsize=7)
    ax.set_yscale("log"); ax.set_xlabel("predicted confidence"); ax.set_ylabel("images (log scale)")
    ax.set_title(f"{t}\nmean conf correct {conf[ok].mean():.3f} vs wrong {conf[~ok].mean():.3f}"
                 f" (separation {conf[ok].mean()-conf[~ok].mean():.3f})")
    ax.legend(fontsize=7)
finish(fig, PKG / "10_CONFIDENCE_CALIBRATION/confidence_correct_vs_wrong.png", "image", "internal 5-fold OOF")

fig, ax = plt.subplots(figsize=(7.4, 3.8))
for nm, yy, pp, pp6, c in (("CURRENT structured — internal OOF", y, p, pr, C_OK),
                           ("CURRENT structured — external locked", ly, lp, lpr, C_BAD),
                           ("flat baseline — internal OOF", fy, fp, fpr, C_MUT)):
    ax.hist(pp6.max(1), bins=np.linspace(0, 1, 41), histtype="step", lw=1.6, color=c,
            density=True, label=f"{nm} (n={len(yy)})")
ax.set_xlabel("predicted confidence (max class probability)"); ax.set_ylabel("density")
ax.legend(fontsize=7); ax.set_title("Confidence distribution — the CORN cascade produces very peaked probabilities")
finish(fig, PKG / "10_CONFIDENCE_CALIBRATION/confidence_distribution.png", "image", "internal + external")

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
sub = cal[cal.name.str.contains("CURRENT structured — fold")]
axes[0].bar(range(5), sub.ece_15bin, color=[C_BAD if v > .15 else C_OK for v in sub.ece_15bin])
for i, v in enumerate(sub.ece_15bin): axes[0].annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=7)
axes[0].set_xticks(range(5)); axes[0].set_xticklabels([f"fold {i}" for i in range(5)])
axes[0].set_ylabel("Expected Calibration Error (15 bins)")
axes[0].set_title("ECE per fold — fold 0 is 2–3× the others")
m = cal[cal.name.isin(["CURRENT structured (s42)", "flat softmax baseline (s42)"])]
x = np.arange(2)
axes[1].bar(x - .2, m.ece_15bin, .4, label="ECE (lower is better)", color=C_OK)
axes[1].bar(x + .2, m.brier_multiclass, .4, label="Brier score (lower is better)", color=C_ALT)
for i in x:
    axes[1].annotate(f"{m.ece_15bin.iloc[i]:.4f}", (i-.2, m.ece_15bin.iloc[i]), ha="center", va="bottom", fontsize=7)
    axes[1].annotate(f"{m.brier_multiclass.iloc[i]:.4f}", (i+.2, m.brier_multiclass.iloc[i]), ha="center", va="bottom", fontsize=7)
axes[1].set_xticks(x); axes[1].set_xticklabels(["CURRENT\nstructured", "flat softmax\nbaseline"])
axes[1].legend(fontsize=7); axes[1].set_ylabel("error")
axes[1].set_title("At identical accuracy (0.786 vs 0.789)\nthe current model is ~3× worse calibrated")
axes[2].bar(x - .2, m.frac_conf_over_90 * 100, .4, color=C_MUT, label="% of predictions with conf > 0.90")
axes[2].bar(x + .2, m.n_high_conf_wrong, .4, color=C_BAD, label="count of confident WRONG predictions")
for i in x:
    axes[2].annotate(f"{m.frac_conf_over_90.iloc[i]*100:.1f}%", (i-.2, m.frac_conf_over_90.iloc[i]*100), ha="center", va="bottom", fontsize=7)
    axes[2].annotate(f"{int(m.n_high_conf_wrong.iloc[i])}", (i+.2, m.n_high_conf_wrong.iloc[i]), ha="center", va="bottom", fontsize=7)
axes[2].set_xticks(x); axes[2].set_xticklabels(["CURRENT\nstructured", "flat softmax\nbaseline"])
axes[2].legend(fontsize=7); axes[2].set_ylabel("percent  /  count")
axes[2].set_title("Confident errors: 359 vs 181 out of 6,700")
finish(fig, PKG / "10_CONFIDENCE_CALIBRATION/structured_vs_flat_calibration.png", "image", "internal 5-fold OOF")

for k, b in binsets.items():
    b.insert(0, "series", k)
save_csv(pd.concat(binsets.values()), PKG / "10_CONFIDENCE_CALIBRATION/reliability_bins.csv")
print(cal.to_string(index=False))
