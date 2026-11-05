"""05_TRAINING — from the CURRENT MODEL's own Kaggle kernel log (kaggle_cache/struct42)."""
import sys, re, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
sys.path.insert(0, str(ROOT))

txt = CUR["log"].read_text(errors="ignore")
pat = (r"Epoch\s+(\d+)/(\d+)\s+loss ([\d.]+) \(stage ([\d.]+) arop ([\d.]+) site ([\d.]+) "
       r"lam ([\d.]+)\)\s+val macroF1 ([\d.]+)%\s+QWK ([-\d.]+)")
rec = re.findall(pat, txt)
heads = re.findall(r"=== structured \w+ \| (\w+) \| train (\d+) val (\d+)", txt)
runs, cur = [], []
for e, tot, l, ls, la, lsi, lam, f1, q in rec:
    if int(e) == 1 and cur: runs.append(cur); cur = []
    cur.append(dict(epoch=int(e), total_epochs=int(tot), train_loss_total=float(l),
                    train_loss_stage=float(ls), train_loss_arop=float(la),
                    train_loss_site=float(lsi), grl_lambda=float(lam),
                    val_macro_f1=float(f1)/100, val_qwk_6class_legacy=float(q)))
if cur: runs.append(cur)
names = [h[0] for h in heads] or [f"run{i}" for i in range(len(runs))]
rows = []
for nm, r in zip(names, runs):
    for d in r: rows.append(dict(run=nm, **d))
hist = pd.DataFrame(rows)
save_csv(hist, PKG / "05_TRAINING/training_history.csv")

# OneCycleLR reconstruction (the trainer hardcodes it; no LR is logged)
import math
lr_rows = []
for nm, r in zip(names, runs):
    n_ep = r[0]["total_epochs"]; steps = 154   # 2474 train imgs / bs16, drop_last -> 154 steps
    tot = n_ep * steps; maxlr = 3e-4; pct = 0.1
    for s in range(tot):
        pr_ = s / tot
        lr = (maxlr*(0.04 + 0.96*(s/(pct*tot))) if s < pct*tot
              else maxlr*(0.5*(1+math.cos(math.pi*((s-pct*tot)/((1-pct)*tot))))*(1-1e-4)+1e-4))
        if s % 10 == 0: lr_rows.append(dict(run=nm, step=s, epoch=s/steps, lr=lr))
lrdf = pd.DataFrame(lr_rows); save_csv(lrdf, PKG / "05_TRAINING/learning_rate_curves/onecycle_lr_schedule.csv")

fig, ax = plt.subplots(figsize=(7.4, 3.4))
s = lrdf[lrdf.run == names[0]]
ax.plot(s.epoch, s.lr, color=C_OK)
ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
ax.set_title("OneCycleLR schedule (RECONSTRUCTED from the hardcoded trainer settings —\n"
             "max_lr 3e-4, pct_start 0.1, cosine anneal. The LR was NOT logged during training.)")
finish(fig, PKG / "05_TRAINING/learning_rate_curves/learning_rate_schedule.png", "optimizer step", "all folds")

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, nm in zip(axes.ravel(), names):
    h = hist[hist.run == nm]
    a2 = ax.twinx()
    ax.plot(h.epoch, h.train_loss_total, color=C_OK, lw=1.6, label="train loss (total)")
    ax.plot(h.epoch, h.train_loss_stage, color=C_OK, ls="--", lw=1, alpha=.7, label="train loss: CORN stage")
    ax.plot(h.epoch, h.train_loss_arop, color=C_ALT, ls="--", lw=1, alpha=.8, label="train loss: AP-ROP branch")
    ax.plot(h.epoch, h.train_loss_site, color=C_MUT, ls=":", lw=1, label="train loss: site adversary")
    a2.plot(h.epoch, h.val_macro_f1, "o-", color=C_BAD, ms=3, lw=1.3, label="VAL macro-F1")
    a2.set_ylim(0, 1); a2.set_ylabel("val macro-F1", color=C_BAD); a2.grid(False)
    last5 = h.val_macro_f1.values[-5:]
    ax.set_title(f"{nm}  ·  final val macro-F1 {h.val_macro_f1.iloc[-1]:.3f}\n"
                 f"best epoch {int(h.val_macro_f1.idxmax()-h.index[0])+1} = {h.val_macro_f1.max():.3f}"
                 f"  ·  last-5-epoch SD {last5.std(ddof=1):.3f}", fontsize=9)
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    if nm == names[0]: ax.legend(fontsize=6, loc="upper right")
fig.suptitle("CURRENT MODEL training behaviour — all 6 runs (5 CV folds + leave-site fold), seed 42, 20 epochs\n"
             "Training loss falls smoothly; validation macro-F1 is highly unstable. "
             "NO VALIDATION LOSS WAS RECORDED — see 05_TRAINING/training_analysis.md", y=1.0)
finish(fig, PKG / "05_TRAINING/training_curves/all_runs_loss_and_val_metric.png", "image", "train + val per fold")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
for nm in names:
    h = hist[hist.run == nm]
    axes[0].plot(h.epoch, h.train_loss_total, lw=1.2, label=nm)
    axes[1].plot(h.epoch, h.val_macro_f1, "o-", ms=3, lw=1.2, label=nm)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("total training loss"); axes[0].legend(fontsize=6.5)
axes[0].set_title("Training loss — smooth, monotone, no divergence\n(sum of CORN stage + AP-ROP + λ·site losses)")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation macro-F1"); axes[1].set_ylim(0, 1)
axes[1].set_title("Validation macro-F1 — swings up to 0.3 between adjacent epochs")
st = []
for nm in names:
    h = hist[hist.run == nm]; v = h.val_macro_f1.values
    st.append(dict(run=nm, last5_sd=v[-5:].std(ddof=1), mean_abs_epoch_delta=np.abs(np.diff(v)).mean(),
                   max_abs_epoch_delta=np.abs(np.diff(v)).max(), final=v[-1], best=v.max(),
                   best_minus_final=v.max()-v[-1], best_epoch=int(v.argmax())+1))
stdf = pd.DataFrame(st); save_csv(stdf, PKG / "05_TRAINING/last_epoch_noise.csv")
x = np.arange(len(names)); w = .38
axes[2].bar(x - w/2, stdf.last5_sd, w, label="SD of val macro-F1 over the LAST 5 epochs", color=C_BAD)
axes[2].bar(x + w/2, stdf.best_minus_final, w, label="best epoch − reported (last) epoch", color=C_MUT)
axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=25, fontsize=7)
axes[2].set_ylabel("macro-F1 units"); axes[2].legend(fontsize=6.5)
axes[2].set_title("The cost of the last-epoch rule\nend-of-schedule noise is never propagated into any reported number")
finish(fig, PKG / "05_TRAINING/training_curves/training_stability.png", "image", "train + val per fold")

fig, ax = plt.subplots(figsize=(7.4, 3.4))
h = hist[hist.run == names[0]]
ax.plot(h.epoch, h.grl_lambda, "o-", color=C_OK, ms=4)
ax.set_xlabel("epoch"); ax.set_ylabel("gradient-reversal λ")
ax.set_title("DANN gradient-reversal ramp actually applied (read from the training log)\n"
             "λ reaches 1.0; the site loss is then weighted by a CONSTANT w_site = 0.5")
finish(fig, PKG / "05_TRAINING/training_curves/dann_lambda_ramp.png", "optimizer step", "fold 0")

save_json(dict(runs=names, epochs_per_run=int(hist.total_epochs.iloc[0]),
               validation_loss_recorded=False,
               learning_rate_recorded=False,
               lr_curve_is_reconstructed_not_measured=True,
               stability=st), PKG / "05_TRAINING/training_summary.json")
print(hist.groupby("run").val_macro_f1.agg(["first", "last", "max", "std"]).to_string())
print(); print(stdf.to_string(index=False))
