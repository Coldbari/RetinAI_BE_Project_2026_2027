"""12_MODEL_COMPARISON + 04_PREPROCESSING + 15_DEPLOYMENT + 01_MODEL artifacts."""
import sys, time, json, yaml, shutil
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
sys.path.insert(0, str(ROOT))
import torch, hashlib
from PIL import Image
from models.common.structured import StructuredROPModel, decode_6class
from models.common.config import load_config
from models.common.preprocessing import (build_transforms, circle_crop, letterbox_square, clahe)

SALT = "retinai-evidence-package-v1"
anon = lambda s: "CASE-" + hashlib.sha256((SALT+str(s)).encode()).hexdigest()[:10].upper()
cfg = load_config(str(CUR["config"]))

# ══ 12_MODEL_COMPARISON ═════════════════════════════════════════════════════
fold = pd.read_csv(PKG / "06_VALIDATION/fold_results.csv")
cal  = pd.read_csv(PKG / "10_CONFIDENCE_CALIBRATION/calibration_metrics.csv")
aggd = pd.read_csv(PKG / "11_AGGREGATION_AND_VIEW_ANALYSIS/aggregation_results.csv")
CURN, FLATN = "CURRENT structured s42", "flat softmax s42 (baseline)"
cm_rows = []
def add(dim, unit, cur, flat, better, note):
    cm_rows.append(dict(dimension=dim, evaluation_unit=unit, current_structured=cur,
                        flat_softmax_baseline=flat, favours=better, note=note))
a, b = fold[fold.arm == CURN], fold[fold.arm == FLATN]
add("macro-F1 (6 classes), CV mean ± SD", "image (5-fold OOF)",
    f"{a.macro_f1_all6.mean():.4f} ± {a.macro_f1_all6.std(ddof=1):.4f}",
    f"{b.macro_f1_all6.mean():.4f} ± {b.macro_f1_all6.std(ddof=1):.4f}",
    "neither (p=0.589 paired t)", "difference reverses sign at seed 1337")
add("QWK (corrected, 5 ordinal), CV mean ± SD", "image (5-fold OOF)",
    f"{a.qwk_ordinal_5class.mean():.4f} ± {a.qwk_ordinal_5class.std(ddof=1):.4f}",
    f"{b.qwk_ordinal_5class.mean():.4f} ± {b.qwk_ordinal_5class.std(ddof=1):.4f}",
    "neither (p=0.977)", "difference 0.0002 — indistinguishable")
add("accuracy, CV mean", "image (5-fold OOF)", f"{a.accuracy.mean():.4f}", f"{b.accuracy.mean():.4f}",
    "neither", "")
cc, cf = cal[cal.name == "CURRENT structured (s42)"].iloc[0], cal[cal.name == "flat softmax baseline (s42)"].iloc[0]
add("Expected Calibration Error", "image (5-fold OOF)", f"{cc.ece_15bin:.4f}", f"{cf.ece_15bin:.4f}",
    "FLAT BASELINE (3.2× better)", "at essentially identical accuracy")
add("Brier score (multiclass)", "image (5-fold OOF)", f"{cc.brier_multiclass:.4f}", f"{cf.brier_multiclass:.4f}",
    "current (marginal)", "")
add("predictions above 0.90 confidence", "image (5-fold OOF)", f"{cc.frac_conf_over_90*100:.1f}%",
    f"{cf.frac_conf_over_90*100:.1f}%", "FLAT BASELINE", "the CORN cascade is strongly peaked")
add("high-confidence WRONG predictions", "image (5-fold OOF)", f"{int(cc.n_high_conf_wrong)}",
    f"{int(cf.n_high_conf_wrong)}", "FLAT BASELINE", "of 6,700 predictions each")
add("decode disagrees with argmax(prob)", "image (5-fold OOF)", f"{cc.argmax_disagrees_with_decode*100:.2f}%",
    "0.00%", "FLAT BASELINE", "AP-ROP threshold decode vs probability argmax")
ext = pd.read_csv(PKG / "13_EXTERNAL_GENERALIZATION/external_metrics_per_checkpoint.csv")
flat_ext = [full_metrics(*load_npz(ROOT / f"results/rop/locked_eval/flat_s42_ckpt{k}.npz")[:2]) for k in range(5)]
add("QWK (corrected), external locked, image level", "image (external locked)",
    f"{ext.qwk_ordinal_5class.mean():.4f} ± {ext.qwk_ordinal_5class.std(ddof=1):.4f}",
    f"{np.mean([m['qwk_ordinal_5class'] for m in flat_ext]):.4f} ± {np.std([m['qwk_ordinal_5class'] for m in flat_ext],ddof=1):.4f}",
    "FLAT BASELINE", "and the sign does NOT flip across seeds externally")
ac = aggd[(aggd.model_head == "CURRENT structured") & (aggd.rule == "max (protocol rule)")].iloc[0]
af = aggd[(aggd.model_head == "flat baseline") & (aggd.rule == "max (protocol rule)")].iloc[0]
add("QWK (corrected), external locked, EYE level (max rule)", "eye (external locked)",
    f"{ac.qwk_mean:.4f}", f"{af.qwk_mean:.4f}", "neither", "0.005 apart")
add("trainable parameters", "—", "20,512,600", "20,180,000 (approx; same backbone, 6-way head)",
    "flat (marginally smaller)", "the structured contribution is 6,405 params = 0.031%")
add("AP-ROP recall, CV pooled", "image (5-fold OOF)", "0.345", "0.364", "flat (marginal)", "")
add("Stage 2 recall, CV pooled", "image (5-fold OOF)", "0.631", "0.615", "current (marginal)", "")
add("taxonomy fidelity (AP-ROP off the stage scale)", "—", "YES", "NO",
    "CURRENT STRUCTURED", "the only dimension where the contributed head clearly wins")
mc = pd.DataFrame(cm_rows); save_csv(mc, PKG / "12_MODEL_COMPARISON/model_comparison.csv")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
x = np.arange(5); w = .38
axes[0].bar(x-w/2, a.sort_values("fold").macro_f1_all6, w, label="CURRENT structured", color=C_OK)
axes[0].bar(x+w/2, b.sort_values("fold").macro_f1_all6, w, label="flat softmax baseline", color=C_BAD)
axes[0].set_xticks(x); axes[0].set_xticklabels([f"fold {i}" for i in range(5)])
axes[0].set_ylabel("macro-F1 (6 classes)"); axes[0].legend(fontsize=7); axes[0].set_ylim(0, .9)
axes[0].set_title("Accuracy-family: indistinguishable\npaired t p=0.589, and the sign flips at seed 1337")
axes[1].bar([0, 1], [cc.ece_15bin, cf.ece_15bin], color=[C_OK, C_BAD])
for i, v in enumerate([cc.ece_15bin, cf.ece_15bin]): axes[1].annotate(f"{v:.4f}", (i, v), ha="center", va="bottom", fontsize=8)
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["CURRENT\nstructured", "flat softmax\nbaseline"])
axes[1].set_ylabel("Expected Calibration Error (lower is better)")
axes[1].set_title("Calibration: the BASELINE is clearly better\n3.2× lower ECE at the same accuracy")
axes[2].bar([0, 1], [ext.qwk_ordinal_5class.mean(), np.mean([m['qwk_ordinal_5class'] for m in flat_ext])],
            yerr=[ext.qwk_ordinal_5class.std(ddof=1), np.std([m['qwk_ordinal_5class'] for m in flat_ext], ddof=1)],
            capsize=4, color=[C_OK, C_BAD])
axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["CURRENT\nstructured", "flat softmax\nbaseline"])
axes[2].set_ylabel("QWK (corrected), external locked, image level")
axes[2].set_title("External hospital: the BASELINE is ahead\n(error bars = SD over 5 checkpoints, not a CI)")
fig.suptitle("CURRENT MODEL vs its own baseline — matched dataset, unit, metric definition and uncertainty method", y=1.02)
finish(fig, PKG / "12_MODEL_COMPARISON/structured_vs_flat.png", "image", "5-fold OOF + external locked")

# ══ 04_PREPROCESSING ════════════════════════════════════════════════════════
steps = [("Raw fundus photograph", "native camera resolution\n(512² – 1600×1200 across sites)"),
         ("circle_crop", "crop to the bounding box of\nnon-black pixels"),
         ("letterbox_square", "pad to a SQUARE with black bars\n— keeps aspect ratio identical\nacross hospitals (anti-shortcut)"),
         ("CLAHE", "contrast-limited adaptive histogram\nequalisation on the L channel of LAB"),
         ("Resize(384, 384)", "single isotropic resize"),
         ("ToTensor + ImageNet\nnormalisation", "mean (.485,.456,.406)\nstd (.229,.224,.225)"),
         ("EfficientNetV2-S\nbackbone", "→ 1280-d embedding"),
         ("3 heads → decode", "CORN ranks + AP-ROP prob\n→ 6-class label + prob vector")]
fig, ax = plt.subplots(figsize=(15, 3.2)); ax.axis("off")
for i, (t, s) in enumerate(steps):
    x = i * 1.9
    ax.add_patch(plt.Rectangle((x, 0.35), 1.62, 1.1, fc="#eaf2f8" if i < 6 else "#e8f0e8",
                               ec=C_OK if i < 6 else C_ALT, lw=1.2))
    ax.text(x+.81, 1.16, t, ha="center", va="center", fontsize=7.6, weight="bold")
    ax.text(x+.81, 0.72, s, ha="center", va="center", fontsize=6.2, color="#333")
    if i < len(steps)-1: ax.annotate("", (x+1.87, .9), (x+1.64, .9), arrowprops=dict(arrowstyle="->", lw=1.1))
ax.set_xlim(-.1, len(steps)*1.9); ax.set_ylim(0, 1.9)
ax.text(0, 1.72, "IDENTICAL in training (val transform), in the committed evaluation artifacts, and in the served web app "
        "— verified: webapp/inference.py calls build_transforms(cfg,'val') on the same config.",
        fontsize=7.5, color="#444")
ax.text(0, 0.12, "Training adds augmentation only: flips · rotation · colour jitter · hue/saturation jitter · RandomGrayscale · "
        "ResolutionJitter · RandomResizedCrop(0.6–1.0). NO augmentation at inference; the served path runs ONE forward pass (no TTA).",
        fontsize=7, color="#666")
finish(fig, PKG / "04_PREPROCESSING/preprocessing_pipeline.png", "image", "all splits")

dev = pd.read_csv(PREP / "test_dev.csv")
sample = dev.groupby("cls6", group_keys=False).sample(1, random_state=7)
fig, axes = plt.subplots(len(sample), 4, figsize=(10, 2.5*len(sample)))
for r, (_, row) in enumerate(sample.iterrows()):
    rgb = np.asarray(Image.open(ROOT / row.image_path).convert("RGB"))
    a1 = circle_crop(rgb); a2 = letterbox_square(a1); a3 = clahe(a2)
    for c, (imz, t) in enumerate(zip([rgb, a1, a2, a3],
            ["raw", "circle_crop", "+ letterbox", "+ CLAHE (model input)"])):
        axes[r, c].imshow(imz); axes[r, c].axis("off")
        if r == 0: axes[r, c].set_title(t, fontsize=8)
        if c == 0: axes[r, c].text(-0.08, .5, f"{row.cls6}\n{anon(row.image_path)}", rotation=90,
                                   transform=axes[r, c].transAxes, va="center", ha="center", fontsize=6.5)
fig.suptitle("Preprocessing applied to one de-identified example per class (external dev half)", y=1.0)
finish(fig, PKG / "04_PREPROCESSING/before_after_examples/preprocessing_examples.png", "image", "external dev half")

# ══ 15_DEPLOYMENT ═══════════════════════════════════════════════════════════
model = StructuredROPModel(cfg.model.arch, n_sites=3, pretrained=False)
model.load_state_dict(torch.load(CUR["served"], map_location="cpu")); model.eval()
tfm = build_transforms(cfg, "val")
img = Image.open(ROOT / dev.image_path.iloc[0]).convert("RGB")
lat = []
for name, fn in (("preprocess only (CPU)", lambda: tfm(img)),
                 ("forward pass only (CPU)", None), ("end-to-end per image (CPU)", None)):
    pass
t = tfm(img).unsqueeze(0)
with torch.no_grad():
    model(t)
    t0 = time.time(); [tfm(img) for _ in range(10)]; pre = (time.time()-t0)/10*1000
    t0 = time.time(); [model(t) for _ in range(10)]; fwd = (time.time()-t0)/10*1000
lat = pd.DataFrame([dict(stage="preprocessing (circle_crop+letterbox+CLAHE+resize+normalise)",
                         device="CPU (Apple Silicon)", ms_per_image=round(pre, 1)),
                    dict(stage="model forward pass (384×384)", device="CPU (Apple Silicon)", ms_per_image=round(fwd, 1)),
                    dict(stage="end-to-end single image", device="CPU (Apple Silicon)", ms_per_image=round(pre+fwd, 1)),
                    dict(stage="end-to-end single image (measured over 663 dev images in s01)",
                         device="CPU (Apple Silicon)",
                         ms_per_image=json.loads((PKG/"20_AUDIT/served_checkpoint_identification.json").read_text())["served_rerun_per_image_ms"])])
save_csv(lat, PKG / "15_DEPLOYMENT/inference_latency.csv")
mem = pd.DataFrame([dict(item="checkpoint file on disk", bytes=CUR["served"].stat().st_size,
                         mb=round(CUR["served"].stat().st_size/1e6, 1)),
                    dict(item="parameters in memory (fp32)", bytes=20512600*4, mb=round(20512600*4/1e6, 1)),
                    dict(item="site-adversary head (inert at inference, still shipped)",
                         bytes=328707*4, mb=round(328707*4/1e6, 2)),
                    dict(item="single 384×384×3 float32 input tensor", bytes=384*384*3*4, mb=round(384*384*3*4/1e6, 2))])
save_csv(mem, PKG / "15_DEPLOYMENT/memory_usage.csv")
(PKG / "15_DEPLOYMENT/model_size.txt").write_text(
    f"""CURRENT MODEL — SIZE
checkpoint file          : {CUR['served'].stat().st_size:,} bytes ({CUR['served'].stat().st_size/1e6:.1f} MB)
sha256                   : {hashlib.sha256(CUR['served'].read_bytes()).hexdigest()}
total parameters         : 20,512,600  (all trainable, none frozen)
  backbone EfficientNetV2-S: 20,177,488  (98.4%)
  CORN stage head          :      5,124
  AP-ROP branch            :      1,281
  site adversary (inert)   :    328,707  (1.6% — training-time machinery, shipped but unused)
the architectural contribution (stage head + AP-ROP branch) is 6,405 parameters = 0.031% of the model
""")

# ══ 01_MODEL ════════════════════════════════════════════════════════════════
shutil.copy(CUR["config"], PKG / "01_MODEL/model_configuration.yaml")
with open(PKG / "01_MODEL/architecture.txt", "w") as fh:
    fh.write("CURRENT MODEL — FULL MODULE TREE\n" + "="*70 + "\n")
    fh.write(str(model) + "\n")
with open(PKG / "01_MODEL/model_parameters.txt", "w") as fh:
    fh.write(f"{'module':<62}{'shape':<26}{'params':>12}\n" + "="*100 + "\n")
    for n, p in model.named_parameters():
        fh.write(f"{n:<62}{str(tuple(p.shape)):<26}{p.numel():>12,}\n")
    fh.write("="*100 + f"\n{'TOTAL':<88}{sum(p.numel() for p in model.parameters()):>12,}\n")
print(mc.to_string(index=False))
print("\nlatency\n", lat.to_string(index=False))
