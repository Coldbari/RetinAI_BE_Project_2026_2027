"""14_EXPLAINABILITY — Grad-CAM audit of the CURRENT (served) model, run fresh."""
import sys, hashlib
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
sys.path.insert(0, str(ROOT))
import torch, cv2
from PIL import Image
from models.common.structured import StructuredROPModel, decode_6class
from models.common.config import load_config
from models.common.preprocessing import build_transforms, FundusPreprocess, circle_crop, letterbox_square
from models.common.data_quality import retina_mask

SALT = "retinai-evidence-package-v1"
anon = lambda s: "CASE-" + hashlib.sha256((SALT+str(s)).encode()).hexdigest()[:10].upper()

cfg = load_config(str(CUR["config"]))
model = StructuredROPModel(cfg.model.arch, n_sites=3, pretrained=False)
model.load_state_dict(torch.load(CUR["served"], map_location="cpu")); model.eval()
tfm = build_transforms(cfg, "val")
S = int(cfg.preprocess.image_size)

feats, grads = {}, {}
target = model.backbone.features[-1]
target.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
target.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))

def cam_for(img):
    x = tfm(img).unsqueeze(0); x.requires_grad_(False)
    out = model(x)
    pred, prob = decode_6class(out, arop_threshold=0.5, variant="full")
    score = torch.log(prob[0, 1:].sum() + 1e-8)      # the SERVED objective: P(any ROP)
    model.zero_grad(); score.backward()
    a, g = feats["a"][0], grads["g"][0]
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((w * a).sum(0)).detach().numpy()
    cam = cv2.resize(cam, (S, S))
    cam = cam / (cam.max() + 1e-9)
    return cam, int(pred[0]), prob[0].detach().numpy()

def masks_for(img):
    """The model's own 384x384 frame, split into retina / synthetic letterbox padding /
    real camera surround — so invented pixels cannot masquerade as 'reading the frame'."""
    rgb = np.asarray(img.convert("RGB"))
    rgb = circle_crop(rgb)
    h, w = rgb.shape[:2]; side = max(h, w)
    pad = np.zeros((side, side), bool)
    y0, x0 = (side-h)//2, (side-w)//2
    pad[y0:y0+h, x0:x0+w] = True                      # True = real pixels
    box = letterbox_square(rgb)
    ret = retina_mask(box)
    ret = np.zeros((side, side), bool) if ret is None else ret.astype(bool)
    rs = lambda m: cv2.resize(m.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST).astype(bool)
    real, retina = rs(pad), rs(ret)
    retina &= real
    return retina, (~real), (real & ~retina), cv2.resize(box, (S, S))

dev = pd.read_csv(PREP / "test_dev.csv")
rng = np.random.default_rng(42)
sel = (dev.groupby("cls6", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), 24), random_state=42), include_groups=False)
          .join(dev[["cls6"]]).reset_index(drop=True)) if False else \
      dev.groupby("cls6", group_keys=False).sample(n=20, random_state=42, replace=True).drop_duplicates("image_path")
rows, store = [], []
for _, r in sel.iterrows():
    img = Image.open(ROOT / r.image_path).convert("RGB")
    cam, pred, prob = cam_for(img)
    retina, padding, surround, disp = masks_for(img)
    tot = cam.sum() + 1e-9
    peak = np.unravel_index(cam.argmax(), cam.shape)
    where = "retina" if retina[peak] else ("synthetic_padding" if padding[peak] else "camera_surround")
    rows.append(dict(case_id=anon(r.image_path), true_class=r.cls6, predicted_class=CLASSES[pred],
                     correct=(CLASSES[pred] == r.cls6), confidence=float(prob.max()),
                     any_rop_score=float(prob[1:].sum()),
                     attention_on_retina_pct=100*float(cam[retina].sum()/tot),
                     attention_on_synthetic_padding_pct=100*float(cam[padding].sum()/tot),
                     attention_on_camera_surround_pct=100*float(cam[surround].sum()/tot),
                     peak_location=where,
                     retina_area_pct=100*float(retina.mean())))
    store.append((anon(r.image_path), disp, cam, retina, r.cls6, CLASSES[pred], float(prob.max()), where))
cam_df = pd.DataFrame(rows)
save_csv(cam_df, PKG / "14_EXPLAINABILITY/gradcam_quality_per_image.csv")

summ = dict(n_images=len(cam_df),
            mean_attention_on_retina_pct=float(cam_df.attention_on_retina_pct.mean()),
            mean_attention_on_synthetic_padding_pct=float(cam_df.attention_on_synthetic_padding_pct.mean()),
            mean_attention_on_camera_surround_pct=float(cam_df.attention_on_camera_surround_pct.mean()),
            peak_outside_retina_pct=100*float((cam_df.peak_location != "retina").mean()),
            peak_on_synthetic_padding_pct=100*float((cam_df.peak_location == "synthetic_padding").mean()),
            mean_retina_area_pct=float(cam_df.retina_area_pct.mean()),
            attention_lift_vs_area=float(cam_df.attention_on_retina_pct.mean()/cam_df.retina_area_pct.mean()),
            note="lift 1.0 = attention is spread exactly as the retina occupies the frame; <1 = the "
                 "model attends non-retina MORE than chance given its area")
save_json(summ, PKG / "14_EXPLAINABILITY/gradcam_quality_summary.json")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
axes[0].hist(cam_df.attention_on_retina_pct, bins=20, color=C_OK)
axes[0].axvline(cam_df.retina_area_pct.mean(), ls="--", color=C_BAD)
axes[0].annotate(f"retina occupies {cam_df.retina_area_pct.mean():.0f}% of the frame\n"
                 "(attention left of this line = worse than chance)",
                 (cam_df.retina_area_pct.mean()+1, axes[0].get_ylim()[1]*.6), fontsize=7, color=C_BAD)
axes[0].set_xlabel("% of Grad-CAM attention falling inside the retina")
axes[0].set_ylabel(f"images (n={len(cam_df)})")
axes[0].set_title(f"Attention inside the retina\nmean {summ['mean_attention_on_retina_pct']:.1f}%")
pk = cam_df.peak_location.value_counts()
axes[1].bar(pk.index, pk.values, color=[C_ALT if i == "retina" else C_BAD for i in pk.index])
for i, v in enumerate(pk.values): axes[1].annotate(f"{v}\n({v/len(cam_df)*100:.0f}%)", (i, v), ha="center", va="bottom", fontsize=8)
axes[1].set_ylabel("images"); axes[1].tick_params(axis="x", labelsize=7.5)
axes[1].set_title(f"Where the CAM PEAK lands\n{summ['peak_outside_retina_pct']:.0f}% of peaks are outside the retina")
for cl, c in zip(CLASSES, [C_OK, "#4f9ecf", C_ALT, "#e0a458", C_MUT, C_BAD]):
    s = cam_df[cam_df.true_class == cl]
    if len(s): axes[2].scatter(s.attention_on_retina_pct, s.confidence, s=22, color=c, label=f"{cl} (n={len(s)})", alpha=.8)
axes[2].set_xlabel("% attention inside the retina"); axes[2].set_ylabel("model confidence")
axes[2].legend(fontsize=6); axes[2].set_title("Confidence is unrelated to whether the model\nlooked at the retina")
finish(fig, PKG / "14_EXPLAINABILITY/gradcam_quality_analysis.png", "image",
       f"external dev half, stratified sample n={len(cam_df)}")

def overlay_grid(items, path, title):
    n = len(items); rows_ = int(np.ceil(n/4))
    fig, axes = plt.subplots(rows_, 4, figsize=(11.5, 3.1*rows_)); axes = np.atleast_1d(axes).ravel()
    for ax, (cid, disp, cam, retina, t, pd_, cf, where) in zip(axes, items):
        ax.imshow(disp); ax.imshow(cam, cmap="jet", alpha=0.42); ax.axis("off")
        ax.contour(retina.astype(float), levels=[0.5], colors="w", linewidths=0.7)
        ax.set_title(f"{cid}  true {t} → pred {pd_}\nconf {cf:.2f} · CAM peak on {where.replace('_',' ')}",
                     fontsize=7, color=C_ALT if where == "retina" else C_BAD)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle(title, y=1.0, fontsize=10)
    return finish(fig, path, "image", "external dev half")

good = [s for s in store if s[7] == "retina"][:8]
bad  = [s for s in store if s[7] != "retina"][:8]
overlay_grid(good, PKG / "14_EXPLAINABILITY/gradcam_examples/gradcam_on_retina.png",
             "Grad-CAM examples where the peak IS on the retina (white outline = retinal field of view)")
overlay_grid(bad, PKG / "14_EXPLAINABILITY/gradcam_failure_examples/gradcam_off_retina.png",
             "FAILURE CASES — Grad-CAM peak OUTSIDE the retina, including on letterbox padding the "
             "preprocessing invented")
for cid, disp, cam, retina, t, pd_, cf, where in good[:6] + bad[:6]:
    out = (PKG / "14_EXPLAINABILITY" / ("gradcam_examples" if where == "retina" else "gradcam_failure_examples"))
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.imshow(disp); ax.imshow(cam, cmap="jet", alpha=.42); ax.axis("off")
    ax.set_title(f"{cid}\ntrue {t} → pred {pd_} (conf {cf:.2f})", fontsize=7)
    fig.savefig(out / f"{cid}.png", bbox_inches="tight"); plt.close(fig)
print(json.dumps(summ, indent=1))
print(cam_df.groupby("true_class")[["attention_on_retina_pct","attention_on_synthetic_padding_pct"]].mean().round(1).to_string())
