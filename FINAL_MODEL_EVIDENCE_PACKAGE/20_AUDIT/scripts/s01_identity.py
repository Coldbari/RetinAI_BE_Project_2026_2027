"""Establish the CURRENT MODEL's identity, and resolve the served-checkpoint question
empirically by re-running the served weights over test_dev and matching the output
against the five committed per-fold checkpoint predictions."""
import sys, hashlib, json, time, platform
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
sys.path.insert(0, str(ROOT))
import torch, yaml
from PIL import Image
from models.common.structured import StructuredROPModel, decode_6class
from models.common.config import load_config
from models.common.preprocessing import build_transforms

out = {}
w = CUR["served"]
out["served_weights_path"] = str(w.relative_to(ROOT))
out["served_weights_bytes"] = w.stat().st_size
out["served_weights_sha256"] = hashlib.sha256(w.read_bytes()).hexdigest()
out["served_weights_mtime"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(w.stat().st_mtime))

cfg = load_config(str(CUR["config"]))
state = torch.load(w, map_location="cpu")
n_sites = state["site_head.3.bias"].shape[0]
model = StructuredROPModel(cfg.model.arch, n_sites=n_sites, pretrained=False)
model.load_state_dict(state, strict=True)
model.eval()
out["state_dict_keys"] = len(state)
out["n_sites_in_checkpoint"] = int(n_sites)
out["total_parameters"] = sum(p.numel() for p in model.parameters())
out["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
out["frozen_parameters"] = 0
out["backbone_parameters"] = sum(p.numel() for p in model.backbone.parameters())
out["head_parameters"] = {n: sum(p.numel() for p in getattr(model, n).parameters())
                          for n in ("stage_head", "arop_head", "site_head")}

# ── empirical checkpoint identification ──────────────────────────────────────
df = pd.read_csv(PREP / "test_dev.csv")
tfm = build_transforms(cfg, "val")
thr = float(cfg.train.get("arop_threshold", 0.5))
preds, probs = [], []
t0 = time.time()
with torch.no_grad():
    for i, p in enumerate(df.image_path):
        x = tfm(Image.open(ROOT / p).convert("RGB")).unsqueeze(0)
        o = model(x)
        pr, pb = decode_6class(o, arop_threshold=thr, variant="full")
        preds.append(int(pr[0])); probs.append(pb[0].numpy())
        if i % 200 == 0: print(f"  {i}/{len(df)}", flush=True)
served_pred = np.array(preds); served_prob = np.array(probs)
out["served_rerun_seconds"] = round(time.time() - t0, 1)
out["served_rerun_per_image_ms"] = round((time.time() - t0) / len(df) * 1000, 1)
np.savez_compressed(PKG / "20_AUDIT/served_checkpoint_dev_rerun.npz",
                    y_true=df.label.values, y_pred=served_pred, prob6=served_prob)

match = {}
for k in range(5):
    z = np.load(ROOT / f"results/rop/dev_eval/structured_s42_ckpt{k}.npz")
    agree = float((z["y_pred"] == served_pred).mean())
    pdiff = float(np.abs(z["prob6"].astype(float) - served_prob).max())
    match[f"fold{k}_ckpt"] = dict(pred_agreement=agree, max_abs_prob_diff=pdiff)
for k in range(5):
    z = np.load(ROOT / f"results/rop/dev_eval/structured_s1337_ckpt{k}.npz")
    match[f"fold{k}_ckpt_seed1337"] = dict(pred_agreement=float((z["y_pred"] == served_pred).mean()),
                                           max_abs_prob_diff=float(np.abs(z["prob6"].astype(float) - served_prob).max()))
out["served_checkpoint_match_vs_committed_dev_predictions"] = match
best = max(match.items(), key=lambda kv: kv[1]["pred_agreement"])
out["best_match"] = dict(checkpoint=best[0], **best[1])
out["identified"] = bool(best[1]["pred_agreement"] > 0.995 and best[1]["max_abs_prob_diff"] < 1e-3)
out["environment"] = dict(torch=torch.__version__, python=platform.python_version(),
                          platform=platform.platform(), numpy=np.__version__)
out["git_commit"] = git_hash()
save_json(out, PKG / "20_AUDIT/served_checkpoint_identification.json")
print(json.dumps({k: v for k, v in out.items() if k != "served_checkpoint_match_vs_committed_dev_predictions"}, indent=1, default=str))
print("\nMATCH TABLE"); print(json.dumps(match, indent=1))
