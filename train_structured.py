#!/usr/bin/env python
"""Train the clinically-structured ROP staging model (ordinal + AP-ROP + site adversary).

    python train_structured.py --config configs/rop_staging_structured.yaml \
        --tag structured_effnetv2s_fold0 --set data.manifest=... model.arch=... seed=...

Separate from train.py on purpose: the multi-task losses, the ordinal mask on AP-ROP
samples, and the DANN lambda ramp do not fit build_loss's (loss_fn, decode) contract, and
bending the shared loop around them would risk the DR / ROP-binary training paths for no
benefit. Everything else — folds, preprocessing, logger, metrics — is reused, so results
stay directly comparable with the flat-softmax benchmark.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.common.config import load_config
from models.common.datasets import FundusDataset
from models.common.experiment_logger import ExperimentLogger, pick_device, set_seed
from models.common.metrics import compute_metrics
from models.common.preprocessing import build_transforms, cache_root_from_cfg
from models.common.structured import (AROP_LABEL, StructuredROPModel, dann_lambda,
                                      decode_6class, structured_losses)


class SiteDataset(FundusDataset):
    """FundusDataset plus the source-hospital index the adversary trains against."""

    def __init__(self, df, site_index, **kw):
        super().__init__(df, split=None, **kw)
        self.sites = self.df["source"].map(site_index).astype(int).tolist()

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, self.sites[idx]


def build_loaders(cfg, manifest):
    df = pd.read_csv(manifest)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    # Site vocabulary comes from TRAIN only and is saved with the run: an unseen-site val
    # row (never happens in our folds, but cheap to guard) maps to index 0 rather than
    # crashing — the site head is train-time-only machinery anyway.
    sites = sorted(tr.source.unique())
    site_index = {s: i for i, s in enumerate(sites)}
    idx = {"__missing__": 0, **site_index}
    # preprocess.cached: build_transforms then omits the fundus stage, so the Dataset must read
    # the cache (None otherwise). Same pairing as train_utils.build_dataloaders.
    cache_root = cache_root_from_cfg(cfg)
    tr_ds = SiteDataset(tr, lambda s: idx.get(s, 0),
                        transform=build_transforms(cfg, split="train"),
                        image_size=int(cfg.preprocess.image_size), cache_root=cache_root)
    va_ds = SiteDataset(va, lambda s: idx.get(s, 0),
                        transform=build_transforms(cfg, split="val"),
                        image_size=int(cfg.preprocess.image_size), cache_root=cache_root)

    counts = np.bincount(tr.label.values, minlength=6).astype(float)
    sampler = None
    if bool(cfg.train.get("weighted_sampler", False)):
        w = (1.0 / np.maximum(counts, 1.0))[tr.label.values]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(tr_ds), replacement=True)
    nw = int(cfg.train.get("num_workers", 4))
    bs = int(cfg.train.batch_size)
    pin = torch.cuda.is_available()
    return (DataLoader(tr_ds, batch_size=bs, sampler=sampler, shuffle=sampler is None,
                       num_workers=nw, pin_memory=pin, drop_last=True),
            DataLoader(va_ds, batch_size=bs, num_workers=nw, pin_memory=pin),
            counts, sites)


@torch.no_grad()
def evaluate(model, loader, device, thr):
    model.eval()
    yt, yp, pr = [], [], []
    for img, label, _site in loader:
        out = model(img.to(device))
        pred6, prob6 = decode_6class(out, arop_threshold=thr, variant=model.variant)
        yt += label.tolist()
        yp += pred6.cpu().tolist()
        pr.append(prob6.cpu().numpy())
    return np.asarray(yt), np.asarray(yp), np.concatenate(pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    set_seed(int(cfg.seed))
    device = pick_device()

    train_loader, val_loader, counts, sites = build_loaders(cfg, cfg.data.manifest)
    print(f"[structured] sites {sites} | train counts {counts.astype(int).tolist()}")

    # Ablation arms swap head structure only; "full" is the model the paper describes.
    variant = str(cfg.model.get("variant", "full"))
    model = StructuredROPModel(cfg.model.arch, n_sites=len(sites),
                               pretrained=bool(cfg.model.get("pretrained", True)),
                               dropout=float(cfg.model.get("dropout", 0.3)),
                               variant=variant).to(device)
    print(f"[structured] head variant: {variant}")

    # sqrt-damped positive weight for the AP-ROP branch (see config for the why).
    pw_cfg = cfg.train.get("arop_pos_weight", "auto_sqrt")
    n_pos = counts[AROP_LABEL]
    n_neg = counts.sum() - n_pos
    pos_weight = (math.sqrt(n_neg / max(n_pos, 1.0)) if pw_cfg == "auto_sqrt"
                  else float(pw_cfg))
    print(f"[structured] arop pos_weight {pos_weight:.2f}")
    pw = torch.tensor(pos_weight, device=device)

    epochs = int(cfg.train.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr),
                            weight_decay=float(cfg.train.get("weight_decay", 0.05)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(cfg.train.lr), steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.1, anneal_strategy="cos")
    use_amp = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    w_arop = float(cfg.train.get("w_arop", 1.0))
    w_site = float(cfg.train.get("w_site", 0.5))
    thr = float(cfg.train.get("arop_threshold", 0.5))
    names = list(cfg.data.class_names)
    logger = ExperimentLogger(cfg, tag=args.tag or f"structured_{cfg.model.arch}")

    history = []
    steps_total = epochs * len(train_loader)
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        sums = np.zeros(4)
        for img, label, site in train_loader:
            # DANN ramp on GLOBAL step, not epoch, so the adversary wakes smoothly.
            model.grl.lam = dann_lambda(step / max(steps_total, 1), max_lam=1.0)
            step += 1
            img = img.to(device)
            label = label.to(device)
            site = site.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(img)
                ls, la, lsite = structured_losses(out, label, site, arop_pos_weight=pw,
                                                  variant=variant)
                loss = ls + w_arop * la + w_site * lsite
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums += [t.detach().item() for t in (loss, ls, la, lsite)]

        n = len(train_loader)
        yt, yp, pr = evaluate(model, val_loader, device, thr)
        vm = compute_metrics(yt, yp, pr, num_classes=6, class_names=names)
        history.append({"epoch": epoch, "loss": sums[0] / n, "stage_loss": sums[1] / n,
                        "arop_loss": sums[2] / n, "site_loss": sums[3] / n,
                        "grl_lambda": model.grl.lam,
                        "val_macro_f1": vm["macro_f1"], "val_qwk": vm["qwk"]})
        # LAST epoch is kept, same rule as the benchmark — saving every epoch means the
        # file always holds the epoch the reported number came from.
        logger.save_weights(model.state_dict())
        print(f"Epoch {epoch:2d}/{epochs}  loss {sums[0]/n:.4f} "
              f"(stage {sums[1]/n:.4f} arop {sums[2]/n:.4f} site {sums[3]/n:.4f} "
              f"lam {model.grl.lam:.2f})  val macroF1 {vm['macro_f1']*100:.2f}%  "
              f"QWK {vm['qwk']:.3f}", flush=True)

    yt, yp, pr = evaluate(model, val_loader, device, thr)
    final = compute_metrics(yt, yp, pr, num_classes=6, class_names=names)
    # Per-sample probabilities are the raw material for everything the paper needs beyond a
    # point estimate: AP-ROP PR curves, bootstrapped CIs, per-session aggregation. A run
    # that only leaves metrics.json behind cannot support any of those retroactively.
    np.savez_compressed(logger.dir / "val_predictions.npz",
                        y_true=yt, y_pred=yp, prob6=pr)
    arop_rec = final["per_class"]["AP-ROP"]["recall"]
    arop_prec = final["per_class"]["AP-ROP"]["precision"]
    logger.log_metrics({"val": final, "history": history, "variant": variant,
                        "sites": sites, "arop_pos_weight": pos_weight})
    print(f"[val] acc {final['accuracy']*100:.2f}%  macroF1 {final['macro_f1']*100:.2f}%  "
          f"QWK {final['qwk']:.3f}")
    print(f"[val] AP-ROP recall {arop_rec:.3f}  precision {arop_prec:.3f}  "
          f"(flat-softmax baseline: 0.16-0.22 recall)")
    print(f"[train] done -> {logger.dir}")


if __name__ == "__main__":
    main()
