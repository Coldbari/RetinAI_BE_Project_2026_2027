"""Shared training / evaluation loop used by every disease and the sweep.

One loop: AdamW + OneCycleLR + AMP + gradient accumulation + grad clipping +
early stopping + per-epoch checkpointing (resume-safe for Kaggle's 12-hr cap).
``plot_run`` emits the full metric panel (loss, accuracy, P/R/F1, LR, confusion
matrix, ROC, PR) per the project's plotting standard.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import uda as uda_mod
from .bag_loss import bag_loader, bag_settings, pool_bags
from .datasets import FundusDataset, class_counts
from .fourier_aug import fourier_swap_batch
from .losses import build_loss, probabilities
from .metrics import compute_metrics
from .optim import build_optimizer_and_scheduler, format_group_lrs, head_param_names
from .preprocessing import aug_v4_settings, build_transforms, cache_root_from_cfg


# ── data ─────────────────────────────────────────────────────────────────────
def make_weighted_sampler(targets, num_classes):
    counts = np.bincount(targets, minlength=num_classes).astype(float)
    weights = 1.0 / np.maximum(counts, 1)
    sample_w = torch.tensor([weights[t] for t in targets], dtype=torch.float)
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


def build_dataloaders(cfg, manifest):
    size = int(cfg.preprocess.image_size)
    n = int(cfg.data.num_classes)
    bs = int(cfg.train.batch_size)
    nw = int(cfg.train.get("num_workers", 4))
    # None unless preprocess.cached. The transforms below then leave the fundus stage out, so
    # every Dataset MUST read from the cache; a miss raises rather than yielding a blank image.
    cache_root = cache_root_from_cfg(cfg)

    train_ds = FundusDataset(manifest, "train", build_transforms(cfg, "train"), size,
                             cache_root=cache_root)
    val_ds = FundusDataset(manifest, "val", build_transforms(cfg, "val"), size,
                           cache_root=cache_root)

    bag = bag_settings(cfg)              # None unless train.bag_loss.enabled (arm D)
    uda = uda_mod.uda_settings(cfg)      # None unless train.uda.enabled (arm F)
    if uda is not None and bag is not None:
        raise ValueError("train.uda and train.bag_loss cannot be combined")
    if bag is not None:
        if bool(cfg.train.get("weighted_sampler", False)):
            raise ValueError("train.bag_loss and train.weighted_sampler cannot be combined")
        train_loader = bag_loader(train_ds, bs, bag, seed=int(cfg.get("seed", 0)), num_workers=nw,
                                  pin_memory=torch.cuda.is_available())
    elif bool(cfg.train.get("weighted_sampler", False)):
        train_loader = DataLoader(train_ds, batch_size=bs, num_workers=nw,
                                  sampler=make_weighted_sampler(train_ds.targets, n),
                                  pin_memory=torch.cuda.is_available())
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=torch.cuda.is_available())

    loaders = {"train": train_loader, "val": val_loader,
               "counts": class_counts(train_ds.targets, n)}
    if uda is not None:                  # arm F: unlabelled target photos, image paths only
        loaders["target"] = uda_mod.target_loader(cfg, uda, num_workers=nw,
                                                  pin_memory=torch.cuda.is_available())

    # Optional held-out test split (ROP / Glaucoma carry one).
    test_ds = FundusDataset(manifest, "test", build_transforms(cfg, "val"), size,
                            cache_root=cache_root)
    if len(test_ds):
        loaders["test"] = DataLoader(test_ds, batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=torch.cuda.is_available())
    return loaders


# ── evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, decode_fn, num_classes, device, head="classification"):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for imgs, labels in loader:
        out = model(imgs.to(device))
        y_pred.extend(decode_fn(out).cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())
        if head == "classification":
            y_prob.extend(torch.softmax(out, 1).cpu().numpy().tolist())
    probs = np.asarray(y_prob) if y_prob else None
    return np.asarray(y_true), np.asarray(y_pred), probs


def select_metric(metrics, name):
    return {
        "qwk": metrics.get("qwk", 0.0),
        "auc": metrics.get("auc", 0.0),
        "recall": metrics.get("macro_recall", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "accuracy": metrics.get("accuracy", 0.0),
    }.get(name, metrics.get("macro_f1", 0.0))


# ── training ─────────────────────────────────────────────────────────────────
def train_model(cfg, model, loaders, logger, device):
    head = cfg.model.get("head", "classification")
    num_classes = int(cfg.data.num_classes)
    class_names = list(cfg.data.class_names)
    primary = cfg.eval.get("primary_metric", "macro_f1")

    loss_fn, decode_fn, _ = build_loss(cfg, loaders["counts"])
    loss_fn = loss_fn.to(device) if isinstance(loss_fn, nn.Module) else loss_fn

    epochs = int(cfg.train.epochs)
    grad_accum = max(1, int(cfg.train.get("grad_accum", 1)))
    grad_clip = float(cfg.train.get("grad_clip", 1.0))
    patience = int(cfg.train.get("patience", 7))
    # eval.early_stopping: false disables BOTH halves of val-based selection, which are
    # separate mechanisms and were previously entangled — patience alone stops the run early,
    # but the argmax-over-epochs checkpoint below is what actually biases the reported metric.
    # With 16-19 images in the rarest class per val fold, macro-F1 swings epoch to epoch, and
    # taking its maximum over 30 epochs reports the luckiest epoch rather than the model.
    # Default stays True so DR / ROP-binary runs behave exactly as before.
    early_stop = bool(cfg.eval.get("early_stopping", True))
    if not early_stop:
        patience = epochs + 1          # never trips
    keep_last = not early_stop         # save every epoch, so the file ends up at the last one

    steps_per_epoch = math.ceil(len(loaders["train"]) / grad_accum)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, steps_per_epoch)

    use_amp = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # Mixup regularizer (classification heads only) — strong defence against overfitting
    # on small datasets like ROP. Enabled via train.mixup_alpha > 0.
    mixup_alpha = float(cfg.train.get("mixup_alpha", 0.0))
    use_mixup = mixup_alpha > 0 and head == "classification"
    # Arm D: the train loader yields (images, labels, groups); bag batches pool each group's
    # logits by LSE before the loss, image batches (every row its own group) are left as they are.
    bag = bag_settings(cfg)
    if bag is not None and use_mixup:
        raise ValueError("train.bag_loss and mixup cannot be combined")
    # v4 recipe (Amendment 5b), each off by default (arm B): EMA weights, validated and saved as arm
    # F's teacher is; epochs 1..warmup train the head only; the batch-level Fourier amplitude swap.
    ema_decay = float(cfg.train.get("ema_decay", 0.0))
    warmup = int(cfg.train.get("head_warmup_epochs", 0))
    v4 = aug_v4_settings(cfg)
    fourier_p = v4["fourier_p"] if v4 is not None else 0.0
    if not 0.0 <= ema_decay < 1.0 or warmup < 0:
        raise ValueError("train.ema_decay must lie in [0, 1) and train.head_warmup_epochs be >= 0, "
                         f"got {ema_decay} and {warmup}")
    if bag is not None and (warmup > 0 or v4 is not None):
        raise ValueError("train.bag_loss cannot be combined with train.head_warmup_epochs or train.aug_v4")
    if warmup > 0:
        head_names = head_param_names(model)
        if not head_names:
            raise ValueError("train.head_warmup_epochs: the model has no head parameters "
                             "(optim.head_param_names is empty)")
        own_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    # Arm F: self-ensembling. The teacher is a copy of the student made here, after the model and
    # its head are built; it is validated and saved in place of the student (see below).
    uda = uda_mod.uda_settings(cfg)
    if uda is not None:
        if bag is not None or use_mixup:
            raise ValueError("train.uda cannot be combined with train.bag_loss or mixup")
        if ema_decay > 0 or v4 is not None:
            raise ValueError("train.uda cannot be combined with train.ema_decay or train.aug_v4")
        if "target" not in loaders:
            raise ValueError("train.uda is on but loaders has no 'target' (use build_dataloaders)")
        teacher = uda_mod.EMATeacher(model, uda["ema_decay"])
        target_batches = uda_mod.cycle(loaders["target"])
        ramp_steps = uda["ramp_epochs"] * len(loaders["train"])
    ema = uda_mod.EMATeacher(model, ema_decay) if ema_decay > 0 else None
    eval_model = teacher.model if uda is not None else ema.model if ema is not None else model

    history = {k: [] for k in ["train_loss", "lr", "val_accuracy", "val_macro_f1",
                               "val_macro_precision", "val_macro_recall", "val_primary"]}
    best = -1.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if warmup > 0:  # epochs 1..warmup: the head only; after them every parameter's own flag
            for n, p in model.named_parameters():
                p.requires_grad_(own_grad[n] and (epoch > warmup or n in head_names))
        running = 0.0
        optimizer.zero_grad()
        n_batches = len(loaders["train"])
        t0, imgs_seen = time.time(), 0
        tally = {"img": [0, 0.0], "bag": [0, 0.0], "bags": 0, "frames": 0}
        u_tally = {"n": 0, "loss": 0.0, "mask": 0.0, "src": 0.0, "w": 0.0, "tmax": 0.0}
        for i, batch in enumerate(loaders["train"]):
            imgs, labels = batch[0], batch[1]
            imgs_seen += imgs.size(0)
            imgs = imgs.to(device)
            if fourier_p > 0:  # on the normalised batch, on the device; the run's seeded torch RNG
                imgs = fourier_swap_batch(imgs, v4["fourier_block"], fourier_p)
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if use_mixup:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(imgs.size(0), device=device)
                    imgs = lam * imgs + (1 - lam) * imgs[perm]
                    out = model(imgs)
                    loss = (lam * loss_fn(out, labels)
                            + (1 - lam) * loss_fn(out, labels[perm])) / grad_accum
                elif uda is not None:
                    weak, strong = (t.to(device) for t in next(target_batches))
                    with torch.no_grad():
                        t_logits = teacher.model(weak)
                    both = model(torch.cat([imgs, strong]))   # one student pass, source + target
                    out, s_logits = both[:imgs.size(0)], both[imgs.size(0):]
                    src_loss = loss_fn(out, labels)
                    tgt_loss, mask_rate = uda_mod.consistency_loss(s_logits, t_logits,
                                                                   uda["conf_threshold"])
                    w = uda_mod.ramp_weight((epoch - 1) * n_batches + i, ramp_steps, uda["weight"])
                    loss = (src_loss + w * tgt_loss) / grad_accum
                    u_tally["n"] += 1
                    u_tally["loss"] += tgt_loss.item()
                    u_tally["mask"] += mask_rate.item()
                    u_tally["src"] += src_loss.item()
                    u_tally["w"] = w
                    u_tally["tmax"] += torch.softmax(t_logits.float(), 1).max(1).values.mean().item()
                else:
                    out = model(imgs)
                    if bag is not None:
                        out, labels = pool_bags(out, labels, batch[2].to(device), bag["tau"])
                    loss = loss_fn(out, labels) / grad_accum
            scaler.scale(loss).backward()
            if bag is not None:
                kind = "bag" if out.size(0) < imgs.size(0) else "img"
                tally[kind][0] += 1
                tally[kind][1] += loss.item() * grad_accum
                if kind == "bag":
                    tally["bags"] += out.size(0)
                    tally["frames"] += imgs.size(0)

            if (i + 1) % grad_accum == 0 or (i + 1) == n_batches:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                if uda is not None:
                    teacher.update(model)
                if ema is not None:
                    ema.update(model)
                if epoch == 1 and i + 1 == grad_accum:
                    print(format_group_lrs(optimizer))
            if epoch == 1 and i + 1 == 50:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                print(f"[throughput] {imgs_seen / (time.time() - t0):.1f} img/s")
                print(format_group_lrs(optimizer))
            running += loss.item() * grad_accum

        if device.type == "cuda":
            print(f"[memory] epoch {epoch}: peak allocated "
                  f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB of "
                  f"{torch.cuda.get_device_properties(device).total_memory / 2**30:.2f} GiB", flush=True)
        if bag is not None:
            (ni, li), (nb, lb) = tally["img"], tally["bag"]
            print(f"[bag] epoch {epoch}: {ni} image batches (loss {li / max(1, ni):.4f}), "
                  f"{nb} bag batches = {tally['bags']} sessions / {tally['frames']} frames "
                  f"(loss {lb / max(1, nb):.4f})", flush=True)
        if uda is not None:
            nu = max(1, u_tally["n"])
            print(f"[uda] epoch {epoch}: mask_rate={u_tally['mask'] / nu:.3f} "
                  f"target_loss={u_tally['loss'] / nu:.5f} weight={u_tally['w']:.3f} "
                  f"source_loss={u_tally['src'] / nu:.4f} "
                  f"teacher_maxprob_mean={u_tally['tmax'] / nu:.3f}", flush=True)
        if ema is not None or warmup > 0 or fourier_p > 0:
            print(f"[v4] epoch {epoch}: warmup={epoch <= warmup} ema_decay={ema_decay} "
                  f"fourier_p={fourier_p}", flush=True)

        # validate (arm F: the teacher; v4 ema_decay: the EMA)
        yt, yp, pr = evaluate(eval_model, loaders["val"], decode_fn, num_classes, device, head)
        vm = compute_metrics(yt, yp, pr, num_classes, class_names)
        val_primary = select_metric(vm, primary)

        history["train_loss"].append(running / max(1, n_batches))
        history["lr"].append(scheduler.get_last_lr()[0])
        history["val_accuracy"].append(vm["accuracy"])
        history["val_macro_f1"].append(vm["macro_f1"])
        history["val_macro_precision"].append(vm["macro_precision"])
        history["val_macro_recall"].append(vm["macro_recall"])
        history["val_primary"].append(val_primary)

        improved = val_primary > best
        if improved:
            best, no_improve = val_primary, 0
        else:
            no_improve += 1
        if improved or keep_last:
            # Arm F: the TEACHER's weights go into weights.pth, so train.py's final val pass
            # (val_predictions.npz) and the kernel's dev scoring (dev_predictions.npz) are the teacher.
            # The same holds for the v4 EMA.
            logger.save_weights(eval_model.state_dict())
        flag = "  ✓ saved" if (improved or keep_last) else ""

        print(f"Epoch {epoch:2d}/{epochs}  loss {history['train_loss'][-1]:.4f}  "
              f"acc {vm['accuracy']*100:.2f}%  F1 {vm['macro_f1']*100:.2f}%  "
              f"{primary} {val_primary:.4f}{flag}")

        # resume-safe checkpoint
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "best": best}
                   | ({"teacher": teacher.model.state_dict()} if uda is not None else {})
                   | ({"ema": ema.model.state_dict()} if ema is not None else {}),
                   logger.dir / "last_checkpoint.pth")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best {primary}={best:.4f}).")
            break

    # When keeping the last epoch, return ITS score. Returning `best` would report a
    # maximum-over-epochs number next to a last-epoch checkpoint — two different models.
    return history, (val_primary if keep_last else best)


# ── inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_with_tta(model, image_pil, tta_transforms, device, head="classification"):
    model.eval()
    probs = []
    for tfm in tta_transforms:
        t = tfm(image_pil).unsqueeze(0).to(device)
        out = model(t)
        probs.append(probabilities(out, head)[0].detach().cpu())
    return torch.stack(probs).mean(0)


# ── plotting ─────────────────────────────────────────────────────────────────
def plot_run(history, final_eval, cfg, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    yt = final_eval.get("y_true")
    yp = final_eval.get("y_pred")
    prob = final_eval.get("y_prob")
    num_classes = int(cfg.data.num_classes)
    class_names = list(cfg.data.class_names)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f"{cfg.get('disease', '').upper()} — {cfg.model.arch} — training & eval",
                 fontsize=15, fontweight="bold")

    ax[0, 0].plot(epochs, history["train_loss"], color="#60A5FA"); ax[0, 0].set_title("Train Loss")
    ax[0, 1].plot(epochs, [v*100 for v in history["val_accuracy"]], color="#34D399")
    ax[0, 1].set_title("Val Accuracy (%)")
    ax[0, 2].plot(epochs, history["lr"], color="#FBBF24"); ax[0, 2].set_yscale("log")
    ax[0, 2].set_title("Learning Rate")

    ax[1, 0].plot(epochs, [v*100 for v in history["val_macro_precision"]], label="Precision")
    ax[1, 0].plot(epochs, [v*100 for v in history["val_macro_recall"]], label="Recall")
    ax[1, 0].plot(epochs, [v*100 for v in history["val_macro_f1"]], label="F1")
    ax[1, 0].set_title("Val Macro P/R/F1 (%)"); ax[1, 0].legend()
    ax[1, 1].plot(epochs, history["val_primary"], color="#A78BFA")
    ax[1, 1].set_title(f"Val {cfg.eval.get('primary_metric','metric')}")

    # confusion matrix
    if yt is not None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(yt, yp, labels=list(range(num_classes)))
        im = ax[1, 2].imshow(cm, cmap="Blues")
        ax[1, 2].set_title("Confusion Matrix")
        ax[1, 2].set_xticks(range(num_classes)); ax[1, 2].set_yticks(range(num_classes))
        ax[1, 2].set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax[1, 2].set_yticklabels(class_names, fontsize=8)
        for r in range(num_classes):
            for c in range(num_classes):
                ax[1, 2].text(c, r, cm[r, c], ha="center", va="center", fontsize=8)

    # ROC + PR (binary) or per-class one-vs-rest
    if prob is not None and yt is not None:
        prob = np.asarray(prob)
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(yt, prob[:, 1])
            ax[2, 0].plot(fpr, tpr, color="#EF4444"); ax[2, 0].plot([0, 1], [0, 1], "--", color="gray")
            ax[2, 0].set_title("ROC"); ax[2, 0].set_xlabel("FPR"); ax[2, 0].set_ylabel("TPR")
            prec, rec, _ = precision_recall_curve(yt, prob[:, 1])
            ax[2, 1].plot(rec, prec, color="#10B981")
            ax[2, 1].set_title("PR Curve"); ax[2, 1].set_xlabel("Recall"); ax[2, 1].set_ylabel("Precision")
        else:
            for k in range(num_classes):
                fpr, tpr, _ = roc_curve((yt == k).astype(int), prob[:, k])
                ax[2, 0].plot(fpr, tpr, label=class_names[k])
            ax[2, 0].plot([0, 1], [0, 1], "--", color="gray")
            ax[2, 0].set_title("ROC (one-vs-rest)"); ax[2, 0].legend(fontsize=7)
            for k in range(num_classes):
                prec, rec, _ = precision_recall_curve((yt == k).astype(int), prob[:, k])
                ax[2, 1].plot(rec, prec, label=class_names[k])
            ax[2, 1].set_title("PR (one-vs-rest)"); ax[2, 1].legend(fontsize=7)

    ax[2, 2].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {save_path}")
