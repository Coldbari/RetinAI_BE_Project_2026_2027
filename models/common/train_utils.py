"""Shared training / evaluation loop used by every disease and the sweep.

One loop: AdamW + OneCycleLR + AMP + gradient accumulation + grad clipping +
early stopping + per-epoch checkpointing (resume-safe for Kaggle's 12-hr cap).
``plot_run`` emits the full metric panel (loss, accuracy, P/R/F1, LR, confusion
matrix, ROC, PR) per the project's plotting standard.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import FundusDataset, class_counts
from .losses import build_loss, probabilities
from .metrics import compute_metrics
from .preprocessing import build_transforms


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

    train_ds = FundusDataset(manifest, "train", build_transforms(cfg, "train"), size)
    val_ds = FundusDataset(manifest, "val", build_transforms(cfg, "val"), size)

    if bool(cfg.train.get("weighted_sampler", False)):
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

    # Optional held-out test split (ROP / Glaucoma carry one).
    test_ds = FundusDataset(manifest, "test", build_transforms(cfg, "val"), size)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr),
                                  weight_decay=float(cfg.train.get("weight_decay", 1e-4)))
    steps_per_epoch = math.ceil(len(loaders["train"]) / grad_accum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(cfg.train.lr), steps_per_epoch=steps_per_epoch,
        epochs=epochs, pct_start=0.1, anneal_strategy="cos")

    use_amp = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # Mixup regularizer (classification heads only) — strong defence against overfitting
    # on small datasets like ROP. Enabled via train.mixup_alpha > 0.
    mixup_alpha = float(cfg.train.get("mixup_alpha", 0.0))
    use_mixup = mixup_alpha > 0 and head == "classification"

    history = {k: [] for k in ["train_loss", "lr", "val_accuracy", "val_macro_f1",
                               "val_macro_precision", "val_macro_recall", "val_primary"]}
    best = -1.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        n_batches = len(loaders["train"])
        for i, (imgs, labels) in enumerate(loaders["train"]):
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if use_mixup:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(imgs.size(0), device=device)
                    imgs = lam * imgs + (1 - lam) * imgs[perm]
                    out = model(imgs)
                    loss = (lam * loss_fn(out, labels)
                            + (1 - lam) * loss_fn(out, labels[perm])) / grad_accum
                else:
                    out = model(imgs)
                    loss = loss_fn(out, labels) / grad_accum
            scaler.scale(loss).backward()

            if (i + 1) % grad_accum == 0 or (i + 1) == n_batches:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            running += loss.item() * grad_accum

        # validate
        yt, yp, pr = evaluate(model, loaders["val"], decode_fn, num_classes, device, head)
        vm = compute_metrics(yt, yp, pr, num_classes, class_names)
        val_primary = select_metric(vm, primary)

        history["train_loss"].append(running / max(1, n_batches))
        history["lr"].append(scheduler.get_last_lr()[0])
        history["val_accuracy"].append(vm["accuracy"])
        history["val_macro_f1"].append(vm["macro_f1"])
        history["val_macro_precision"].append(vm["macro_precision"])
        history["val_macro_recall"].append(vm["macro_recall"])
        history["val_primary"].append(val_primary)

        flag = ""
        if val_primary > best:
            best, no_improve = val_primary, 0
            logger.save_weights(model.state_dict())
            flag = "  ✓ saved"
        else:
            no_improve += 1

        print(f"Epoch {epoch:2d}/{epochs}  loss {history['train_loss'][-1]:.4f}  "
              f"acc {vm['accuracy']*100:.2f}%  F1 {vm['macro_f1']*100:.2f}%  "
              f"{primary} {val_primary:.4f}{flag}")

        # resume-safe checkpoint
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "best": best},
                   logger.dir / "last_checkpoint.pth")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best {primary}={best:.4f}).")
            break

    return history, best


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
