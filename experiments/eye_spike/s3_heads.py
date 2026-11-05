"""Step 3 — heads on frozen features, 5-fold CV over BAGS + train-on-all -> Multi-View DEV eyes.

    .venv/bin/python experiments/eye_spike/s3_heads.py [enc ...]

Methods (all natural sampling, UNWEIGHTED cross-entropy — fixed before any result was seen):
  img_lr_max   sklearn multinomial logistic regression on single images (image labels); C picked
               from {0.01, 0.1, 1} by image log-loss on an inner patient-disjoint fold of the
               training folds; bag prediction = max predicted class over ALL the bag's images.
  img_mlp_max  same phi-MLP as the MIL heads, trained on single images; bag = max predicted class.
  meanpool     phi-MLP -> masked mean over instances -> linear, bag label.
  maxpool      phi-MLP -> masked max  over instances -> linear, bag label.
  attn         phi-MLP -> gated attention (Ilse et al. 2018) -> linear, bag label.
  MIL training: every bag shows exactly K=4 instances (without replacement if n>=4, with
  replacement otherwise) so bag size is not visible; eval: up to 16 instances, deterministic
  (evenly spaced over frame order), masked. phi = Linear(d,128)-ReLU-Dropout(.25).
  AdamW lr 1e-3 wd 1e-2, cosine, 30 epochs, no early stopping, seeds {0,1}.
  size_only    LR on one-hot bag-size bins (the shortcut control).
Writes cache/preds_<enc>.npz: oof/dev bag predictions per method x seed.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CACHE, HERE, load_pool

K_TRAIN, K_EVAL, EPOCHS, SEEDS = 4, 16, 30, (0, 1)
torch.set_num_threads(1)


def load_X(enc):
    z = np.load(CACHE / f"feat_{enc}.npz")
    if "pool_cls" in z.files:
        return (np.concatenate([z["pool_cls"], z["pool_mean"]], 1).astype(np.float32),
                np.concatenate([z["dev_cls"], z["dev_mean"]], 1).astype(np.float32))
    return z["pool_pool"].astype(np.float32), z["dev_pool"].astype(np.float32)


def bag_index(img_bag, order):
    """list over bags of image-row arrays sorted by frame order."""
    df = pd.DataFrame({"b": img_bag, "o": order, "i": np.arange(len(img_bag))}).sort_values(["b", "o"])
    return [g.i.values for _, g in df.groupby("b", sort=True)]


class Head(nn.Module):
    def __init__(self, d, pool, h=128, p=0.25):
        super().__init__()
        self.pool = pool
        self.phi = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p))
        if pool == "attn":
            self.V, self.U, self.w = nn.Linear(h, 64), nn.Linear(h, 64), nn.Linear(64, 1)
        self.cls = nn.Linear(h, 6)

    def forward(self, x, mask, return_attn=False):
        h = self.phi(x)                                            # (B, N, h)
        a = None
        if self.pool == "none":
            z = h[:, 0]
        elif self.pool == "mean":
            m = mask[..., None].float()
            z = (h * m).sum(1) / m.sum(1)
        elif self.pool == "max":
            z = h.masked_fill(~mask[..., None], float("-inf")).max(1).values
        else:
            s = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
            a = torch.softmax(s.masked_fill(~mask, float("-inf")), 1)
            z = (a[..., None] * h).sum(1)
        out = self.cls(z)
        return (out, a) if return_attn else out


def train_head(X, units, y, pool, seed, bs):
    """units: list of index arrays (bags) — for pool=='none' each unit is one image."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Head(X.shape[1], pool)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = EPOCHS * int(np.ceil(len(units) / bs))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))
    k = 1 if pool == "none" else K_TRAIN
    L = np.array([len(u) for u in units])
    M = np.full((len(units), L.max()), -1, dtype=np.int64)
    for j, u in enumerate(units):
        M[j, :len(u)] = u
    m.train()
    for _ in range(EPOCHS):
        perm = rng.permutation(len(units))
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            # K instances per bag: without replacement if n >= K, with replacement otherwise
            keys = rng.random((len(b), M.shape[1]))
            keys[M[b] < 0] = np.inf
            pos_wo = np.argsort(keys, 1)[:, :k]
            pos_w = (rng.random((len(b), k)) * L[b, None]).astype(np.int64)
            pos = np.where((L[b] >= k)[:, None], pos_wo, pos_w)
            idx = np.take_along_axis(M[b], pos, 1)
            assert (idx >= 0).all()
            x = Xt[torch.from_numpy(idx)]
            loss = F.cross_entropy(m(x, torch.ones(x.shape[:2], dtype=torch.bool)), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()


def eval_bags(m, X, bags, k=K_EVAL):
    idx = np.zeros((len(bags), k), dtype=np.int64)
    mask = np.zeros((len(bags), k), dtype=bool)
    for j, b in enumerate(bags):
        take = b if len(b) <= k else b[np.round(np.linspace(0, len(b) - 1, k)).astype(int)]
        idx[j, :len(take)], mask[j, :len(take)] = take, True
    with torch.no_grad():
        return m(torch.from_numpy(X)[torch.from_numpy(idx)], torch.from_numpy(mask)).argmax(1).numpy()


def img_to_bag_max(pred_img, bags):
    return np.array([pred_img[b].max() for b in bags])


def standardize(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return [(a - mu) / sd for a in (Xtr, *others)]


def fit_lr(Xtr, ytr, inner_val_mask):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    best = None
    for C in (0.01, 0.1, 1.0):
        lr = LogisticRegression(C=C, max_iter=3000).fit(Xtr[~inner_val_mask], ytr[~inner_val_mask])
        ll = log_loss(ytr[inner_val_mask], lr.predict_proba(Xtr[inner_val_mask]), labels=list(range(6)))
        if best is None or ll < best[0]:
            best = (ll, C)
    return LogisticRegression(C=best[1], max_iter=3000).fit(Xtr, ytr), best[1]


SIZE_BINS = [1, 2, 3, 4, 6, 11, 16, 21, 31, 10**6]


def size_onehot(n):
    b = np.digitize(n, SIZE_BINS[1:])
    return np.eye(len(SIZE_BINS) - 1)[b]


def run(enc):
    pool = load_pool()
    mem = np.load(CACHE / "membership.npz")
    bags = pd.read_csv(HERE / "bags.csv")
    pbag, dbag = bags[bags.fold >= 0].reset_index(drop=True), bags[bags.fold < 0].reset_index(drop=True)
    P_bags = bag_index(mem["pool_img_bag"], mem["pool_order"])
    D_bags = bag_index(mem["dev_img_bag"], mem["dev_order"])
    y_img = pool.label.values
    yb, ybd = pbag.label.values, dbag.label.values
    for j in (0, len(P_bags) - 1):
        assert y_img[P_bags[j]].max() == yb[j]
    fold_img, fold_bag = pool.fold.values, pbag.fold.values
    Xp_raw, Xd_raw = load_X(enc) if enc != "size" else (None, None)
    methods = (["size_only"] if enc == "size" else
               ["img_lr_max", "img_mlp_max", "meanpool", "maxpool", "attn"])
    seeds_of = {m: ((0,) if m in ("img_lr_max", "size_only") else SEEDS) for m in methods}
    oof = {(m, s): np.full(len(P_bags), -1) for m in methods for s in seeds_of[m]}
    dev = {(m, s): None for m in methods for s in seeds_of[m]}
    chosen_C, t0 = {}, time.time()
    for k in list(range(5)) + ["dev"]:
        tr_img = fold_img != k if k != "dev" else np.ones(len(pool), bool)
        tr_bag = np.where(fold_bag != k)[0] if k != "dev" else np.arange(len(P_bags))
        te_bags = [P_bags[j] for j in np.where(fold_bag == k)[0]] if k != "dev" else D_bags
        if enc == "size":
            from sklearn.linear_model import LogisticRegression
            n_tr = np.array([len(P_bags[j]) for j in tr_bag])
            n_te = np.array([len(b) for b in te_bags])
            lr = LogisticRegression(C=1.0, max_iter=3000).fit(size_onehot(n_tr), yb[tr_bag])
            pr = lr.predict(size_onehot(n_te))
            if k == "dev":
                dev[("size_only", 0)] = pr
            else:
                oof[("size_only", 0)][fold_bag == k] = pr
            continue
        # re-index: train images -> local rows; standardize on training images only
        tr_rows = np.where(tr_img)[0]
        Xtr, Xp_all, Xd_all = standardize(Xp_raw[tr_rows], Xp_raw, Xd_raw)
        X_te = Xd_all if k == "dev" else Xp_all
        # LR: inner patient-disjoint fold = next training fold
        inner = (fold_img[tr_rows] == ((k + 1) % 5 if k != "dev" else 0))
        lr, C = fit_lr(Xtr, y_img[tr_rows], inner)
        chosen_C[str(k)] = C
        p_img = lr.predict(X_te)
        bp = img_to_bag_max(p_img, te_bags)
        if k == "dev":
            dev[("img_lr_max", 0)] = bp
        else:
            oof[("img_lr_max", 0)][fold_bag == k] = bp
        img_units = [np.array([i]) for i in tr_rows]
        bag_units = [P_bags[j] for j in tr_bag]
        for s in SEEDS:
            mi = train_head(Xp_all, img_units, y_img[tr_rows], "none", s, bs=128)
            with torch.no_grad():
                p_img = mi(torch.from_numpy(X_te)[:, None], torch.ones(len(X_te), 1, dtype=torch.bool)).argmax(1).numpy()
            res = {"img_mlp_max": img_to_bag_max(p_img, te_bags)}
            for pl, name in (("mean", "meanpool"), ("max", "maxpool"), ("attn", "attn")):
                mb = train_head(Xp_all, bag_units, yb[tr_bag], pl, s, bs=32)
                res[name] = eval_bags(mb, X_te, te_bags)
                if name == "attn" and k == "dev" and s == 0:
                    torch.save(mb.state_dict(), CACHE / f"attn_dev_{enc}_s0.pt")
            for name, bp in res.items():
                if k == "dev":
                    dev[(name, s)] = bp
                else:
                    oof[(name, s)][fold_bag == k] = bp
        print(f"  {enc} fold {k} done  C={C}  {time.time() - t0:.0f}s", flush=True)
    for v in oof.values():
        assert (v >= 0).all()
    np.savez(CACHE / f"preds_{enc}.npz",
             **{f"oof|{m}|{s}": v for (m, s), v in oof.items()},
             **{f"dev|{m}|{s}": v for (m, s), v in dev.items()},
             chosen_C=np.array(list(chosen_C.items()), dtype=str) if chosen_C else np.array([]))


if __name__ == "__main__":
    for enc in (sys.argv[1:] or ["size", "rfg", "dino", "effnet", "rfg_clahe"]):
        run(enc)
