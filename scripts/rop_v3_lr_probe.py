"""Read the six peak-lr probe kernels and fix each arm's peak lr (protocol v2, "Peak-lr probe").

Rule (plan Task 8 step 6): per arm, keep the default 1e-4 unless another grid value's inner
macro-F1 beats it by MORE than 0.03 (the fold-0 seed SD); if several do, take the best.
Only summary.json is read. Any dev_predictions.npz the probes wrote is deleted unopened
(PROTOCOL_V2 Amendment 1 d), and the count is recorded.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE_DIR = ROOT / "kaggle_cache/v3_probe"
OUT = ROOT / "results/rop/v3/lr_probe.json"
ARCHS = ("retfound_green", "dinov2_s")
GRID = ("3e-5", "1e-4", "2e-4")
DEFAULT = "1e-4"
MARGIN = 0.03


def slug(arch: str, lr: str) -> str:
    return f"rop-v3-{arch.replace('_', '-')}-s42-probe-lr{lr}"


def read_summary(path: Path, arch: str, lr: str) -> dict:
    s = json.loads(path.read_text())
    want = {"arch": arch, "lr": lr, "probe": "inner", "seed": 42, "epochs": 20, "jobs_completed": 1}
    got = {k: s.get(k) for k in want}
    if got != want:
        raise ValueError(f"{path}: expected {want}, got {got}")
    inner = s["folds"]["inner"]
    return {"macro_f1": inner["macro_f1"], "qwk": inner["qwk"],
            "balanced_accuracy": inner["balanced_accuracy"], "gpu": s["gpu"],
            "n_inner": sum(c["support"] for c in inner["per_class"].values())}


def choose(f1: dict) -> str:
    """f1: lr -> inner macro-F1. Returns the chosen lr."""
    if set(f1) != set(GRID):
        raise ValueError(f"need all of {GRID}, got {sorted(f1)}")
    # rounded so a gain that prints as exactly 0.03 is not "more than" 0.03 by float noise
    better = {lr: v for lr, v in f1.items() if lr != DEFAULT and round(v - f1[DEFAULT], 6) > MARGIN}
    return max(better, key=better.get) if better else DEFAULT


def delete_dev_predictions(root: Path) -> int:
    hits = sorted(root.rglob("dev_predictions.npz"))
    for p in hits:
        p.unlink()
    return len(hits)


def main() -> None:
    # cumulative, so a rerun (which finds nothing left to delete) keeps the original record
    prev = json.loads(OUT.read_text()).get("dev_predictions_deleted_unopened", 0) if OUT.exists() else 0
    deleted = prev + delete_dev_predictions(PROBE_DIR)
    out = {"rule": f"keep {DEFAULT} unless another lr's inner macro-F1 beats it by > {MARGIN}; "
                   "if several do, the best", "grid": list(GRID), "arms": {},
           "dev_predictions_deleted_unopened": deleted}
    for arch in ARCHS:
        runs = {lr: read_summary(PROBE_DIR / slug(arch, lr) / "summary.json", arch, lr) for lr in GRID}
        chosen = choose({lr: r["macro_f1"] for lr, r in runs.items()})
        out["arms"][arch] = {"runs": runs, "chosen_lr": chosen}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({a: v["chosen_lr"] for a, v in out["arms"].items()}), f"deleted={deleted}")


if __name__ == "__main__":
    main()
