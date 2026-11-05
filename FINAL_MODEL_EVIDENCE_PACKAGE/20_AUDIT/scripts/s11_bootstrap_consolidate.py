"""16_STATISTICAL_ANALYSIS/bootstrap_results.csv — every clustered bootstrap in one table."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
sys.path.insert(0, str(ROOT))
rows = []
for f, unit in (("07_TEST_RESULTS/statistical_uncertainty/uncertainty_image_level.csv", "image"),
                ("08_EYE_LEVEL_ANALYSIS/eye_confidence_intervals.csv", "eye")):
    d = pd.read_csv(PKG / f)
    for _, r in d.iterrows():
        rows.append(dict(scope="internal 5-fold OOF" if unit == "image" else "external locked",
                         evaluation_unit=unit, metric=r.metric, point=r.point,
                         ci95_lo=r.ci95_lo, ci95_hi=r.ci95_hi, ci_width=r.ci95_hi - r.ci95_lo,
                         n_units=r.get("n_images", r.get("n_eyes")),
                         n_clusters_resampled=r.get("n_patients"),
                         replicates=2000, rng_seed=42, cluster="infant"))
pc = pd.read_csv(PKG / "07_TEST_RESULTS/per_class_metrics/per_class_metrics.csv")
for _, r in pc.iterrows():
    for m, lo, hi in (("recall", "recall_lo", "recall_hi"), ("precision", "precision_lo", "precision_hi")):
        rows.append(dict(scope="internal 5-fold OOF", evaluation_unit="image",
                         metric=f"{m} [{r.cls}]", point=r[m], ci95_lo=r[lo], ci95_hi=r[hi],
                         ci_width=r[hi] - r[lo], n_units=int(r.support), n_clusters_resampled=1528,
                         replicates=1000, rng_seed=42, cluster="infant"))
b = pd.DataFrame(rows)
b["interpretation"] = np.where(b.ci_width > 0.30, "VERY WIDE — small or clustered support, do not headline",
                       np.where(b.ci_width > 0.15, "wide", "usable"))
save_csv(b, PKG / "16_STATISTICAL_ANALYSIS/bootstrap_results.csv")
print(b.to_string(index=False))
