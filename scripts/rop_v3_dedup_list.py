"""Write the exact-duplicate exclusion list for the 6,700-image CV pool.

The audit (exact_duplicate_members.csv) lists 10 groups / 20 images dataset-wide. Only
7 groups / 14 images lie in the CV pool (union of fold_*.csv val rows); those are excluded.
The 3 groups wholly inside one held-out half are NOT excluded (they cross no split, and
dropping them would change paper 1's 661-image / 230-eye counts); protocol v2 discloses them:
  052b2d068f1ada2fe84c9f8917bf0e9f  both multiview_dev
  7a1684d9c7b8582b87cc1294c1e081c7  both multiview_locked
  82fe7c001b9db49aeee9d8ebdfb2aac2  both multiview_locked
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MEMBERS = ROOT / "FINAL_MODEL_EVIDENCE_PACKAGE/20_AUDIT/verification_pass_2/03_DATA_LEAKAGE/exact_duplicate_members.csv"
OUT = ROOT / "results/prepared/rop_staging/exclude_duplicates.csv"


def main() -> None:
    pool = pd.concat([pd.read_csv(ROOT / f"results/prepared/rop_staging/fold_{k}.csv").query("split == 'val'")
                      for k in range(5)])
    assert len(pool) == 6700
    m = pd.read_csv(MEMBERS)
    m["image_path"] = m.image_path.str.replace(r"^.*?(data/rop/)", r"\1", regex=True)
    m = m[m.image_path.isin(set(pool.image_path))]
    m = m[m.groupby("md5").md5.transform("size") >= 2].sort_values(["md5", "image_path"])
    m["group"] = m.md5.rank(method="dense").astype(int) - 1
    out = m[["image_path", "md5", "group"]]
    assert len(out) == 14 and out.group.nunique() == 7, (len(out), out.group.nunique())
    out.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
