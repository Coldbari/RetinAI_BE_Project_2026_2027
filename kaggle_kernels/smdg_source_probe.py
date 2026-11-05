"""T11a — can we hold out a glaucoma SOURCE from data we already have?

The blocker on record is "no usable REFUGE source found", which assumed an external dataset
had to be sourced from outside. That assumption is probably wrong: SMDG-19 is not one
dataset, it is a META-dataset aggregating ~19 public glaucoma collections (REFUGE, ORIGA,
DRISHTI-GS, RIM-ONE, ACRIMA, PAPILA, BEH, ...), and its `names` column is understood to be
prefixed with the originating collection.

If that holds, the held-out-source experiment does not need new data at all:
  * drop one constituent collection entirely from training
  * retrain
  * evaluate zero-shot on the collection that was never seen

This kernel does NOT train anything. It only answers whether the experiment is possible, and
which collection is the best hold-out candidate. It is CPU-only and takes minutes.

A candidate is only usable if it has BOTH classes and enough positives to support an
interval — the T9b lesson: a source with 1 positive cluster measures nothing.
"""
import glob
import json
import os
import re
from collections import Counter

import pandas as pd

csvs = glob.glob("/kaggle/input/**/metadata*.csv", recursive=True)
assert csvs, "SMDG metadata.csv not found"
CSV = csvs[0]
print("metadata:", CSV, flush=True)

m = pd.read_csv(CSV)
print(f"\nrows: {len(m)}")
print("columns:", list(m.columns))

# The label column is `types` (-1 unlabelled / 0 non-glaucoma / 1 glaucoma) per configs/glaucoma.yaml.
label_col = "types" if "types" in m.columns else None
name_col = "names" if "names" in m.columns else m.columns[0]
print(f"\nusing name column '{name_col}', label column '{label_col}'")

print("\n--- first 10 names (is there a source prefix?) ---")
for n in m[name_col].astype(str).head(10):
    print("   ", n)

# Any column that already states the source outright would be better than parsing names.
for c in m.columns:
    if c in (name_col, label_col):
        continue
    u = m[c].dropna().unique()
    if 1 < len(u) <= 30 and m[c].dtype == object:
        print(f"\ncandidate source column '{c}': {sorted(map(str, u))[:30]}")


def prefix(n):
    """SMDG names look like '<COLLECTION>-<id>'; fall back to a leading alpha run."""
    s = str(n)
    if "-" in s:
        return s.split("-")[0]
    mm = re.match(r"^([A-Za-z][A-Za-z0-9]*?)(?=[_\d])", s)
    return mm.group(1) if mm else s


m["_src"] = m[name_col].map(prefix)
counts = Counter(m["_src"])
print(f"\n--- {len(counts)} distinct name prefixes ---")

rows = []
for src, n in counts.most_common():
    sub = m[m["_src"] == src]
    if label_col:
        lab = sub[sub[label_col] != -1]
        pos = int((lab[label_col] == 1).sum())
        neg = int((lab[label_col] == 0).sum())
    else:
        lab, pos, neg = sub, -1, -1
    rows.append({"source": src, "total": int(n), "labelled": int(len(lab)),
                 "glaucoma": pos, "non_glaucoma": neg})

df = pd.DataFrame(rows)
pd.set_option("display.width", 140)
print(df.to_string(index=False))

verdict = {"metadata_csv": CSV, "n_rows": int(len(m)), "n_prefixes": len(counts),
           "per_source": rows}

if len(counts) <= 1:
    print("\n" + "=" * 70)
    print("VERDICT: names carry NO source prefix — the hold-out-a-source experiment is NOT")
    print("possible this way. The recorded blocker stands and an external dataset is needed.")
    verdict["feasible"] = False
else:
    # Usable = both classes present, and enough positives that an interval means something.
    usable = [r for r in rows if r["glaucoma"] >= 30 and r["non_glaucoma"] >= 30]
    usable.sort(key=lambda r: -min(r["glaucoma"], r["non_glaucoma"]))
    print("\n" + "=" * 70)
    print(f"VERDICT: {len(counts)} constituent collections found. "
          f"{len(usable)} carry >=30 of BOTH classes.")
    print("\nBest hold-out candidates (balanced enough to support an interval):")
    for r in usable[:8]:
        print(f"   {r['source']:<18} labelled {r['labelled']:>5}   "
              f"glaucoma {r['glaucoma']:>4}   non {r['non_glaucoma']:>5}")
    print("\nHolding one of these out of TRAINING and scoring it zero-shot is a genuine")
    print("held-out-source experiment using data already in hand.")
    verdict["feasible"] = True
    verdict["candidates"] = usable[:8]

json.dump(verdict, open("/kaggle/working/smdg_source_probe.json", "w"), indent=2, default=str)
print("\nDONE", flush=True)
