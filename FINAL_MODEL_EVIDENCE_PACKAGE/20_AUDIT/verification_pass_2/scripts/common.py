"""Shared loaders for the evidence package. Every number in the package comes from here."""
import numpy as np, pandas as pd, json
from pathlib import Path
ROOT = Path("/Users/praharsh/Project/RetinAI")
PKG  = ROOT / "FINAL_MODEL_EVIDENCE_PACKAGE"
P    = ROOT / "results/prepared/rop_staging"
R    = ROOT / "results/rop"
CL   = ["normal","stage_1","stage_2","stage_3","stage_4_5","arop"]
NICE = ["Normal","Stage 1","Stage 2","Stage 3","Stage 4/5","AP-ROP"]
VIEWCOLS = ["Left temporal","Left posterior pole","Left nasal",
            "Right nasal","Right posterior pole","Right temporal"]

def view_table():
    av = pd.read_excel(ROOT/"data/rop/multiview/ROP-DATASET/Dataset Acquisition View Table.xlsx")
    av["view"] = av[VIEWCOLS].notna().idxmax(axis=1)
    av["eye"]  = av["view"].str.split().str[0]
    av["region"] = av["view"].str.split(n=1).str[1]
    return av[["Image_name","eye","region"]]

def half(which):
    """multiview dev or locked, joined to eye/region. Row order == npz row order."""
    df = pd.read_csv(P/f"test_{which}.csv")
    df["Image_name"] = df.image_path.str.rsplit("/",n=1).str[1].str.replace(".jpg","",regex=False)
    m = df.merge(view_table(), on="Image_name", how="left")
    assert m.eye.notna().all() and len(m)==len(df), "view join incomplete"
    return m

def preds(which, arm="structured", seed=42):
    d = R/("locked_eval" if which=="locked" else "dev_eval")
    out=[]
    for k in range(5):
        z=np.load(d/f"{arm}_s{seed}_ckpt{k}.npz")
        out.append({"y":z["y_true"],"pred":z["y_pred"],"prob":z["prob6"]})
    return out

def oof(arm="structured", seed=42):
    """Pooled out-of-fold CV predictions with provenance columns. n=6700."""
    tag = {"structured":"structured_5fold","flat":"flat_natural_5fold"}[arm]
    if seed!=42: tag += "_s1337"
    rows=[]
    for i in range(5):
        f=pd.read_csv(P/f"fold_{i}.csv"); v=f[f.split=="val"].reset_index(drop=True)
        z=np.load(R/f"staging_runs/{tag}/fold_{i}.npz")
        pk = "prob6" if "prob6" in z.files else "prob"
        assert len(v)==len(z["y_true"]) and (v.label.values==z["y_true"]).all(), f"{tag} fold {i} misaligned"
        v=v.assign(pred=z["y_pred"], conf=z[pk].max(1), fold=i)
        for c in range(6): v[f"p{c}"]=z[pk][:,c]
        rows.append(v)
    return pd.concat(rows, ignore_index=True)

def ece(conf, correct, nb=10):
    bins=np.linspace(0,1,nb+1); e=0.0; n=len(conf)
    for lo,hi in zip(bins[:-1],bins[1:]):
        s=(conf>lo)&(conf<=hi)
        if s.sum()==0: continue
        e+=s.sum()/n*abs(correct[s].mean()-conf[s].mean())
    return float(e)

def write(df, relpath, **kw):
    p=PKG/relpath; p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, **kw); print("  wrote", relpath, f"({len(df)} rows)")
    return p
