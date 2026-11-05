"""P2 - evaluation core: QWK definitions, locked/dev image+eye+view, aggregation, clustered CIs."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/Users/praharsh/Project/RetinAI/FINAL_MODEL_EVIDENCE_PACKAGE/_build")
from common import *
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support, confusion_matrix

# ---------- QWK variants -------------------------------------------------------------
def qwk6(y,p):                      # what the project currently computes (AP-ROP = ordinal rank 5)
    return float(cohen_kappa_score(y,p,weights="quadratic"))
def qwk5_drop(y,p):                 # documented 5-level scale; AP-ROP rows/preds removed
    k=(y!=5)&(p!=5)
    if k.sum()<2 or len(set(y[k].tolist()))<2: return np.nan
    return float(cohen_kappa_score(y[k],p[k],weights="quadratic"))
def qwk5_map(y,p):                  # documented 5-level scale; AP-ROP folded onto Stage 4/5
    y2=np.where(y==5,4,y); p2=np.where(p==5,4,p)
    if len(set(y2.tolist()))<2: return np.nan
    return float(cohen_kappa_score(y2,p2,weights="quadratic"))
def kappa_unw(y,p): return float(cohen_kappa_score(y,p))

def metrics(y,p):
    present=sorted(set(y.tolist()))
    k=(y!=5)&(p!=5)
    return dict(n=len(y), qwk_6class=qwk6(y,p), qwk_5class_drop=qwk5_drop(y,p),
                qwk_5class_map=qwk5_map(y,p), n_dropped_5class=int((~k).sum()),
                kappa_unweighted=kappa_unw(y,p), accuracy=float((y==p).mean()),
                macro_f1_present=float(f1_score(y,p,average="macro",labels=present,zero_division=0)),
                macro_f1_6label=float(f1_score(y,p,average="macro",labels=list(range(6)),zero_division=0)),
                weighted_f1=float(f1_score(y,p,average="weighted",zero_division=0)))

# ---------- aggregation rules --------------------------------------------------------
def aggregate(m, pred, prob, rule):
    """Returns (y_eye, p_eye) for one eye-aggregation rule. Label side is always max."""
    d=m.assign(pred=pred)
    for c in range(6): d[f"p{c}"]=prob[:,c]
    g=d.groupby(["patient_id","eye"])
    y=g.label.max()
    if   rule=="max":            p=g.pred.max()
    elif rule=="mean_prob":      p=g[[f"p{c}" for c in range(6)]].mean().values.argmax(1); return y.values,p
    elif rule=="mean_class":     p=g.pred.mean().round().astype(int)
    elif rule=="median":         p=g.pred.median().round().astype(int)
    elif rule=="majority":       p=g.pred.agg(lambda s: s.value_counts().index[0])
    elif rule in ("temporal","nasal","posterior pole"):
        d2=d[d.region==rule]; g2=d2.groupby(["patient_id","eye"])
        return g2.label.max().values, g2.pred.max().values
    return y.values, p.values

RULES=["max","mean_prob","mean_class","median","majority","temporal","nasal","posterior pole"]

# ---------- clustered bootstrap ------------------------------------------------------
def cluster_boot(m, preds_list, fn, B=2000, seed=42, eye=False, rule="max"):
    """Resample patient_id; re-tag redrawn patients so duplicate eyes do not collapse."""
    rng=np.random.default_rng(seed); pats=m.patient_id.unique()
    idx={p:np.where(m.patient_id.values==p)[0] for p in pats}
    out=[]
    for _ in range(B):
        draw=rng.choice(len(pats),len(pats),replace=True)
        rows=np.concatenate([idx[pats[i]] for i in draw])
        tag=np.concatenate([np.full(len(idx[pats[i]]),j) for j,i in enumerate(draw)])
        vals=[]
        for pk in preds_list:
            if eye:
                sub=m.iloc[rows].assign(pred=pk["pred"][rows], _k=tag)
                gg=sub.groupby(["_k","eye"]); yy=gg.label.max().values; pp=gg.pred.max().values
            else:
                yy=m.label.values[rows]; pp=pk["pred"][rows]
            if len(set(yy.tolist()))<2: continue
            v=fn(yy,pp)
            if not np.isnan(v): vals.append(v)
        if vals: out.append(np.mean(vals))
    a=np.array(out)
    return float(np.percentile(a,2.5)), float(np.percentile(a,97.5)), len(a)

# ====================================================================================
print("== QWK recalculation ==")
rows=[]
for which in ["locked","dev"]:
    m=half(which)
    for arm in ["structured","flat"]:
        for seed in [42,1337]:
            pl=preds(which,arm,seed)
            for lvl,agg in [("image",None),("eye","max")]:
                vals={k:[] for k in ["qwk_6class","qwk_5class_drop","qwk_5class_map","accuracy","macro_f1_present","kappa_unweighted","weighted_f1","macro_f1_6label"]}
                nd=[]
                for pk in pl:
                    if lvl=="image": y,p=pk["y"],pk["pred"]
                    else:            y,p=aggregate(m,pk["pred"],pk["prob"],"max")
                    mm=metrics(np.asarray(y),np.asarray(p))
                    for k in vals: vals[k].append(mm[k])
                    nd.append(mm["n_dropped_5class"])
                rows.append(dict(dataset=f"multiview_{which}", arm=arm, seed=seed, unit=lvl,
                                 aggregation=("none" if lvl=="image" else "max over views"),
                                 n=(len(m) if lvl=="image" else m.groupby(['patient_id','eye']).ngroups),
                                 n_dropped_5class=int(np.mean(nd)),
                                 **{k:round(float(np.mean(v)),4) for k,v in vals.items()},
                                 qwk_6class_sd=round(float(np.std(vals["qwk_6class"],ddof=1)),4)))
# CV folds, both arms, both seeds
for arm in ["structured","flat"]:
    for seed in [42,1337]:
        try: o=oof(arm,seed)
        except Exception as e: print("  skip",arm,seed,e); continue
        for f,g in o.groupby("fold"):
            mm=metrics(g.label.values,g.pred.values)
            rows.append(dict(dataset=f"cv_fold_{f}", arm=arm, seed=seed, unit="image", aggregation="none",
                             n=mm["n"], n_dropped_5class=mm["n_dropped_5class"],
                             **{k:round(mm[k],4) for k in ["qwk_6class","qwk_5class_drop","qwk_5class_map","accuracy","macro_f1_present","kappa_unweighted","weighted_f1","macro_f1_6label"]},
                             qwk_6class_sd=np.nan))
        mm=metrics(o.label.values,o.pred.values)
        rows.append(dict(dataset="cv_pooled_oof", arm=arm, seed=seed, unit="image", aggregation="none",
                         n=mm["n"], n_dropped_5class=mm["n_dropped_5class"],
                         **{k:round(mm[k],4) for k in ["qwk_6class","qwk_5class_drop","qwk_5class_map","accuracy","macro_f1_present","kappa_unweighted","weighted_f1","macro_f1_6label"]},
                         qwk_6class_sd=np.nan))
q=pd.DataFrame(rows)
q["qwk_delta_5map_minus_6"]=(q.qwk_5class_map-q.qwk_6class).round(4)
q["qwk_delta_5drop_minus_6"]=(q.qwk_5class_drop-q.qwk_6class).round(4)
write(q,"06_VALIDATION/qwk_5class_vs_6class_recalculation.csv")
write(q[q.dataset.str.startswith("cv")],"06_VALIDATION/cv_results_corrected.csv")
print(q[(q.arm=="structured")&(q.seed==42)][["dataset","unit","n","qwk_6class","qwk_5class_map","qwk_5class_drop","accuracy"]].to_string(index=False))
