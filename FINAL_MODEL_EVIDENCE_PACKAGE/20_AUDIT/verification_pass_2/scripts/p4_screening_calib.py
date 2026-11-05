"""P4 - screening endpoint at 3 units, calibration (structured vs flat), site standardisation, errors."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,"/Users/praharsh/Project/RetinAI/FINAL_MODEL_EVIDENCE_PACKAGE/_build")
from common import *
from sklearn.metrics import roc_auc_score

REB=json.loads((R/"screening_rebase.json").read_text()); THR=float(REB["threshold"])
print("deployed screening threshold (from results/rop/screening_rebase.json):", THR)

# ============ 1. SCREENING at image / eye / infant ===================================
def screen_stats(y,flag):
    tp=int(((y==1)&(flag==1)).sum()); fn=int(((y==1)&(flag==0)).sum())
    fp=int(((y==0)&(flag==1)).sum()); tn=int(((y==0)&(flag==0)).sum())
    return dict(n=len(y), pos=tp+fn, neg=fp+tn, TP=tp,FN=fn,FP=fp,TN=tn,
        sensitivity=round(tp/(tp+fn),4) if tp+fn else np.nan,
        specificity=round(tn/(tn+fp),4) if tn+fp else np.nan,
        PPV=round(tp/(tp+fp),4) if tp+fp else np.nan,
        NPV=round(tn/(tn+fn),4) if tn+fn else np.nan,
        prevalence=round((tp+fn)/len(y),4))

rows=[]; boot_rows=[]
for which in ["locked","dev"]:
    m=half(which); pl=preds(which,"structured",42)
    scores={"served ckpt0 (fold-0 = deployed weights)":[pl[0]],
            "5-checkpoint mean score":pl}
    for tag,pk_list in scores.items():
        S=np.mean([1-pk["prob"][:,0] for pk in pk_list],axis=0)
        d=m.assign(score=S, y=(m.label>0).astype(int), flag=(S>=THR).astype(int))
        for unit,key in [("image",None),("eye",["patient_id","eye"]),("infant",["patient_id"])]:
            if key is None: y,f=d.y.values,d.flag.values; agg="none (one frame = one decision)"
            else:
                g=d.groupby(key).agg(y=("y","max"),flag=("flag","max")); y,f=g.y.values,g.flag.values
                agg="any frame flags the unit (max)"
            st=screen_stats(y,f)
            rows.append(dict(dataset=f"multiview_{which}", score_stream=tag, unit=unit, aggregation=agg,
                             threshold=THR, auc=round(roc_auc_score(d.y,d.score),4) if unit=="image" else np.nan, **st))
            print(f"  {which:6s} {tag[:22]:22s} {unit:6s} n={st['n']:4d} sens={st['sensitivity']} spec={st['specificity']} PPV={st['PPV']} NPV={st['NPV']}")
S=pd.DataFrame(rows); write(S,"07_TEST_RESULTS/screening_image_eye_infant.csv")
write(S[(S.dataset=="multiview_locked")],"07_TEST_RESULTS/locked_screening_results.csv")

# clustered CIs for screening (resample infants)
print("\n== screening clustered CIs (locked, served ckpt0) ==")
m=half("locked"); pl=preds("locked","structured",42)
ci=[]
for tag,pk_list in [("served ckpt0",[pl[0]]),("5-checkpoint mean",pl)]:
    Sc=np.mean([1-pk["prob"][:,0] for pk in pk_list],axis=0)
    d=m.assign(score=Sc,y=(m.label>0).astype(int),flag=(Sc>=THR).astype(int))
    pats=d.patient_id.unique(); idx={p:np.where(d.patient_id.values==p)[0] for p in pats}
    rng=np.random.default_rng(42)
    acc={u:{"sens":[],"spec":[]} for u in ["image","eye","infant"]}
    for _ in range(2000):
        draw=rng.choice(len(pats),len(pats),replace=True)
        rr=np.concatenate([idx[pats[i]] for i in draw])
        tag2=np.concatenate([np.full(len(idx[pats[i]]),j) for j,i in enumerate(draw)])
        sub=d.iloc[rr].assign(_k=tag2)
        for u,key in [("image",None),("eye",["_k","eye"]),("infant",["_k"])]:
            if key is None: y,f=sub.y.values,sub.flag.values
            else:
                g=sub.groupby(key).agg(y=("y","max"),flag=("flag","max")); y,f=g.y.values,g.flag.values
            if (y==1).sum(): acc[u]["sens"].append(f[y==1].mean())
            if (y==0).sum(): acc[u]["spec"].append(1-f[y==0].mean())
    for u in ["image","eye","infant"]:
        for k in ["sens","spec"]:
            a=np.array(acc[u][k])
            ci.append(dict(dataset="multiview_locked",score_stream=tag,unit=u,metric=k,
                           ci95_lo=round(float(np.percentile(a,2.5)),4),ci95_hi=round(float(np.percentile(a,97.5)),4),
                           replicates=len(a),method="patient-clustered bootstrap, percentile"))
            print(f"  {tag[:18]:18s} {u:6s} {k}: [{ci[-1]['ci95_lo']}, {ci[-1]['ci95_hi']}]")
write(pd.DataFrame(ci),"07_TEST_RESULTS/screening_confidence_intervals.csv")

# ============ 2. CALIBRATION: structured vs flat, OOF ================================
print("\n== calibration (pooled OOF) ==")
cal=[]
for arm in ["structured","flat"]:
    o=oof(arm,42); corr=(o.label==o.pred).values; conf=o.conf.values
    cal.append(dict(model=arm, dataset="cv_pooled_oof", unit="image", n=len(o), n_clusters=o.patient_id.nunique(),
                    accuracy=round(corr.mean(),4), mean_confidence=round(conf.mean(),4),
                    gap_conf_minus_acc=round(conf.mean()-corr.mean(),4), ece_10bin=round(ece(conf,corr),4),
                    brier_top1=round(float(np.mean((corr.astype(float)-conf)**2)),4),
                    mean_conf_correct=round(conf[corr].mean(),4), mean_conf_wrong=round(conf[~corr].mean(),4),
                    separation=round(conf[corr].mean()-conf[~corr].mean(),4),
                    frac_conf_gt_090=round(float((conf>0.90).mean()),4),
                    n_conf_gt_090=int((conf>0.90).sum()), n_wrong_conf_gt_090=int(((conf>0.90)&~corr).sum()),
                    err_rate_when_conf_gt_090=round(float(((conf>0.90)&~corr).sum()/max((conf>0.90).sum(),1)),4)))
    print("  ",cal[-1])
    # patient-weighted ECE
    pw=o.groupby("patient_id").apply(lambda g: pd.Series({"acc":(g.label==g.pred).mean(),"conf":g.conf.mean()}),include_groups=False)
    cal[-1]["patient_weighted_accuracy"]=round(float(pw.acc.mean()),4)
    cal[-1]["patient_weighted_confidence"]=round(float(pw.conf.mean()),4)
    cal[-1]["patient_weighted_gap"]=round(float(pw.conf.mean()-pw.acc.mean()),4)
CAL=pd.DataFrame(cal); write(CAL,"10_CONFIDENCE_CALIBRATION/calibration_summary.csv")

# per-class calibration, structured
o=oof("structured",42); corr=(o.label==o.pred).values
pc=[]
for c in range(6):
    s=(o.label==c).values
    if s.sum()==0: continue
    pc.append(dict(klass=NICE[c],n=int(s.sum()),accuracy=round(corr[s].mean(),4),
                   mean_confidence=round(o.conf.values[s].mean(),4),
                   gap=round(o.conf.values[s].mean()-corr[s].mean(),4),
                   ece=round(ece(o.conf.values[s],corr[s]),4)))
write(pd.DataFrame(pc),"10_CONFIDENCE_CALIBRATION/per_class_calibration.csv")

# ============ 3. SITE CALIBRATION with direct standardisation ========================
print("\n== site calibration, raw vs directly standardised ==")
common=[c for c in range(6) if all((o[(o.source==s)].label==c).sum()>=30 for s in o.source.unique())]
w=o[o.label.isin(common)].label.value_counts(normalize=True)
def std_gap(d):
    t=0
    for c in common:
        g=d[d.label==c]
        if len(g)==0: return np.nan
        t+=w[c]*(g.conf.mean()-(g.label==g.pred).mean())
    return t
rng=np.random.default_rng(42); site=[]
sub=o[o.label.isin(common)]
for s,g in o.groupby("source"):
    gs=sub[sub.source==s]
    idx={p:v.index.values for p,v in gs.groupby("patient_id")}
    pats=list(idx); vals=[]
    for _ in range(2000):
        dr=rng.choice(len(pats),len(pats),replace=True)
        rr=np.concatenate([idx[pats[i]] for i in dr])
        v=std_gap(sub.loc[rr])
        if not np.isnan(v): vals.append(v)
    a=np.array(vals)
    site.append(dict(source=s, n_images=len(g), n_patient_ids=g.patient_id.nunique(),
        accuracy=round((g.label==g.pred).mean(),4), mean_confidence=round(g.conf.mean(),4),
        raw_gap=round(g.conf.mean()-(g.label==g.pred).mean(),4), raw_ece=round(ece(g.conf.values,(g.label==g.pred).values),4),
        normal_only_ece=round(ece(g[g.label==0].conf.values,(g[g.label==0].label==g[g.label==0].pred).values),4),
        standardised_gap=round(float(std_gap(gs)),4),
        std_gap_ci95_lo=round(float(np.percentile(a,2.5)),4), std_gap_ci95_hi=round(float(np.percentile(a,97.5)),4),
        valid_replicates=len(a), classes_standardised="|".join(NICE[c] for c in common)))
    print("  ",site[-1]["source"],"raw",site[-1]["raw_gap"],"normal-only ECE",site[-1]["normal_only_ece"],
          "STANDARDISED",site[-1]["standardised_gap"],[site[-1]["std_gap_ci95_lo"],site[-1]["std_gap_ci95_hi"]])
SITE=pd.DataFrame(site); write(SITE,"10_CONFIDENCE_CALIBRATION/site_calibration_corrected.csv")

# per-class gap by site (the table that explains the correction)
g2=[]
for s,g in o.groupby("source"):
    for c in range(6):
        gg=g[g.label==c]
        if len(gg)==0: continue
        g2.append(dict(source=s,klass=NICE[c],n=len(gg),accuracy=round((gg.label==gg.pred).mean(),4),
                       mean_confidence=round(gg.conf.mean(),4),gap=round(gg.conf.mean()-(gg.label==gg.pred).mean(),4)))
write(pd.DataFrame(g2),"10_CONFIDENCE_CALIBRATION/per_class_gap_by_site.csv")

# ============ 4. CURRENT-MODEL ERROR ANALYSIS ========================================
print("\n== current-model error analysis (locked, structured s42) ==")
m=half("locked"); pl=preds("locked","structured",42)
recs=[]
for k,pk in enumerate(pl):
    for i in range(len(m)):
        recs.append(dict(ckpt=k, image=m.Image_name.values[i], patient_id=m.patient_id.values[i],
                         eye=m.eye.values[i], region=m.region.values[i],
                         true=NICE[pk["y"][i]], pred=NICE[pk["pred"][i]],
                         correct=bool(pk["y"][i]==pk["pred"][i]), confidence=float(pk["prob"][i].max())))
E=pd.DataFrame(recs)
summ=E.groupby(["true","pred"]).size().reset_index(name="count").sort_values("count",ascending=False)
summ["pct_of_true_class"]=summ.apply(lambda r: round(r["count"]/E[E.true==r["true"]].shape[0],4),axis=1)
write(summ[summ.true!=summ.pred].head(25),"09_ERROR_ANALYSIS/most_confused_pairs.csv")
per=E.groupby("true").agg(predictions=("correct","size"),errors=("correct",lambda s:(~s).sum()),
                          error_rate=("correct",lambda s: round(1-s.mean(),4)),
                          mean_conf_when_wrong=("confidence",lambda s: np.nan)).reset_index()
per["mean_conf_when_wrong"]=[round(E[(E.true==t)&(~E.correct)].confidence.mean(),4) for t in per.true]
per["mean_conf_when_right"]=[round(E[(E.true==t)&(E.correct)].confidence.mean(),4) for t in per.true]
write(per,"09_ERROR_ANALYSIS/current_model_error_summary.csv")
print(per.to_string(index=False))
hc=E[(~E.correct)&(E.confidence>0.90)]
write(hc.groupby(["true","pred"]).agg(n=("image","size"),mean_conf=("confidence","mean"),
       max_conf=("confidence","max")).reset_index().sort_values("n",ascending=False),
      "09_ERROR_ANALYSIS/high_confidence_errors.csv")
print(f"  high-confidence (>0.90) errors: {len(hc)} of {int((E.confidence>0.90).sum())} confident predictions "
      f"({len(E)} total); error rate when confident = {len(hc)/max(int((E.confidence>0.90).sum()),1):.4f}")
# by view
bv=E[~E.correct].groupby("region").size().reset_index(name="errors")
bv["predictions"]=[int((E.region==r).sum()) for r in bv.region]
bv["error_rate"]=(bv.errors/bv.predictions).round(4)
bv["called_normal_when_stage123"]=[round(float(((E.region==r)&(E.true.isin(["Stage 1","Stage 2","Stage 3"]))&(E.pred=="Normal")).sum()/
                                    max(((E.region==r)&(E.true.isin(["Stage 1","Stage 2","Stage 3"]))).sum(),1)),4) for r in bv.region]
write(bv,"09_ERROR_ANALYSIS/errors_by_view.csv"); print(bv.to_string(index=False))
E.to_csv(PKG/"_build/locked_predictions_long.csv",index=False)
