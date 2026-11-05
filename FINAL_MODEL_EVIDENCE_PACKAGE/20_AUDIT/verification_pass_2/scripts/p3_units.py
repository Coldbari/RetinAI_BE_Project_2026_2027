"""P3 - aggregation sensitivity, view stratification, clustered CIs, per-class, confusion."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,"/Users/praharsh/Project/RetinAI/FINAL_MODEL_EVIDENCE_PACKAGE/_build")
from common import *
from p2_evaluation import metrics, aggregate, qwk6, qwk5_map, cluster_boot, RULES
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

ARM,SEED="structured",42
out=[]
print("== aggregation sensitivity (both halves) ==")
for which in ["locked","dev"]:
    m=half(which); pl=preds(which,ARM,SEED)
    # image level, all views + per region
    for reg in ["ALL","temporal","nasal","posterior pole"]:
        sel=np.ones(len(m),bool) if reg=="ALL" else (m.region==reg).values
        acc=[];q6=[];q5=[];mf=[]
        for pk in pl:
            mm=metrics(pk["y"][sel],pk["pred"][sel])
            acc.append(mm["accuracy"]);q6.append(mm["qwk_6class"]);q5.append(mm["qwk_5class_map"]);mf.append(mm["macro_f1_present"])
        out.append(dict(dataset=f"multiview_{which}", unit="image", rule=f"view={reg}", n=int(sel.sum()),
                        qwk_6class=round(np.mean(q6),4), qwk_5class_map=round(np.mean(q5),4),
                        accuracy=round(np.mean(acc),4), macro_f1=round(np.mean(mf),4),
                        qwk_sd_across_ckpt=round(np.std(q6,ddof=1),4)))
    # eye level, every aggregation rule
    for rule in RULES:
        q6=[];q5=[];acc=[];mf=[];n=None
        for pk in pl:
            y,p=aggregate(m,pk["pred"],pk["prob"],rule); y=np.asarray(y);p=np.asarray(p); n=len(y)
            mm=metrics(y,p); q6.append(mm["qwk_6class"]);q5.append(mm["qwk_5class_map"])
            acc.append(mm["accuracy"]);mf.append(mm["macro_f1_present"])
        out.append(dict(dataset=f"multiview_{which}", unit="eye", rule=rule, n=n,
                        qwk_6class=round(np.mean(q6),4), qwk_5class_map=round(np.mean(q5),4),
                        accuracy=round(np.mean(acc),4), macro_f1=round(np.mean(mf),4),
                        qwk_sd_across_ckpt=round(np.std(q6,ddof=1),4)))
A=pd.DataFrame(out)
write(A,"08_EYE_LEVEL_ANALYSIS/aggregation_sensitivity.csv")
write(A[A.rule.str.startswith("view=")|A.rule.isin(["max","majority"])],
      "11_AGGREGATION_AND_VIEW_ANALYSIS/view_stratified_performance.csv")
print(A.to_string(index=False))

print("\n== clustered CIs (patient-resampled, B=2000) ==")
ci=[]
for which in ["locked","dev"]:
    m=half(which); pl=preds(which,ARM,SEED)
    for unit,eye in [("image",False),("eye",True)]:
        for name,fn in [("qwk_6class",qwk6),("qwk_5class_map",qwk5_map),
                        ("accuracy",lambda y,p: float((y==p).mean())),
                        ("macro_f1_present",lambda y,p: float(f1_score(y,p,average="macro",labels=sorted(set(y.tolist())),zero_division=0)))]:
            lo,hi,nb=cluster_boot(m,pl,fn,B=2000,eye=eye)
            pt=np.mean([fn(*( (pk["y"],pk["pred"]) if not eye else tuple(map(np.asarray,aggregate(m,pk["pred"],pk["prob"],"max"))))) for pk in pl])
            ci.append(dict(dataset=f"multiview_{which}", unit=unit,
                           aggregation=("none" if unit=="image" else "max over views"),
                           metric=name, point=round(float(pt),4), ci95_lo=round(lo,4), ci95_hi=round(hi,4),
                           ci_width=round(hi-lo,4), n_units=(len(m) if unit=="image" else m.groupby(['patient_id','eye']).ngroups),
                           n_clusters=m.patient_id.nunique(), valid_replicates=nb,
                           uncertainty="patient-clustered bootstrap, 2000 reps, percentile"))
            print("  ",ci[-1]["dataset"],unit,name,ci[-1]["point"],[ci[-1]["ci95_lo"],ci[-1]["ci95_hi"]])
C=pd.DataFrame(ci); write(C,"08_EYE_LEVEL_ANALYSIS/eye_level_clustered_ci.csv")
write(C,"16_STATISTICAL_ANALYSIS/confidence_intervals.csv")

print("\n== per-class + confusion, locked ==")
m=half("locked"); pl=preds("locked",ARM,SEED)
for unit in ["image","eye"]:
    P_,R_,F_,S_=[],[],[],None
    for pk in pl:
        if unit=="image": y,p=pk["y"],pk["pred"]
        else: y,p=map(np.asarray,aggregate(m,pk["pred"],pk["prob"],"max"))
        pr,rc,f1,su=precision_recall_fscore_support(y,p,labels=range(6),zero_division=0)
        P_.append(pr);R_.append(rc);F_.append(f1);S_=su
    df=pd.DataFrame(dict(klass=NICE, support=S_,
        precision=np.round(np.mean(P_,0),4), recall=np.round(np.mean(R_,0),4), f1=np.round(np.mean(F_,0),4),
        precision_sd_ckpt=np.round(np.std(P_,0,ddof=1),4), recall_sd_ckpt=np.round(np.std(R_,0,ddof=1),4)))
    df["unit"]=unit; df["dataset"]="multiview_locked"; df["aggregation"]=("none" if unit=="image" else "max over views")
    df["reportable"]=np.where(df.support>=30,"yes","NO - n<30, not externally validated")
    write(df,f"07_TEST_RESULTS/per_class_{unit}_locked.csv")
    print(df.to_string(index=False))
    # confusion
    Y=np.concatenate([ (pk["y"] if unit=="image" else np.asarray(aggregate(m,pk["pred"],pk["prob"],"max")[0])) for pk in pl])
    Pr=np.concatenate([ (pk["pred"] if unit=="image" else np.asarray(aggregate(m,pk["pred"],pk["prob"],"max")[1])) for pk in pl])
    cm=confusion_matrix(Y,Pr,labels=range(6))
    cd=pd.DataFrame(cm,index=[f"true_{n}" for n in NICE],columns=[f"pred_{n}" for n in NICE])
    cd.to_csv(PKG/f"07_TEST_RESULTS/confusion_counts_{unit}_locked.csv"); print("  wrote confusion_counts_%s_locked.csv"%unit)
    rn=pd.DataFrame(np.round(cm/cm.sum(1,keepdims=True),4),index=cd.index,columns=cd.columns)
    rn.to_csv(PKG/f"07_TEST_RESULTS/confusion_rownorm_{unit}_locked.csv")
    if unit=="image":
        print("   stage1->Normal %d/%d = %.4f | Normal correct %.4f"%(cm[1,0],cm[1].sum(),cm[1,0]/cm[1].sum(),cm[0,0]/cm[0].sum()))
    else:
        dis=(Y>=1)&(Y<=3); print("   stage1-3 EYES called Normal: %d/%d = %.4f"%((Pr[dis]==0).sum(),dis.sum(),(Pr[dis]==0).mean()))
        alld=(Y>=1); print("   ALL diseased EYES called Normal: %d/%d = %.4f"%((Pr[alld]==0).sum(),alld.sum(),(Pr[alld]==0).mean()))
