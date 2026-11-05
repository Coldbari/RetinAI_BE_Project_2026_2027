"""P6 - the two figures pass-2 adds. Nothing here duplicates an existing package figure."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0,"/Users/praharsh/Project/RetinAI/FINAL_MODEL_EVIDENCE_PACKAGE/_build")
from common import PKG
V=PKG/"20_AUDIT/verification_pass_2"
plt.rcParams.update({"figure.dpi":180,"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
BLUE,ORANGE,GREY,RED="#3987e5","#d95926","#8a8a86","#c0392b"

# ---- FIG A: screening at three units, served checkpoint, locked half ----------------
S=pd.read_csv(V/"07_TEST_RESULTS/screening_image_eye_infant.csv")
C=pd.read_csv(V/"07_TEST_RESULTS/screening_confidence_intervals.csv")
s=S[(S.dataset=="multiview_locked")&(S.score_stream.str.startswith("served"))]
c=C[(C.score_stream=="served ckpt0")]
units=["image","eye","infant"]
fig,ax=plt.subplots(1,2,figsize=(9.6,3.6))
for k,(metric,title) in enumerate([("sensitivity","Sensitivity — catching disease"),
                                   ("specificity","Specificity — not flagging the healthy")]):
    v=[float(s[s.unit==u][metric].iloc[0]) for u in units]
    lo=[float(c[(c.unit==u)&(c.metric==("sens" if metric=="sensitivity" else "spec"))].ci95_lo.iloc[0]) for u in units]
    hi=[float(c[(c.unit==u)&(c.metric==("sens" if metric=="sensitivity" else "spec"))].ci95_hi.iloc[0]) for u in units]
    col=[BLUE if metric=="sensitivity" else (RED if x<0.15 else ORANGE) for x in v]
    ax[k].bar(range(3),v,color=col,width=.62)
    ax[k].errorbar(range(3),v,yerr=[np.array(v)-np.array(lo),np.array(hi)-np.array(v)],
                   fmt="none",ecolor="#222",capsize=4,lw=1.3)
    for i,x in enumerate(v):
        ax[k].text(i,max(x,0)+.045,f"{x:.3f}",ha="center",fontweight="bold")
    labs=[]
    for u,t in zip(units,["per image\n(1 frame)","per eye\n(any frame flags)","per infant\n(any frame flags)"]):
        r=s[s.unit==u].iloc[0]; labs.append(f"{t}\nn={int(r.n)}  ({int(r.pos)}+ / {int(r.neg)}−)")
    ax[k].set_xticks(range(3)); ax[k].set_xticklabels(labs,fontsize=7.8)
    ax[k].set_ylim(0,1.12); ax[k].set_title(title,fontsize=10,fontweight="bold"); ax[k].grid(axis="y",alpha=.18)
ax[1].text(2,.30,"every healthy infant\nis flagged",ha="center",color=RED,fontsize=8.6,fontweight="bold")
fig.suptitle("Screening endpoint, served checkpoint (fold-0), locked test — threshold 0.0155",
             fontsize=11,fontweight="bold",y=1.0)
fig.text(.5,-.06,"Locked half of Multi-View: 661 images / 230 eyes / 116 infants, never used in development. "
    "Bars are the served single-forward-pass score; whiskers are patient-clustered bootstrap 95% CIs (2,000 reps). "
    "Sensitivity rises with aggregation because any one flagged frame convicts the unit — and so does the false-positive rate.",
    ha="center",fontsize=7.6,color="#444",wrap=True)
fig.tight_layout(); fig.savefig(PKG/"07_TEST_RESULTS/screening_performance.png",bbox_inches="tight"); print("wrote screening_performance.png")

# ---- FIG B: site calibration, three adjustments ------------------------------------
T=pd.read_csv(V/"10_CONFIDENCE_CALIBRATION/site_calibration_corrected.csv")
G=pd.read_csv(V/"10_CONFIDENCE_CALIBRATION/per_class_gap_by_site.csv")
fig,ax=plt.subplots(1,2,figsize=(10.2,3.8),gridspec_kw={"width_ratios":[1,1.25]})
x=np.arange(len(T)); w=.26
ax[0].bar(x-w,T.raw_gap,w,label="raw gap (conf − acc)",color=GREY)
ax[0].bar(x,T.normal_only_ece,w,label="Normal-only 'control' (committed)",color=BLUE)
ax[0].bar(x+w,T.standardised_gap,w,label="case-mix standardised (correct)",color=ORANGE)
ax[0].errorbar(x+w,T.standardised_gap,yerr=[T.standardised_gap-T.std_gap_ci95_lo,T.std_gap_ci95_hi-T.standardised_gap],
               fmt="none",ecolor="#222",capsize=3,lw=1.2)
for i,v in enumerate(T.standardised_gap): ax[0].text(i+w,v+.008,f"{v:+.3f}",ha="center",fontsize=7.6,fontweight="bold")
ax[0].axhline(0,color="#222",lw=.8); ax[0].set_xticks(x); ax[0].set_xticklabels(T.source)
ax[0].set_ylabel("over-confidence (conf − accuracy)"); ax[0].legend(fontsize=7.2,frameon=False,loc="upper right")
ax[0].set_title("The control chose the easiest class",fontsize=10,fontweight="bold"); ax[0].grid(axis="y",alpha=.18)
piv=G.pivot(index="klass",columns="source",values="gap").reindex(["Normal","Stage 1","Stage 2","Stage 3","Stage 4/5","AP-ROP"])
piv.plot(kind="bar",ax=ax[1],color=[ORANGE,BLUE,"#199e70"],width=.78,legend=True)
ax[1].axhline(0,color="#222",lw=.8); ax[1].set_ylabel("conf − accuracy"); ax[1].set_xlabel("")
ax[1].legend(fontsize=7.4,frameon=False,title=None); ax[1].grid(axis="y",alpha=.18)
ax[1].tick_params(axis="x",rotation=0,labelsize=8)
ax[1].set_title("Why: Ostrava's over-confidence lives in the diseased classes",fontsize=10,fontweight="bold")
fig.suptitle("Per-hospital calibration — the committed 'case-mix' reading does not survive a proper adjustment",
             fontsize=11,fontweight="bold",y=1.02)
fig.text(.5,-.08,"Pooled 5-fold out-of-fold, n=6,700. Direct standardisation reweights each site's per-class gaps to the pooled class mix "
  "over the four classes with n≥30 at all three sites. Ostrava − Shenzhen after adjustment: +0.178, 95% CI [+0.140, +0.215] "
  "(patient-clustered, 2,000 reps). The Normal-only control collapses the spread only because every site is ≥94% accurate on Normal.",
  ha="center",fontsize=7.6,color="#444",wrap=True)
fig.tight_layout(); fig.savefig(PKG/"10_CONFIDENCE_CALIBRATION/site_calibration_corrected.png",bbox_inches="tight")
print("wrote site_calibration_corrected.png")
