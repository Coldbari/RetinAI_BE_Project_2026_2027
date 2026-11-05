"""P1 - dataset reconciliation, label granularity, leakage (exact + perceptual), patient-id semantics."""
import sys, json, hashlib, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
sys.path.insert(0,"/Users/praharsh/Project/RetinAI/FINAL_MODEL_EVIDENCE_PACKAGE/_build")
from common import *

spl=pd.read_csv(P/"splits.csv"); man=pd.read_csv(P/"manifest.csv")
prov=json.loads((ROOT/"data/rop/PROVENANCE.json").read_text())

# ---------- 1. reconciliation --------------------------------------------------------
acq=sum(d["images"] for d in prov["datasets"] if d["folder"]!="_redundant")
msk=sum(d.get("masks_or_overlays",0) for d in prov["datasets"] if d["folder"]!="_redundant")
rec=[dict(stage="0 acquired (images, masks excluded)", claimed=13037, recomputed=acq, source="data/rop/PROVENANCE.json"),
     dict(stage="0b masks/overlays on disk (not images)", claimed=None, recomputed=msk, source="PROVENANCE"),
     dict(stage="1 ICROP stage-labelled", claimed=10427, recomputed=len(man), source="results/prepared/rop_staging/manifest.csv"),
     dict(stage="2 six-class eligible", claimed=8024, recomputed=len(spl), source="splits.csv"),
     dict(stage="3a multiview_dev", claimed=663, recomputed=int((spl.split=='multiview_dev').sum()), source="splits.csv"),
     dict(stage="3b multiview_locked", claimed=661, recomputed=int((spl.split=='multiview_locked').sum()), source="splits.csv"),
     dict(stage="3c CV pool", claimed=6700, recomputed=int((spl.split=='train_cv').sum()), source="splits.csv"),
     dict(stage="4 trained after per-infant caps", claimed=3112, recomputed=int(((spl.kept)&(spl.split=='train_cv')).sum()), source="splits.csv kept flag"),
     dict(stage="patient_ids in CV pool", claimed=1528, recomputed=int(spl[spl.split=='train_cv'].patient_id.nunique()), source="splits.csv"),
     dict(stage="patient_ids eligible corpus", claimed=1760, recomputed=int(spl.patient_id.nunique()), source="splits.csv")]
rd=pd.DataFrame(rec); rd["match"]=np.where(rd.claimed.isna(),"n/a",np.where(rd.claimed==rd.recomputed,"YES","NO"))
write(rd,"02_DATASET/dataset_reconciliation.csv")

cls=spl.groupby(["split","cls6"]).size().unstack(fill_value=0)
cls.loc["TOTAL"]=cls.sum()
cls.to_csv(PKG/"02_DATASET/class_distribution.csv"); print("  wrote 02_DATASET/class_distribution.csv")
src=spl.groupby("source").agg(images=("image_path","size"), patient_ids=("patient_id","nunique"),
      images_per_id=("patient_id",lambda s: round(len(s)/s.nunique(),2)))
srccls=pd.crosstab(spl.source,spl.cls6)
pd.concat([src,srccls],axis=1).to_csv(PKG/"02_DATASET/source_class_matrix.csv"); print("  wrote 02_DATASET/source_class_matrix.csv")

# ---------- 2. label granularity -----------------------------------------------------
ann=pd.read_excel(ROOT/"data/rop/multiview/ROP-DATASET/Dataset Annotation Table.xlsx")
av=view_table()
a=ann.merge(av,on="Image_name",how="left")
pp=a.groupby("Patient").Label.nunique(); pe=a.groupby(["Patient","eye"]).Label.nunique()
lg=dict(images=len(a), patients=int(a.Patient.nunique()), eyes=int(pe.size),
        eyes_single_label=int((pe==1).sum()), eyes_multi_label=int((pe>1).sum()),
        patients_single_label=int((pp==1).sum()), patients_multi_label=int((pp>1).sum()),
        label_values=a.Label.astype(str).value_counts().to_dict())
(PKG/"02_DATASET/label_granularity_raw.json").write_text(json.dumps(lg,indent=2))
print("  label granularity:",{k:v for k,v in lg.items() if k!='label_values'})

# ---------- 3. exact duplicates ------------------------------------------------------
def md5(p):
    h=hashlib.md5()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): h.update(c)
    return h.hexdigest()
with ThreadPoolExecutor(16) as ex: spl["md5"]=list(ex.map(md5, spl.image_path.tolist()))
dup=spl[spl.duplicated("md5",keep=False)].sort_values(["md5","image_path"]).copy()
grp=dup.groupby("md5").agg(n=("image_path","size"), n_splits=("split","nunique"), splits=("split",lambda s:"|".join(sorted(set(s)))),
                           n_folds=("fold","nunique"), folds=("fold",lambda s:"|".join(map(str,sorted(set(s))))),
                           n_patient_ids=("patient_id","nunique"), patient_ids=("patient_id",lambda s:"|".join(sorted(set(s)))),
                           n_labels=("cls6","nunique"), labels=("cls6",lambda s:"|".join(sorted(set(s)))),
                           sources=("source",lambda s:"|".join(sorted(set(s))))).reset_index()
grp["crosses_split"]=grp.n_splits>1; grp["crosses_fold"]=grp.n_folds>1
grp["label_conflict"]=grp.n_labels>1; grp["id_conflict"]=grp.n_patient_ids>1
write(grp,"03_DATA_LEAKAGE/exact_duplicate_audit.csv")
write(dup[["md5","image_path","source","patient_id","cls6","split","fold"]],"03_DATA_LEAKAGE/exact_duplicate_members.csv")
write(grp[grp.label_conflict][["md5","labels","patient_ids","folds","splits","sources"]],"03_DATA_LEAKAGE/label_conflict_audit.csv")
print(f"  exact dup groups={len(grp)} cross-split={int(grp.crosses_split.sum())} cross-fold={int(grp.crosses_fold.sum())} label-conflict={int(grp.label_conflict.sum())}")
spl.to_csv(PKG/"_build/splits_md5.csv",index=False)

# ---------- 4. split leakage ---------------------------------------------------------
rows=[]
for a_,b_ in [("train_cv","multiview_dev"),("train_cv","multiview_locked"),("multiview_dev","multiview_locked")]:
    A=spl[spl.split==a_]; B=spl[spl.split==b_]
    rows.append(dict(comparison=f"{a_} vs {b_}", shared_patient_ids=len(set(A.patient_id)&set(B.patient_id)),
                     shared_image_paths=len(set(A.image_path)&set(B.image_path)),
                     shared_md5=len(set(A.md5)&set(B.md5)), n_a=len(A), n_b=len(B)))
for i in range(5):
    f=pd.read_csv(P/f"fold_{i}.csv")
    tr=set(f[f.split=="train"].patient_id); va=set(f[f.split=="val"].patient_id)
    rows.append(dict(comparison=f"fold_{i} train vs val", shared_patient_ids=len(tr&va), shared_image_paths=0,
                     shared_md5=None, n_a=int((f.split=='train').sum()), n_b=int((f.split=='val').sum())))
fs=pd.read_csv(P/"fold_site.csv")
rows.append(dict(comparison="fold_site train vs val", shared_patient_ids=len(set(fs[fs.split=='train'].patient_id)&set(fs[fs.split=='val'].patient_id)),
                 shared_image_paths=0, shared_md5=None, n_a=int((fs.split=='train').sum()), n_b=int((fs.split=='val').sum())))
write(pd.DataFrame(rows),"03_DATA_LEAKAGE/split_leakage_matrix.csv")

# ---------- 5. perceptual duplicates (full corpus) -----------------------------------
def dhash(path, size=16):
    try:
        im=Image.open(path).convert("L").resize((size+1,size),Image.BILINEAR)
        a=np.asarray(im,dtype=np.int16)
        return np.packbits((a[:,1:]>a[:,:-1]).ravel())
    except Exception: return None
with ThreadPoolExecutor(12) as ex: H=list(ex.map(dhash, spl.image_path.tolist()))
ok=np.array([h is not None for h in H]); Hm=np.stack([h for h in H if h is not None]).astype(np.uint8)
sub=spl[ok].reset_index(drop=True)
POP=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1).sum(1).astype(np.int16)
THR=10
pairs=[]
B=512
for i0 in range(0,len(Hm),B):
    blk=Hm[i0:i0+B]
    d=POP[np.bitwise_xor(blk[:,None,:],Hm[None,:,:])].sum(-1)
    for a_ in range(len(blk)):
        gi=i0+a_
        hit=np.where((d[a_]<=THR))[0]
        for gj in hit:
            if gj<=gi: continue
            pairs.append((gi,int(gj),int(d[a_][gj])))
pr=[]
for i,j,dd in pairs:
    A=sub.iloc[i]; Bb=sub.iloc[j]
    pr.append(dict(hamming=dd, exact=(A.md5==Bb.md5), path_a=A.image_path, path_b=Bb.image_path,
                   source_a=A.source, source_b=Bb.source, id_a=A.patient_id, id_b=Bb.patient_id,
                   label_a=A.cls6, label_b=Bb.cls6, split_a=A.split, split_b=Bb.split, fold_a=A.fold, fold_b=Bb.fold,
                   crosses_split=A.split!=Bb.split, crosses_fold=(A.fold!=Bb.fold) and (A.split==Bb.split=="train_cv"),
                   label_conflict=A.cls6!=Bb.cls6, different_patient_id=A.patient_id!=Bb.patient_id))
pdf=pd.DataFrame(pr).sort_values("hamming")
write(pdf,"03_DATA_LEAKAGE/perceptual_duplicate_audit.csv")
print(f"  perceptual pairs (dHash-256, hamming<={THR}): {len(pdf)} | cross-split {int(pdf.crosses_split.sum()) if len(pdf) else 0}"
      f" | cross-fold {int(pdf.crosses_fold.sum()) if len(pdf) else 0} | label-conflict {int(pdf.label_conflict.sum()) if len(pdf) else 0}")

# ---------- 6. patient-id semantics --------------------------------------------------
sem=[]
for s,g in spl.groupby("source"):
    ids=g.patient_id.nunique(); n=len(g)
    dupids=grp[(grp.sources==s)&(grp.id_conflict)]
    sem.append(dict(source=s, images=n, patient_ids=ids, images_per_id=round(n/ids,2),
                    ids_with_1_image=int((g.groupby('patient_id').size()==1).sum()),
                    max_images_per_id=int(g.groupby('patient_id').size().max()),
                    identical_images_under_different_ids=int(len(dupids))))
write(pd.DataFrame(sem),"03_DATA_LEAKAGE/patient_id_semantics.csv")
print(pd.DataFrame(sem).to_string(index=False))
