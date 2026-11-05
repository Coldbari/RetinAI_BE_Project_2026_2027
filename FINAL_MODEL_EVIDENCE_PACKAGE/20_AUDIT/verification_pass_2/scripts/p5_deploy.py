"""P5 - served-model deployment footprint: params, size, latency (CPU + MPS), TTA cost."""
import sys, json, time, hashlib, numpy as np, torch
from pathlib import Path
sys.path.insert(0,"/Users/praharsh/Project/RetinAI")
sys.path.insert(0,"/Users/praharsh/Project/RetinAI/FINAL_MODEL_EVIDENCE_PACKAGE/_build")
from common import ROOT, PKG
from models.common.structured import StructuredROPModel
from models.common.config import load_config
from models.common.preprocessing import build_transforms
from PIL import Image

W=ROOT/"results/rop_staging/weights.pth"; cfg=load_config(str(ROOT/"configs/rop_staging_structured.yaml"))
sha=hashlib.sha256(W.read_bytes()).hexdigest()
state=torch.load(W,map_location="cpu"); n_sites=state["site_head.3.bias"].shape[0]
model=StructuredROPModel(cfg.model.arch,n_sites=n_sites,pretrained=False); model.load_state_dict(state,strict=True); model.eval()
def count(mod): return sum(p.numel() for p in mod.parameters())
parts={"total":count(model)}
for name,mod in model.named_children(): parts[name]=count(mod)
res=dict(checkpoint=str(W.relative_to(ROOT)), sha256=sha, size_bytes=W.stat().st_size,
         size_mb=round(W.stat().st_size/1024**2,2), state_dict_keys=len(state), n_sites=int(n_sites),
         arch=cfg.model.arch, image_size=cfg.preprocess.image_size, params=parts,
         params_million=round(parts["total"]/1e6,2),
         inference_only_params=round((parts["total"]-parts.get("site_head",0))/1e6,2))
# latency
img=Image.open(sorted((ROOT/"test_images/rop").glob("*.jpg"))[0]).convert("RGB")
tfm=build_transforms(cfg,"val")
t0=time.perf_counter(); x=tfm(image=np.array(img))["image"] if hasattr(tfm,"__call__") and "albumentations" in str(type(tfm)) else tfm(img); pre_ms=(time.perf_counter()-t0)*1000
x=x.unsqueeze(0)
lat={}
for dev in (["cpu"]+(["mps"] if torch.backends.mps.is_available() else [])):
    m=model.to(dev); xx=x.to(dev)
    with torch.no_grad():
        for _ in range(3): m(xx)
        if dev=="mps": torch.mps.synchronize()
        ts=[]
        for _ in range(20):
            t=time.perf_counter(); m(xx)
            if dev=="mps": torch.mps.synchronize()
            ts.append((time.perf_counter()-t)*1000)
    lat[dev]=dict(median_ms=round(float(np.median(ts)),2), p90_ms=round(float(np.percentile(ts,90)),2),
                  mean_ms=round(float(np.mean(ts)),2), n_runs=len(ts))
res["preprocess_ms_single_image"]=round(pre_ms,2); res["latency_single_forward"]=lat
res["tta_used_at_serve_time"]=False
res["note_tta"]=("webapp/inference.py runs ONE forward pass per uploaded image for the staging model. "
                 "The 5-view TTA in the artifacts belongs to the RETIRED ResNet50 screening path "
                 "(results/rop/operating_point.json, 'derived_on: 5-view TTA').")
res["forward_passes_per_uploaded_image"]=1
res["examination_estimate_6_frames_ms"]={k:round(v["median_ms"]*6+pre_ms*6,1) for k,v in lat.items()}
(PKG/"15_DEPLOYMENT/deployment_footprint.json").write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
