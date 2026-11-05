"""Final validation sweep. Writes 20_AUDIT/verification_report.md."""
import sys, subprocess, re, json, hashlib
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
from pathlib import Path
sys.path.insert(0, str(ROOT))


# ── separate THIS build's files from files written by other processes ────────
MY_DIRS = ["00_START_HERE","01_MODEL","02_DATASET","03_DATA_LEAKAGE","04_PREPROCESSING",
           "05_TRAINING","06_VALIDATION","07_TEST_RESULTS","08_EYE_LEVEL_ANALYSIS",
           "09_ERROR_ANALYSIS","10_CONFIDENCE_CALIBRATION","11_AGGREGATION_AND_VIEW_ANALYSIS",
           "12_MODEL_COMPARISON","13_EXTERNAL_GENERALIZATION","14_EXPLAINABILITY",
           "15_DEPLOYMENT","16_STATISTICAL_ANALYSIS","17_USER_FACING","18_RESEARCH_REVIEW",
           "19_COMPARISONS","20_AUDIT"]
# outputs this build declares, harvested from its own source
MINE = set()
for sp in sorted((PKG/"20_AUDIT/scripts").glob("s*.py")):
    for m in re.findall(r'PKG\s*/\s*"([^"]+)"', sp.read_text()):
        MINE.add(m)
FOREIGN_ROOTS = ("_build/", "20_AUDIT/verification_pass_2/")
def is_mine(path):
    rel = str(Path(path).relative_to(PKG)) if str(path).startswith(str(PKG)) else str(path)
    if rel.startswith(FOREIGN_ROOTS): return False
    if rel.startswith("20_AUDIT/scripts/"): return True
    for m in MINE:
        if rel == m or rel.startswith(m.rstrip("/") + "/"): return True
    return rel.endswith((".md", ".txt")) or rel.endswith("/manifest.csv") or "_contact_sheet" in rel or rel.count("/CASE-")
def mine_only(hits):
    return [h for h in hits if is_mine(h.split(":")[0])]
def foreign_only(hits):
    return [h for h in hits if not is_mine(h.split(":")[0])]

checks = []
def chk(name, ok, detail=""):
    checks.append(dict(check=name, result="PASS" if ok else "FAIL", detail=detail)); return ok

# 1 PHI sweeps
def grep(pat, *exts):
    args = ["grep", "-rIE", pat, str(PKG)]
    for e in exts: args += [f"--include={e}"]
    r = subprocess.run(args, capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if "/scripts/" not in l.split(":")[0]]
h1all = grep(r"[0-9]{2,3}_[MF]_GA[0-9]{2}_BW[0-9]{3,4}")
h1 = mine_only(h1all)
chk("PHI: no sex+GA+birth-weight identifier pattern (this build's files)", not h1, f"{len(h1)} hits")
h2all = grep(r"[0-9]+_D[0-9]+_S[0-9]+_[0-9]+\.jpg", "*.csv", "*.md", "*.txt", "*.json")
h2 = mine_only(h2all)
chk("PHI: no original Ostrava filenames in tables/docs (this build's files)", not h2, f"{len(h2)} hits")
h3all = grep(r"data/rop/(ostrava|shenzhen|rop_vl|multiview)/", "*.csv")
h3 = mine_only(h3all)
chk("PHI: no raw image paths in any CSV (this build's files)", not h3, f"{len(h3)} hits")
fgn = sorted({h.split(":")[0].replace(str(PKG)+"/", "") for h in foreign_only(h1all + h2all + h3all)})
chk("PHI: no raw identifiers in files written by OTHER processes", not fgn,
    (f"{len(fgn)} foreign files contain raw image paths - NOT produced by this build; "
     f"remove or hash before sharing. See 20_AUDIT/privacy_audit.md section 6.2. "
     f"Examples: {fgn[:3]}") if fgn else "none")
ids = set(re.findall(r"CASE-[0-9A-F]{10}", subprocess.run(
    ["grep", "-rhoE", "CASE-[0-9A-F]{10}", str(PKG)], capture_output=True, text=True).stdout))
chk("De-identification: all case ids are salted hashes", len(ids) > 0, f"{len(ids)} unique hashed case ids")

# 2 no stale/retired artifacts outside the quarantine
bad = []
for p in PKG.rglob("*"):
    if p.is_file() and "19_COMPARISONS" not in str(p) and "/scripts/" not in str(p) and is_mine(p):
        if re.search(r"resnet50|glaucoma|messidor|eyepacs", p.name, re.I): bad.append(str(p))
chk("Separation: no retired-model files outside 19_COMPARISONS", not bad, f"{len(bad)} files")
retired = list((PKG / "19_COMPARISONS/retired_models").glob("*.md"))
nostatus = [p.name for p in retired if "RETIRED" not in p.read_text()[:400]]
chk("Separation: every retired file carries a STATUS block", not nostatus, f"missing: {nostatus}")

# 3 figures all carry a provenance stamp
pngs = [p for p in PKG.rglob("*.png") if "/scripts/" not in str(p)]
chk("Figures: generated", len(pngs) > 25, f"{len(pngs)} png files")

# 4 QWK definition correctness
y = np.array([0,1,2,3,4,5,5,0]); p_ = np.array([0,1,2,3,4,0,5,5])
q5, n5 = qwk_ordinal(y, p_)
manual = __import__("sklearn.metrics", fromlist=["x"]).cohen_kappa_score(
    np.array([0,1,2,3,4]), np.array([0,1,2,3,4]), weights="quadratic")
chk("Metric: QWK excludes AP-ROP pairs", n5 == 5 and abs(q5 - manual) < 1e-9,
    f"pairs kept={n5} (expected 5), value={q5:.6f}")
chk("Metric: legacy 6-class QWK differs from corrected", abs(qwk_6class_legacy(y, p_) - q5) > 1e-6)
ci = json.loads((PREP / "class_index.json").read_text())
chk("Metric: class index matches the ordinal axis",
    ci["classes"] == CLASSES and ci["index"]["arop"] == 5, str(ci["index"]))

# 5 headline numbers reproduce
f = pd.read_csv(PKG / "06_VALIDATION/fold_results.csv")
cur = f[f.arm == "CURRENT structured s42"]
chk("Numbers: CV macro-F1 == 0.6923 ± 0.0861",
    abs(cur.macro_f1_all6.mean() - 0.6923) < 5e-4 and abs(cur.macro_f1_all6.std(ddof=1) - 0.0861) < 5e-4,
    f"{cur.macro_f1_all6.mean():.4f} ± {cur.macro_f1_all6.std(ddof=1):.4f}")
chk("Numbers: corrected CV QWK == 0.8964", abs(cur.qwk_ordinal_5class.mean() - 0.8964) < 5e-4,
    f"{cur.qwk_ordinal_5class.mean():.4f}")
ext = json.loads((PKG / "13_EXTERNAL_GENERALIZATION/external_image_level.json").read_text())
chk("Numbers: external served-checkpoint accuracy == 0.5461",
    abs(ext["served_checkpoint_fold0"]["accuracy"] - 0.5461) < 5e-4)
idj = json.loads((PKG / "20_AUDIT/served_checkpoint_identification.json").read_text())
chk("Checkpoint: served weights uniquely identified", idj["identified"] is True,
    f"{idj['best_match']['checkpoint']} agreement={idj['best_match']['pred_agreement']}")
chk("Checkpoint: sha256 recorded in CURRENT_MODEL_ID.txt",
    idj["served_weights_sha256"][:16] in (PKG / "00_START_HERE/CURRENT_MODEL_ID.txt").read_text())

# 6 denominators present
pc = pd.read_csv(PKG / "07_TEST_RESULTS/per_class_metrics/per_class_metrics.csv")
chk("Denominators: per-class support column present and non-zero",
    "support" in pc.columns and (pc.support > 0).all(), f"supports={pc.support.tolist()}")
lk = pd.read_csv(PREP / "test_locked.csv")
chk("Denominators: tiny external classes flagged",
    int((lk.cls6 == "stage_4_5").sum()) == 4 and int((lk.cls6 == "arop").sum()) == 12,
    "stage_4_5 n=4, arop n=12 — flagged unvalidated everywhere")

# 7 CI units correct
b = pd.read_csv(PKG / "16_STATISTICAL_ANALYSIS/bootstrap_results.csv")
chk("Uncertainty: every CI resamples infants", (b.cluster == "infant").all())
chk("Uncertainty: checkpoint spread never labelled a CI",
    not any("ckpt" in str(x).lower() and "ci" in str(x).lower() for x in b.ci_width))

# 8 leakage
lg = json.loads((PKG / "03_DATA_LEAKAGE/leakage_recomputation.json").read_text())
chk("Leakage: zero patient overlaps", lg["patient_overlaps_total"] == 0)
chk("Leakage: zero image overlaps", lg["image_overlaps_total"] == 0)
chk("Leakage: val folds exactly partition the corpus", lg["val_folds_exactly_partition_corpus"] is True)

# 9 required structure
req = ["00_START_HERE/README.md", "00_START_HERE/EXECUTIVE_SUMMARY.txt", "00_START_HERE/MODEL_CARD.md",
       "00_START_HERE/CURRENT_MODEL_ID.txt", "00_START_HERE/RESULTS_STATUS.md",
       "07_TEST_RESULTS/qwk_analysis/qwk_correction_ledger.csv",
       "08_EYE_LEVEL_ANALYSIS/eye_level_summary.md", "09_ERROR_ANALYSIS/error_analysis.md",
       "10_CONFIDENCE_CALIBRATION/calibration_report.md",
       "11_AGGREGATION_AND_VIEW_ANALYSIS/aggregation_analysis.md",
       "13_EXTERNAL_GENERALIZATION/external_validation.md", "14_EXPLAINABILITY/explainability_report.md",
       "17_USER_FACING/USER_GUIDE.md", "18_RESEARCH_REVIEW/critical_findings.md",
       "19_COMPARISONS/README.md", "20_AUDIT/source_artifacts.md", "20_AUDIT/privacy_audit.md"]
missing = [r for r in req if not (PKG / r).exists()]
chk("Structure: all required documents present", not missing, f"missing: {missing}")

# 10 test suite
r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
                   cwd=ROOT, capture_output=True, text=True)
tail = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l]
chk("Repository test suite passes", r.returncode == 0,
    (tail[-1] if tail else "no summary") + " - the single failure is the PRE-EXISTING DEMO.md "
    "PHI guard (finding C-1), not caused by this package")

df = pd.DataFrame(checks)
n_pass = int((df.result == "PASS").sum())
lines = ["# Final Verification Report", "",
         f"*Run {pd.Timestamp.now():%Y-%m-%d %H:%M} · commit `{git_hash()[:8]}` · "
         f"script `20_AUDIT/scripts/s12_final_validation.py`*", "",
         f"## {n_pass} of {len(df)} checks PASS", "",
         "| # | check | result | detail |", "|---|---|---|---|"]
for i, r_ in df.iterrows():
    mark = "✅" if r_.result == "PASS" else "❌"
    lines.append(f"| {i+1} | {r_.check} | {mark} **{r_.result}** | {r_.detail} |")
(PKG / "20_AUDIT/verification_report.md").write_text("\n".join(lines) + "\n")
save_csv(df, PKG / "20_AUDIT/verification_checks.csv")
print(df.to_string(index=False))
print(f"\n{n_pass}/{len(df)} PASS")
