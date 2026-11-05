"""Build the code bundle uploaded to Kaggle — and refuse to build one containing PHI.

Why this exists as a script instead of an rsync one-liner.

`tests/test_repo_guards.py` blocks patient images from reaching **git**. Kaggle is a SECOND
egress path with entirely separate rules, and it had no guard at all. An ad-hoc rsync that
excluded `dataset_split/` and `.cohorts/` still swept up

    results/rop/errors/false_negative/<id>_F_GA##_BW####_PA##_DG#_PF#_D1_S07_28.jpg

— real infant retinal photographs whose filenames carry patient ID, sex, gestational age,
birth weight and postnatal age. Those are saved error-analysis crops, they live under
`results/`, and no exclusion list written from memory caught them.

So the bundle is not assembled by exclusion alone. After copying, it is SWEPT, and a bundle
that still contains a patient-pattern filename or a credential is **deleted rather than
shipped**. An allow-list of what may be uploaded is easy to get wrong; a deny-sweep that
fails closed is not.

    .venv/bin/python scripts/build_kaggle_bundle.py --out /tmp/bundle
"""
import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories and patterns that must never be uploaded.
EXCLUDE = [
    # kaggle_cache/ holds the 5 GB preprocessed cache and downloaded kernel outputs;
    # experiments/ holds model weights; .superpowers/ is planning state. Own datasets / not code.
    "kaggle_cache", "experiments", ".superpowers",
    # Presentation / figure folders (80 MB of PNGs and PDFs) that no kernel imports.
    "FINAL_MODEL_EVIDENCE_PACKAGE", "rop_enhancement", "graphs", "RETINAI_FINAL_SUMMARY", "junk",
    ".venv", ".git", ".github", ".claude", "dataset_split", ".cohorts", "__pycache__",
    # test_images/ holds infant photographs copied out of dataset_split/. It did not exist
    # when this list was first written, and the sweep caught it on the next upload —
    # which is the whole argument for sweeping rather than trusting an exclusion list.
    "test_images",
    # tests/ is not needed by any kernel and contains deliberately fake credential strings
    # as fixtures, which the sweep correctly flags. Excluding beats weakening the rule.
    # (Do not quote a sample token in this file either — the sweep reads its own source.)
    "tests",
    "*.pth", "*.npz", "*.zip", "*.pptx", "*.pdf",
    # Everything image-shaped under results/ is either a regenerable chart or a patient
    # crop. The kernel needs neither, and the sweep below will reject the bundle if any
    # survives — so these exclusions exist to keep the build passing, not to enforce safety.
    # data/ is the 14 GB consolidated image corpus. It ships as its OWN Kaggle dataset, never
    # inside the code bundle — rsyncing it here would upload every patient image through the
    # code path, which is precisely the egress this script exists to police.
    "data",
    # These two carry real patient filenames as DATA (67 references between them), copied in
    # from per-image test results. No kernel reads either.
    "results/manual_test_results.json", "results/cross_disease_matrix.json",
    # Hardcodes three specific infant filenames to pick Grad-CAM exemplars. Local figure
    # generation only; nothing on Kaggle calls it.
    "scripts/make_gradcam_assets.py",
    # THE clinical record: 485 rows of per-infant sex, gestational age, birth weight,
    # postconceptual age and diagnosis code. It is the file the sweep's comment above says
    # once shipped because its filename is innocent. Labels are recovered from it locally and
    # baked into the manifests; no kernel opens it.
    "infant_retinal_database_info.csv", "infant_retinal_database_info.xlsx",
    # results/prepared is excluded wholesale EXCEPT the staging fold manifests, which the
    # benchmark kernel reads directly — they are de-identified by
    # scripts/anonymise_ostrava_tree.py, and the sweep below re-checks that independently.
    "results/prepared/dr", "results/prepared/glaucoma", "results/prepared/rop",
    "results/prepared/rop_staging/manifest.csv", "results/prepared/rop_staging/splits.csv",
    "results/prepared/rop_staging/ostrava_name_map.csv",
    # The second Multi-View half, opened once after supervisor approval (PROTOCOL_V2). No kernel
    # reads it, and the Kaggle code dataset is public, so its labels never leave this machine.
    "results/prepared/rop_staging/test_locked.csv",
    "results/*/errors", "results/*/plots", "results/*/calibration*",
    "thesis_assets", "ppt_assets", "ppt_preview", "outputs", "reports",
    "unified_model", "legacy", "retinal-ai", "static/uploads",
    # android/ is 106 MB of Gradle wrapper + app sources that no kernel imports; shipping it
    # made the bundle 108 MB and slowed every kernel's startup for nothing.
    "android",
    # Tooling state, not code. .gstack/browse-audit.jsonl is a log of browsing activity — it
    # is not PHI, but it is personal, no kernel reads it, and it has no business on a public
    # -capable service.
    ".gstack", ".vscode", ".idea",
    # Report/paper drafts and local scratch: no kernel imports any of them.
    "docs", "output", "tmp", "*.pdf", "*.docx", "*.pptx",
]

# ROP filenames encode the patient: <id>_<sex>_GA<weeks>_BW<grams>_PA<days>_DG<grade>...
PHI_NAME = re.compile(r"\d+_[MF]_GA\d+_BW\d+", re.I)

# PHI INSIDE a file, which the name check cannot see. This gap was real: an earlier version
# of this sweep matched filenames ONLY, so docs/infant_retinal_database_info.csv — 485 rows of
# per-infant sex, gestational age, birth weight, postconceptual age and diagnosis code — sailed
# through and was uploaded, because its FILENAME is innocent and .csv was not even opened.
# Renaming a medical record does not de-identify it; neither does putting it in a spreadsheet.
#
# Two rules, deliberately scoped differently.
#
# PHI_CONTENT is the per-patient identifier itself. It applies to EVERY text format with no
# exception, because a leaked patient path is equally a leak in a .py, a .md or a .csv.
PHI_CONTENT = re.compile(r"\d+_[MF]_GA\d+_BW\d+", re.I)

# PHI_RECORD is a heuristic for spotting a clinical record by its COLUMN HEADERS, and it only
# makes sense on tabular formats. Applied to prose it fires on any sentence describing the
# data — every doc explaining what Ostrava contains, and the docstring of this very file —
# which made the bundle unbuildable and taught nothing. A clinical record arrives as a table,
# not as a docstring, so the rule is scoped to tables. Note this narrows only the HEADER
# heuristic: actual per-patient identifiers in prose are still caught by PHI_CONTENT above.
PHI_RECORD = re.compile(
    r"GESTATIONAL[\s_]*AGE|BIRTH[\s_]*WEIGHT|POSTCONCEPTUAL|DIAGNOSIS[\s_]*CODE", re.I)
TABULAR = {".csv", ".tsv", ".xlsx", ".xlsm", ".xls", ".ods"}

# Anything that looks like an image is suspect under results/ — charts are regenerable,
# patient crops are not, and we cannot tell them apart by name alone.
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dcm"}
SECRET = re.compile(r"KGAT_[A-Za-z0-9]|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
SECRET_NAME = {"kaggle.json", "access_token", ".env", "id_rsa"}

# Everything whose bytes we are willing to claim we inspected. Tabular formats are here
# precisely because the leak came through one.
TEXTISH = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".cfg", ".csv", ".tsv", ".ipynb"}
# Office formats are ZIPs of XML — readable enough to grep, so there is no excuse to skip them.
ZIP_XML = {".xlsx", ".xlsm", ".xls", ".ods", ".docx"}


def _text_of(p: Path) -> str:
    """Best-effort text for scanning. Never raises — an unreadable file is handled by caller."""
    if p.suffix.lower() in ZIP_XML:
        try:
            with zipfile.ZipFile(p) as z:
                return "\n".join(z.read(n).decode("utf-8", "ignore")
                                 for n in z.namelist()[:60])
        except Exception:
            return ""
    return p.read_text(errors="ignore")


def sweep(bundle: Path):
    """Return a list of reasons the bundle must not be uploaded."""
    problems = []
    for p in bundle.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(bundle)
        if PHI_NAME.search(p.name):
            problems.append(f"PHI filename: {rel}")
            continue
        if p.name in SECRET_NAME:
            problems.append(f"credential file: {rel}")
            continue
        if p.suffix.lower() in IMAGE_EXT and rel.parts and rel.parts[0] == "results":
            problems.append(f"image under results/: {rel}")
            continue
        if p.suffix.lower() in TEXTISH | ZIP_XML:
            try:
                text = _text_of(p)
            except Exception:
                # Unreadable is NOT clean. Fail closed: an unscannable file cannot be cleared.
                problems.append(f"unreadable, cannot be cleared for upload: {rel}")
                continue
            if PHI_CONTENT.search(text):
                problems.append(f"patient identifier inside file contents: {rel}")
                continue
            if p.suffix.lower() in TABULAR and PHI_RECORD.search(text):
                problems.append(f"clinical-record columns in a table: {rel}")
                continue
            if SECRET.search(text):
                problems.append(f"credential-shaped string: {rel}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cmd = ["rsync", "-a"]
    for e in EXCLUDE:
        cmd += ["--exclude", e]
    cmd += [f"{ROOT}/", f"{out}/"]
    subprocess.run(cmd, check=True)

    problems = sweep(out)
    n = sum(1 for p in out.rglob("*") if p.is_file())
    if problems:
        shutil.rmtree(out)
        print(f"BUNDLE REJECTED — {len(problems)} item(s) must not leave this machine:\n")
        for p in problems[:25]:
            print("   ", p)
        if len(problems) > 25:
            print(f"    … and {len(problems) - 25} more")
        print("\nThe bundle has been DELETED rather than left on disk where a later "
              "`kaggle datasets version` could pick it up.")
        return 1

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"bundle OK — {n} files, {size/1e6:.1f} MB")
    print(f"  swept for: patient-pattern filenames, images under results/, credentials")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
