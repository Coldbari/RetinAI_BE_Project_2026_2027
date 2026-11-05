"""The Kaggle code bundle must drop drafts/scratch but keep everything a kernel imports."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_kaggle_bundle import EXCLUDE  # noqa: E402

NEEDED = ["kaggle_kernels", "models", "scripts", "configs", "results/prepared/rop_staging"]


def test_excludes_drafts_and_scratch():
    for e in ["docs", "output", "tmp", "*.pdf", "*.docx", "*.pptx"]:
        assert e in EXCLUDE


def test_keeps_kernel_inputs():
    for need in NEEDED:
        for e in EXCLUDE:
            assert e != need and not need.startswith(e.rstrip("/") + "/"), (need, e)


def test_excludes_test_locked_but_keeps_test_dev():
    assert "results/prepared/rop_staging/test_locked.csv" in EXCLUDE
    dev = "results/prepared/rop_staging/test_dev.csv"
    for e in EXCLUDE:
        assert e != dev and not dev.startswith(e.rstrip("/") + "/"), e


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_rsync_with_the_exclude_list_drops_test_locked_only(tmp_path):
    """The build's own rsync call on a stand-in tree (fake files; the real test_locked is never read)."""
    src, out = tmp_path / "src", tmp_path / "out"
    keep = ["results/prepared/rop_staging/test_dev.csv", "results/prepared/rop_staging/fold_0.csv",
            "results/prepared/rop_staging/v4/fold_loso_ropvl.csv", "models/common/fourier_aug.py",
            "models/common/losses.py"]
    for rel in keep + ["results/prepared/rop_staging/test_locked.csv"]:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text("x\n")
    cmd = ["rsync", "-a"] + [a for e in EXCLUDE for a in ("--exclude", e)] + [f"{src}/", f"{out}/"]
    subprocess.run(cmd, check=True)
    got = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert got == sorted(keep)
