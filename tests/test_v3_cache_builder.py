"""Cache builder and gate: pure-function behaviour on synthetic images."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.common.datasets import cached_path  # noqa: E402
from scripts.build_rop_cache import build_cache  # noqa: E402
from scripts.rop_cache_gate import check_stamp, completeness, psnr  # noqa: E402


def _make(tmp_path, n=2):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rel = f"data/rop/ostrava/val/DG0/img_{i}.jpg"
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((200, 300, 3), np.uint8)
        arr[20:180, 40:260] = rng.integers(40, 200, (160, 220, 3), dtype=np.uint8)
        im = Image.new("RGB", (300, 200))
        im.paste(Image.fromarray(arr))
        exif = Image.Exif()
        exif[0x010E] = "secret description"
        im.save(p, exif=exif)
        rows.append({"image_path": rel, "label": 0, "split": "val"})
    m = tmp_path / "fold_0.csv"
    pd.DataFrame(rows).to_csv(m, index=False)
    return m, rows


def test_build_writes_square_768_exif_free_mirrored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m, rows = _make(tmp_path)
    out = tmp_path / "cache"
    res = build_cache([m], out, workers=1)
    assert res["written"] == 2 and res["skipped_existing"] == 0
    for r in rows:
        f = cached_path(r["image_path"], str(out))
        assert f.exists() and f == out / "rop/ostrava/val/DG0" / Path(r["image_path"]).name
        im = Image.open(f)
        assert im.size == (768, 768) and im.format == "PNG"
        assert not im.getexif()
        assert "exif" not in im.info
    again = build_cache([m], out, workers=1)
    assert again == {"written": 0, "skipped_existing": 2}


def test_build_refuses_locked_manifest(tmp_path):
    m, _ = _make(tmp_path)
    locked = tmp_path / "test_locked.csv"
    locked.write_text(m.read_text())
    with pytest.raises(ValueError, match="locked"):
        build_cache([m, locked], tmp_path / "cache")


def test_psnr_identical_is_infinite_and_noise_is_finite():
    a = np.full((8, 8, 3), 100, np.uint8)
    assert psnr(a, a) == float("inf")
    b = a.copy(); b[0, 0] = 110
    assert 40 < psnr(a, b) < 100


def test_check_stamp_flags_mismatch(tmp_path):
    cfg_flags = {"circle_crop": True, "letterbox": True, "clahe": True, "ben_graham": False}
    (tmp_path / "CACHE_INFO.json").write_text(json.dumps(cfg_flags))
    check_stamp(tmp_path, cfg_flags)
    (tmp_path / "CACHE_INFO.json").write_text(json.dumps({**cfg_flags, "clahe": False}))
    with pytest.raises(SystemExit):
        check_stamp(tmp_path, cfg_flags)


def test_completeness_detects_missing_and_locked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m, rows = _make(tmp_path)
    out = tmp_path / "cache"
    build_cache([m], out, workers=1)
    assert completeness([m], out, locked=None)["missing"] == 0
    cached_path(rows[0]["image_path"], str(out)).unlink()
    assert completeness([m], out, locked=None)["missing"] == 1
    lk = tmp_path / "test_locked.csv"
    pd.DataFrame([rows[1]]).to_csv(lk, index=False)   # present in cache -> must be flagged
    assert completeness([m], out, locked=lk)["locked_present"] == 1
