"""Shared test fixtures.

`fundus_like()` exists because Phase 0B's gradability gate runs BEFORE routing. A flat colour
block — the obvious synthetic stand-in — is now correctly rejected (zero Laplacian variance),
so routing tests that used one were testing the gate instead of the router. This builds a
minimal image that passes the gate: circular field of view, red-dominant, saturated, textured.

It is a plumbing fixture, not a fundus photograph. Anything measuring model *accuracy* must use
real cohort images (tests/build_cohorts.py), never this.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def fundus_like(size=512, seed=0):
    """An image that satisfies the FULL gradability gate.

    Since the embedding gate (2026-08-19) the synthetic red disc no longer suffices - the
    learned layer correctly measures that a painted disc does not resemble real retinal
    imagery, which is exactly its job. Tests that need to reach the prediction path now
    get a REAL train-split fundus when one is on disk, and the synthetic disc only as a
    fallback for machines without the (private, gitignored) image tree - on those, tests
    that need to pass the full gate will skip at the fixture that uses this.
    """
    from pathlib import Path
    from PIL import Image as _Image
    for pattern in ("dataset_split/train", "data/rop/shenzhen"):
        root = Path(__file__).resolve().parent.parent / pattern
        if root.exists():
            for f in sorted(root.rglob("*.jpg"))[:1]:
                return _Image.open(f).convert("RGB")
    return _synthetic_fundus_like(size, seed)


def _synthetic_fundus_like(size=512, seed=0):
    """Synthetic disc that satisfies the HEURISTIC gate layers only."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size / 2
    disc = ((yy - cy) ** 2 + (xx - cx) ** 2) < (size * 0.46) ** 2

    img = np.zeros((size, size, 3), np.float32)
    # red-dominant retinal background with a slow illumination gradient
    grad = 0.75 + 0.25 * (1 - ((yy - cy) ** 2 + (xx - cx) ** 2) / (size * 0.46) ** 2).clip(0, 1)
    img[..., 0] = 185 * grad
    img[..., 1] = 70 * grad
    img[..., 2] = 42 * grad
    # vessel-like dark streaks + fine grain, so Laplacian variance is non-trivial
    for k in range(14):
        ang = rng.uniform(0, np.pi)
        d = np.abs((xx - cx) * np.sin(ang) - (yy - cy) * np.cos(ang))
        img[d < rng.uniform(1.2, 2.6)] *= 0.55
    img += rng.normal(0, 7, img.shape)

    img *= disc[..., None]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


@pytest.fixture(scope="session")
def fundus_image():
    return fundus_like()


def pytest_configure():
    """Fail loudly if the fixture cannot pass the gate — otherwise every test using it
    would silently be exercising the gate rather than the behaviour under test."""
    from models.common.data_quality import gradability
    r = gradability(fundus_like())
    if not r["gradable"]:
        raise RuntimeError(
            f"tests/conftest.py: fundus_like() is rejected by the gradability gate "
            f"({r['verdict']}: {r['reason']}). Fix the fixture — routing tests would "
            f"otherwise be testing the gate by accident.")


# This public copy leaves out the patient-level fold files, because their rows are image paths.
# The tests below read those files, so they skip here and run in the development repository.
_NEEDS_FOLD_FILES = {
    "test_duplicate_exclusion_list_has_fourteen_in_pool_images",
    "test_committed_folds_have_no_mixed_label_ostrava_session",
    "test_inner_split_on_committed_fold0",
    "test_duplicates_dropped_from_train_only_on_committed_folds",
    "test_fold_file_paths_for_v3_and_v4_pools_and_loso",
    "test_target_frame_carries_image_paths_only",
}


def pytest_collection_modifyitems(items):
    if (ROOT / "results/prepared/rop_staging/fold_0.csv").exists():
        return
    skip = pytest.mark.skip(reason="reads patient-level fold files, not in this public repository")
    for item in items:
        if getattr(item, "originalname", item.name) in _NEEDS_FOLD_FILES:
            item.add_marker(skip)
