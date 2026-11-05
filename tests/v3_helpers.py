"""Helpers shared by the v3 tests (Tasks 2-6 of the rop-v3 implementation plan,
docs/superpowers/plans/2026-10-19-rop-v3-implementation.md).

Not a test module (pytest only collects ``test_*.py``); the v3 tests import it directly, the
same way test_api_and_report.py imports conftest.
"""
from pathlib import Path

from models.common.config import load_config

BASE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "rop_staging.yaml"

# The v3 tests feed synthetic images (a flat grey square), not fundus photographs, and the three
# fundus steps would change the very pixels a test asserts on: CLAHE re-maps flat grey (128 -> 134
# measured), circle_crop trims and letterbox pads any non-square or dark-bordered input. So they
# are off unless a test asks for them back: later overrides win, so ``_cfg(clahe=True)`` works,
# and ``_cfg(plain=False)`` leaves the YAML's own values for a test about the flags themselves.
_SYNTHETIC_IMAGE_SAFE = ("preprocess.circle_crop=false",
                         "preprocess.clahe=false",
                         "preprocess.letterbox=false")

# Which config section each ``_cfg`` kwarg belongs to. CLOSED on purpose: an unknown kwarg raises
# KeyError. The helper used to send every kwarg to ``preprocess.<k>``, so ``_cfg(arch="dinov2_s")``
# set a key nothing reads and the test silently ran the baseline convnext_tiny. A new knob is one
# line here, added by the task that needs it.
_ROUTES = {
    "model": ("arch", "drop_path", "weights_path", "dropout"),
    "train": ("lr", "layer_decay", "epochs", "batch_size", "weight_decay"),
    "preprocess": ("image_size", "normalize", "cached", "cache_root",
                   "circle_crop", "clahe", "letterbox"),
}
_SECTION_OF = {key: section for section, keys in _ROUTES.items() for key in keys}


def _cfg(*, plain: bool = True, **over):
    """``configs/rop_staging.yaml`` with every kwarg applied to the section it belongs to.

    Goes through ``load_config``'s ``key=value`` override path, i.e. the same string parsing a
    Kaggle kernel's ``--set`` flag uses, so ``_cfg(image_size=56)`` is an int and
    ``_cfg(normalize="half")`` a str exactly as they would be in a real run.

    ``plain=True`` (default) forces circle_crop / clahe / letterbox off for synthetic images;
    ``plain=False`` keeps what the YAML says. A kwarg that is not in ``_ROUTES`` raises
    ``KeyError`` rather than becoming a key nothing reads.
    """
    unknown = sorted(set(over) - set(_SECTION_OF))
    if unknown:
        raise KeyError(f"_cfg has no section for {unknown}; known kwargs: {sorted(_SECTION_OF)}")
    overrides = [*(_SYNTHETIC_IMAGE_SAFE if plain else ()),
                 *(f"{_SECTION_OF[k]}.{k}={v}" for k, v in over.items())]
    return load_config(BASE_CONFIG, overrides=overrides)
