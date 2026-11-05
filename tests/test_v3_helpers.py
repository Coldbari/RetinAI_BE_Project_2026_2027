"""The shared v3 test helper is itself a source of silent no-ops, so it is pinned.

``_cfg(arch="dinov2_s")`` used to become ``preprocess.arch=dinov2_s`` and leave ``model.arch`` at
the baseline convnext_tiny; ``_cfg(lr=1e-4)`` left ``train.lr`` at 3e-4. A test written against
that exercised the wrong model and passed. So every kwarg is routed by a CLOSED table to the
section it belongs to, and anything the table does not know raises instead of being written to
``preprocess.<typo>`` where nothing would ever read it.
"""
import pytest

from v3_helpers import _cfg


@pytest.mark.parametrize("section,key,value", [
    ("model", "arch", "dinov2_s"),
    ("model", "drop_path", 0.1),
    ("model", "weights_path", "/w/rfg.pth"),
    ("model", "dropout", 0.5),
    ("train", "lr", 1e-4),
    ("train", "layer_decay", 0.8),
    ("train", "epochs", 20),
    ("train", "batch_size", 8),
    ("train", "weight_decay", 0.01),
    ("preprocess", "image_size", 56),
    ("preprocess", "normalize", "half"),
    ("preprocess", "cached", True),
    ("preprocess", "cache_root", "/kaggle/input/cache"),
    ("preprocess", "circle_crop", True),
    ("preprocess", "clahe", True),
    ("preprocess", "letterbox", True),
])
def test_each_kwarg_lands_in_its_own_section_and_nowhere_else(section, key, value):
    cfg = _cfg(**{key: value})
    assert getattr(cfg, section)[key] == value
    for other in ("model", "train", "preprocess"):
        if other != section:
            assert key not in getattr(cfg, other)


def test_the_named_cases_from_the_ruling():
    assert _cfg(arch="dinov2_s").model.arch == "dinov2_s"
    assert _cfg(lr=1e-4).train.lr == 1e-4
    # and the baseline underneath is still the baseline
    base = _cfg()
    assert base.model.arch == "convnext_tiny" and base.train.lr == 3e-4


@pytest.mark.parametrize("typo", ["bogus", "imagesize", "Arch", "num_workers"])
def test_an_unknown_kwarg_raises_instead_of_becoming_a_preprocess_key(typo):
    with pytest.raises(KeyError, match=typo):
        _cfg(**{typo: 1})


def test_the_error_names_the_typo_and_lists_what_is_known():
    # The bare KeyError a dict lookup would give says only 'imagesize'; the message must also
    # offer image_size, or the person who mistyped has to go and read the helper.
    with pytest.raises(KeyError) as err:
        _cfg(imagesize=56)
    message = err.value.args[0]
    assert "imagesize" in message and "image_size" in message


def test_plain_defaults_to_the_three_fundus_steps_off():
    p = _cfg().preprocess
    assert (p.circle_crop, p.clahe, p.letterbox) == (False, False, False)
    p = _cfg(plain=True).preprocess
    assert (p.circle_crop, p.clahe, p.letterbox) == (False, False, False)


def test_plain_false_leaves_the_yamls_values():
    p = _cfg(plain=False).preprocess
    assert p.clahe is True and p.circle_crop is True and p.letterbox is True


def test_an_explicit_kwarg_beats_the_plain_default():
    p = _cfg(clahe=True).preprocess
    assert (p.circle_crop, p.clahe, p.letterbox) == (False, True, False)
    # and plain=False does not stop an explicit kwarg either
    p = _cfg(plain=False, clahe=False).preprocess
    assert (p.circle_crop, p.clahe, p.letterbox) == (True, False, True)


def test_plain_is_not_mistaken_for_a_config_key():
    cfg = _cfg(plain=False)
    assert all("plain" not in getattr(cfg, s) for s in ("model", "train", "preprocess"))
