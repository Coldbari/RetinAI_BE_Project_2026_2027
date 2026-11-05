"""Task 2 of the v3 plan: the normalisation statistics are a config key that every transform
builder reads.

RETFound-Green was pretrained on pixels scaled to mean = std = 0.5, its DINOv2 control on ImageNet
statistics. A key honoured by only one builder crashes nothing: a model trained under one set of
statistics is validated or served under the other, and the predictions just drift. So these tests
pin the key in every place the repo builds a pipeline (train, val, and the five TTA views the web
app serves) and pin that a config WITHOUT the key still gets exactly the ImageNet numbers every
already-trained checkpoint was fitted with.
"""
import re

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from models.common import preprocessing  # noqa: E402
from v3_helpers import _cfg  # noqa: E402

# Literals on purpose, not the module's constants: a typo in the constants must fail here.
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
HALF = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def _pipelines(cfg):
    """Every transform the repo builds from a config: train, val, and the five TTA views."""
    return [preprocessing.build_transforms(cfg, "train"),
            preprocessing.build_transforms(cfg, "val"),
            *preprocessing.build_tta_transforms(cfg)]


@pytest.mark.parametrize("builder", ["build_transforms", "build_tta_transforms"])
def test_half_normalisation_is_applied_in_every_path(builder):
    cfg = _cfg(normalize="half", image_size=56)
    tfms = (preprocessing.build_tta_transforms(cfg) if builder == "build_tta_transforms"
            else [preprocessing.build_transforms(cfg, "val")])
    grey = Image.new("RGB", (64, 64), (128, 128, 128))
    for t in tfms:
        x = t(grey)
        # Mid-grey under mean = std = 0.5 is ~0 (exactly 128/255*2 - 1). Read the centre, not the
        # whole image: the two +-10 degree TTA views fill their rotated-in corners with black,
        # which is -1 here, so their whole-image mean is -0.07 with every statistic correct. The
        # centre lies inside the inscribed circle, which a rotation about the centre never leaves.
        centre = x[:, 14:42, 14:42]
        assert torch.allclose(centre, torch.full_like(centre, 128 / 255 * 2 - 1), atol=1e-4)


@pytest.mark.parametrize("over,expected",
                         [({}, IMAGENET), ({"normalize": "imagenet"}, IMAGENET),
                          ({"normalize": "half"}, HALF)],
                         ids=["key-absent", "imagenet", "half"])
def test_every_pipeline_carries_one_normalize_with_the_selected_statistics(over, expected):
    cfg = _cfg(**over)
    # "key-absent" tests the default only while the base config really lacks the key.
    assert ("normalize" in cfg.preprocess) == bool(over)
    # Includes the TRAIN pipeline, whose random augmentation rules out a pixel-value check.
    for pipeline in _pipelines(cfg):
        layers = [t for t in pipeline.transforms if isinstance(t, transforms.Normalize)]
        assert len(layers) == 1
        assert layers[0].mean == pytest.approx(expected[0])
        assert layers[0].std == pytest.approx(expected[1])


def test_default_is_imagenet_and_unknown_value_raises():
    n = preprocessing.normalize_from_cfg(_cfg())
    assert n.mean == pytest.approx(IMAGENET[0]) and n.std == pytest.approx(IMAGENET[1])
    with pytest.raises(ValueError):
        preprocessing.normalize_from_cfg(_cfg(normalize="zscore"))


@pytest.mark.parametrize("bad", ["zscore", "Half", None, 0.5, [0.5, 0.5, 0.5]])
def test_any_value_other_than_the_two_names_raises_value_error(bad):
    cfg = _cfg()
    cfg.preprocess.normalize = bad     # set directly: None / a list cannot ride a key=value string
    with pytest.raises(ValueError, match=re.escape(repr(bad))):
        preprocessing.normalize_from_cfg(cfg)


@pytest.mark.parametrize("build",
                         [lambda c: preprocessing.build_transforms(c, "train"),
                          lambda c: preprocessing.build_transforms(c, "val"),
                          preprocessing.build_tta_transforms],
                         ids=["train", "val", "tta"])
def test_a_misspelt_value_stops_every_builder_instead_of_defaulting(build):
    # A typo in a kernel's config must end the run at start-up, not train for hours under the
    # wrong statistics.
    with pytest.raises(ValueError, match="halv"):
        build(_cfg(normalize="halv"))
