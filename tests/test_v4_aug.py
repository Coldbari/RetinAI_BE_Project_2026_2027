"""Task 4 of the v4 plan: arm I's augmentation (Amendment 5b).

Two kinds of change. Per-image ops in ``build_transforms``, read from the ``train.aug_v4`` block:
rotation +-180, RandomResizedCrop scale (0.85, 1.0), gamma U(0.7, 1.4) with p 0.5, Gaussian pixel
noise sigma 0.02 with p 0.3. And a batch op, ``fourier_swap_batch``, which the trainer applies to
the normalised GPU batch: the centred 3x3 low-frequency amplitude block of each chosen image is
taken from a random batch partner, its phase is kept, except at DC, which is the partner's complex
value (so the swap on the normalised batch equals the swap on [0, 1] pixels, then normalised).

Arm B has no ``aug_v4`` block, and its pipeline must stay exactly v3's: the last test pins it
against a literal v3 pipeline, not against the code under test.
"""
import pickle

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from pathlib import Path  # noqa: E402

from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from models.common import preprocessing  # noqa: E402
from models.common.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B_CFG = ROOT / "configs" / "rop_staging_dinov2.yaml"

# The spec's values (Amendment 5b), typed as literals on purpose.
V4_BLOCK = {"enabled": True, "rotation": 180, "min_crop_scale": 0.85, "gamma": [0.7, 1.4],
            "gamma_p": 0.5, "noise_sigma": 0.02, "noise_p": 0.3, "fourier_block": 3,
            "fourier_p": 0.5}


def _cfg(tmp_path, block=None, **preprocess):
    """Arm B's config, with ``train.aug_v4 = block`` when given, written to and loaded from YAML."""
    data = yaml.safe_load(B_CFG.read_text())
    data["preprocess"].update(preprocess)
    if block is not None:
        data["train"]["aug_v4"] = block
    path = tmp_path / f"cfg_{len(list(tmp_path.iterdir()))}.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path)


def _ops(pipeline):
    return list(pipeline.transforms)


def _index(ops, kind):
    found = [i for i, t in enumerate(ops) if isinstance(t, kind)]
    assert len(found) == 1, f"expected exactly one {kind.__name__}, found {len(found)}"
    return found[0]


def _shifted_spectrum(x):
    return torch.fft.fftshift(torch.fft.fft2(x.double(), dim=(-2, -1)), dim=(-2, -1))


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


# ── Fourier amplitude swap ───────────────────────────────────────────────────
@pytest.mark.parametrize("device", _devices())
def test_fourier_swap_is_identity_at_p0_and_preserves_shape_dtype_device(device):
    from models.common.fourier_aug import fourier_swap_batch
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(6, 3, 20, 18, generator=torch.Generator().manual_seed(0)).to(device, dtype)
        before = x.clone()
        out = fourier_swap_batch(x, block=3, p=0.0, generator=torch.Generator().manual_seed(1))
        assert out is not x and torch.equal(out, x)                # p == 0: exactly the input
        for p in (0.5, 1.0):
            # A CPU generator with a batch on another device: draws happen on CPU, indices move.
            out = fourier_swap_batch(x, block=3, p=p, generator=torch.Generator().manual_seed(2))
            assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
            assert torch.isfinite(out.float()).all()
        assert torch.equal(x, before)                              # never modified in place
    # No generator: the global RNG, still shape/dtype/device preserving.
    out = fourier_swap_batch(x, block=3, p=1.0)
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device


def test_fourier_swap_changes_only_low_frequencies():
    from models.common.fourier_aug import fourier_swap_batch
    b, h, w, block = 8, 24, 20, 3
    x = torch.randn(b, 3, h, w, generator=torch.Generator().manual_seed(0))
    out = fourier_swap_batch(x, block=block, p=1.0, generator=torch.Generator().manual_seed(3))

    s_in, s_out = _shifted_spectrum(x), _shifted_spectrum(out)
    a_in, a_out = s_in.abs(), s_out.abs()
    r0, c0 = h // 2 - block // 2, w // 2 - block // 2
    inside = torch.zeros(h, w, dtype=torch.bool)
    inside[r0:r0 + block, c0:c0 + block] = True
    ac = inside.clone()                              # the block less its centre, DC
    ac[h // 2, w // 2] = False

    # Outside the centred block every amplitude is the input's (1e-4 of the largest amplitude).
    scale = a_in.max()
    assert ((a_out - a_in)[..., ~inside].abs().max() / scale) < 1e-4

    swapped = 0
    for i in range(b):
        # The block's amplitudes are image j's for exactly one j in the batch ...
        dist = torch.stack([(a_out[i][..., inside] - a_in[j][..., inside]).abs().max()
                            for j in range(b)])
        j = int(dist.argmin())
        assert dist[j] / scale < 1e-4
        if j == i:                                   # not swapped (partner was itself): untouched
            assert torch.equal(out[i], x[i])
            continue
        swapped += 1
        assert (a_out[i][..., inside] - a_in[i][..., inside]).abs().max() / scale > 1e-2
        # ... the phase is image i's own off DC: out = |X_j| e^{i angle X_i}, so
        # out * |X_i| == X_i * |X_j| there ...
        lhs = s_out[i][..., ac] * a_in[i][..., ac]
        rhs = s_in[i][..., ac] * a_in[j][..., ac]
        assert ((lhs - rhs).abs().max() / scale ** 2) < 1e-4
        # ... and DC is image j's complex value, sign included.
        assert ((s_out[i][..., h // 2, w // 2] - s_in[j][..., h // 2, w // 2]).abs().max()
                / scale) < 1e-4
    assert swapped >= b // 2                         # p = 1: most images get a real partner


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _normalize(x):
    return (x - IMAGENET_MEAN.to(x.dtype)) / IMAGENET_STD.to(x.dtype)


def _pixel_space_swap(x, rows, partners, block):
    """The amplitude swap where it is defined, on [0, 1] pixels, in float64: image i keeps its own
    phase and takes image j's centred ``block x block`` amplitudes. Every pixel-space channel mean
    is positive, so DC has phase 0 and a plain amplitude swap there is the partner's DC value."""
    h, w = x.shape[-2:]
    spec = _shifted_spectrum(x)
    r0, c0 = h // 2 - block // 2, w // 2 - block // 2
    win = (..., slice(r0, r0 + block), slice(c0, c0 + block))
    out = x.double().clone()
    for i, j in zip(rows.tolist(), partners.tolist()):
        s = spec[i].clone()
        s[win] = torch.polar(spec[j][win].abs(), spec[i][win].angle())
        out[i] = torch.fft.ifft2(torch.fft.ifftshift(s, dim=(-2, -1)), dim=(-2, -1)).real
    return out


@pytest.mark.parametrize("h,w", [(32, 32), (33, 30)])
def test_fourier_swap_on_the_normalised_batch_equals_the_pixel_space_swap(h, w):
    from models.common.fourier_aug import fourier_swap_batch
    b, block = 12, 3
    # Photos from dark to bright, so the normalised channel means take both signs.
    level = torch.linspace(0.1, 0.9, b).view(b, 1, 1, 1)
    x = (level + 0.1 * torch.randn(b, 3, h, w, generator=torch.Generator().manual_seed(0))).clamp(0, 1)
    xn = _normalize(x)
    out, rows, partners = fourier_swap_batch(xn, block=block, p=1.0, return_draw=True,
                                             generator=torch.Generator().manual_seed(5))
    same = fourier_swap_batch(xn, block=block, p=1.0, generator=torch.Generator().manual_seed(5))
    assert torch.equal(out, same)                                  # the flag changes no draw
    assert rows.numel() >= b // 2 and not (rows == partners).any()

    ref = _normalize(_pixel_space_swap(x, rows, partners, block))
    for i, j in zip(rows.tolist(), partners.tolist()):
        assert (out[i].double() - ref[i]).abs().max() < 1e-4, (i, j)
    kept = [i for i in range(b) if i not in set(rows.tolist())]
    assert torch.equal(out[kept], xn[kept])                        # unswapped: copied exactly
    # The pairs the DC rule is for: some channel's normalised mean has opposite signs in i and j
    # (the old "partner's amplitude, own phase" DC gave sign(mean_i) * |mean_j| there).
    m = xn.mean(dim=(-2, -1))
    assert any(((m[i] * m[j] < 0) & (m[j].abs() > 1e-2)).any()
               for i, j in zip(rows.tolist(), partners.tolist()))

    # Nothing swapped: empty draws, a plain copy.
    out0, r0_, p0_ = fourier_swap_batch(xn, block=block, p=0.0, return_draw=True)
    assert torch.equal(out0, xn) and r0_.numel() == 0 and p0_.numel() == 0


def test_fourier_swap_rejects_a_block_without_a_centre():
    from models.common.fourier_aug import fourier_swap_batch
    x = torch.randn(4, 3, 16, 16)
    # An even block is not symmetric about DC: the swapped spectrum is no longer conjugate-
    # symmetric, and ``.real`` would silently change frequencies outside the block.
    for bad in ({"block": 2, "p": 0.5}, {"block": 0, "p": 0.5}, {"block": 17, "p": 0.5},
                {"block": 3, "p": 1.5}, {"block": 3, "p": -0.1}):
        with pytest.raises(ValueError):
            fourier_swap_batch(x, **bad)


# ── build_transforms with and without the block ──────────────────────────────
def test_build_transforms_v4_uses_rotation_180_and_crop_085(tmp_path):
    cfg = _cfg(tmp_path, V4_BLOCK)
    assert preprocessing.aug_v4_settings(cfg) == V4_BLOCK
    ops = _ops(preprocessing.build_transforms(cfg, "train"))

    assert ops[_index(ops, transforms.RandomRotation)].degrees == [-180, 180]
    crop = ops[_index(ops, transforms.RandomResizedCrop)]
    assert crop.scale == (0.85, 1.0) and crop.ratio == (0.95, 1.05)

    # Gamma right after the last ColorJitter (PIL); noise on the [0, 1] tensor, before Normalize.
    gamma = ops[_index(ops, preprocessing.GammaJitter)]
    last_jitter = max(i for i, t in enumerate(ops) if isinstance(t, transforms.ColorJitter))
    assert ops.index(gamma) == last_jitter + 1
    assert (gamma.lo, gamma.hi, gamma.p) == (0.7, 1.4, 0.5)
    noise = ops[_index(ops, preprocessing.GaussianNoise)]
    assert (noise.sigma, noise.p) == (0.02, 0.3)
    to_tensor, norm = _index(ops, transforms.ToTensor), _index(ops, transforms.Normalize)
    assert to_tensor + 1 == ops.index(noise) == norm - 1

    # The Fourier swap is a batch op for the trainer, not a per-image transform.
    assert not any("fourier" in type(t).__name__.lower() for t in ops)
    # Validation is untouched by the block.
    assert repr(preprocessing.build_transforms(cfg, "val")) == \
        repr(preprocessing.build_transforms(_cfg(tmp_path), "val"))
    # The whole pipeline runs end to end on an RGB image (small grid to stay fast).
    small = _cfg(tmp_path, V4_BLOCK, image_size=56)
    img = Image.fromarray(torch.randint(0, 256, (80, 90, 3), dtype=torch.uint8).numpy())
    x = preprocessing.build_transforms(small, "train")(img)
    assert x.shape == (3, 56, 56) and x.dtype == torch.float32


def test_build_transforms_without_aug_v4_is_unchanged_from_v3(tmp_path):
    size = 392
    # v3's train pipeline for arm B, written out by hand (site_randomization and strong_aug on).
    v3 = [preprocessing.FundusPreprocess(circle=True, ben=False, ben_sigma=10.0, do_clahe=True,
                                         letterbox=True),
          transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.95, 1.05)),
          transforms.RandomHorizontalFlip(),
          transforms.RandomVerticalFlip(),
          transforms.RandomRotation(15),
          transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
          transforms.ColorJitter(hue=0.05, saturation=0.3),
          transforms.RandomGrayscale(p=0.10),
          preprocessing.ResolutionJitter(p=0.5, min_frac=0.33),
          transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.88, 1.12), shear=8),
          transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
          transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
          transforms.RandomErasing(p=0.35, scale=(0.02, 0.15))]
    expected = [repr(t) for t in v3]
    for block in (None, {**V4_BLOCK, "enabled": False}):
        cfg = _cfg(tmp_path, block)
        assert preprocessing.aug_v4_settings(cfg) is None
        assert [repr(t) for t in _ops(preprocessing.build_transforms(cfg, "train"))] == expected


# ── the block's validation and the two new per-image ops ─────────────────────
def test_settings_refuse_a_malformed_block(tmp_path):
    for bad in ({"enabled": "yes"},                    # a string is not a boolean
                {"gamma": 0.7},                        # gamma is a [lo, hi] pair
                {"gamma": [1.4, 0.7]},
                {"min_crop_scale": 0.0},
                {"noise_p": 1.5},
                {"fourier_block": 2},                  # even: no centre (see the swap test)
                {"rotaton": 180}):                     # misspelt key, nothing would read it
        block = {**V4_BLOCK, **bad}
        if "rotaton" in bad:
            del block["rotation"]
        with pytest.raises(ValueError):
            preprocessing.aug_v4_settings(_cfg(tmp_path, block))
    # min_crop_scale lives in site randomisation's RandomResizedCrop: without it, refuse.
    cfg = _cfg(tmp_path, V4_BLOCK)
    cfg.train["site_randomization"] = False
    with pytest.raises(ValueError):
        preprocessing.build_transforms(cfg, "train")


def test_gamma_and_noise_ops():
    grey = Image.new("RGB", (8, 8), (128, 128, 128))
    out = preprocessing.GammaJitter(2.0, 2.0, p=1.0)(grey)
    assert out.mode == "RGB" and out.size == grey.size
    assert out.getpixel((3, 3)) == (round(255 * (128 / 255) ** 2),) * 3       # 64
    assert preprocessing.GammaJitter(0.7, 1.4, p=0.0)(grey) is grey

    t = torch.full((3, 8, 8), 0.99)
    assert torch.equal(preprocessing.GaussianNoise(0.5, p=0.0)(t), t)
    noisy = preprocessing.GaussianNoise(0.5, p=1.0)(t)
    assert not torch.equal(noisy, t) and noisy.min() >= 0 and noisy.max() <= 1

    for op in (preprocessing.GammaJitter(0.7, 1.4, 0.5), preprocessing.GaussianNoise(0.02, 0.3)):
        assert repr(pickle.loads(pickle.dumps(op))) == repr(op)               # DataLoader workers
