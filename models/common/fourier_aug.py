"""Fourier amplitude swap (Amendment 5b, arm I): a batch-level style augmentation.

The lowest spatial frequencies of a fundus photograph carry its global colour and illumination,
i.e. the camera and the hospital, while the phase and the higher frequencies carry the vessels and
the ridge. For each chosen image, the centred ``block x block`` window of the (shifted) 2-D
amplitude spectrum is replaced by that of a random partner in the same batch, the phase is kept,
and the image is rebuilt. The trainer applies it to the normalised batch on the device, so it is a
function of a batch tensor, not a per-image transform.

``block`` must be odd: only a window centred on DC is symmetric under k -> -k, so only then does a
real image's spectrum stay conjugate-symmetric after the swap. With an even block the rebuilt image
has an imaginary part, and dropping it (``.real``) would change frequencies OUTSIDE the window too.

The zero-frequency (DC) term is copied whole from the partner (its complex value), not as an
amplitude with the image's own phase. On the normalised batch the DC term is (X(0) - N mu) / sigma,
a real number whose phase is 0 or pi with the sign of the centred channel mean, so "partner's
amplitude, own phase" would give sign(mean_i) * |mean_j|: a mirrored colour cast whenever the two
signs differ. Copying the partner's DC value and the amplitudes of the other ``block**2 - 1``
entries equals the amplitude swap done in pixel space (where every channel mean is positive, so
the DC phase is always 0) and then normalised, exactly (PROTOCOL_V2, Amendment 5 pre-run note).
"""
from __future__ import annotations

import torch


def fourier_swap_batch(x: torch.Tensor, block: int, p: float,
                       generator: torch.Generator | None = None, return_draw: bool = False):
    """A new B x C x H x W tensor: each image, with probability ``p``, takes the centred
    ``block x block`` low-frequency amplitudes of a random batch partner (a random permutation's
    entry; a partner that is the image itself leaves it unchanged), keeping its own phase, except
    at DC, where it takes the partner's complex value (see the module docstring).

    Images that are not swapped are copied exactly; ``x`` is never modified. Shape, dtype and device
    are preserved; the FFT runs in float32. ``generator`` seeds the draws, which happen on the
    generator's device (CPU when None) and are moved to ``x``'s. ``return_draw=True`` returns
    ``(out, rows, partners)`` instead: the indices of the swapped images and of their partners
    (``out[rows[n]]`` took ``x[partners[n]]``'s block), on ``x``'s device; the draws are the same.
    """
    if x.dim() != 4:
        raise ValueError(f"fourier_swap_batch expects B x C x H x W, got shape {tuple(x.shape)}")
    b, _, h, w = x.shape
    if not (isinstance(block, int) and block >= 1 and block % 2 == 1 and block <= min(h, w)):
        raise ValueError(f"block must be an odd int in [1, {min(h, w)}], got {block!r}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {p!r}")
    out = x.clone()
    if p == 0.0 or b < 2:
        return _result(out, None, None, return_draw)

    dev = generator.device if generator is not None else torch.device("cpu")
    chosen = torch.rand(b, generator=generator, device=dev) < p
    partner = torch.randperm(b, generator=generator, device=dev)
    rows = torch.nonzero(chosen & (partner != torch.arange(b, device=dev))).flatten()
    if rows.numel() == 0:
        return _result(out, None, None, return_draw)
    partner = partner[rows].to(x.device)
    rows = rows.to(x.device)

    dims = (-2, -1)
    spec = torch.fft.fftshift(torch.fft.fft2(x.float(), dim=dims), dim=dims)
    r0, c0 = h // 2 - block // 2, w // 2 - block // 2       # fftshift puts DC at (h//2, w//2)
    mine = spec[rows]                                        # advanced indexing: copies
    win = mine[..., r0:r0 + block, c0:c0 + block]            # a view into ``mine``
    theirs = spec[partner, ..., r0:r0 + block, c0:c0 + block]
    swapped = torch.polar(theirs.abs(), win.angle())
    mid = block // 2                                         # DC inside the block
    swapped[..., mid, mid] = theirs[..., mid, mid]           # the partner's complex DC term
    win.copy_(swapped)
    rebuilt = torch.fft.ifft2(torch.fft.ifftshift(mine, dim=dims), dim=dims).real
    out[rows] = rebuilt.to(x.dtype)
    return _result(out, rows, partner, return_draw)


def _result(out, rows, partners, return_draw):
    if not return_draw:
        return out
    if rows is None:                                         # nothing swapped
        rows = partners = torch.zeros(0, dtype=torch.long, device=out.device)
    return out, rows, partners
