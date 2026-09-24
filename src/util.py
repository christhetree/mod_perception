import logging
import os
from typing import Optional, Tuple, List

import numpy as np
import pyloudnorm as pyln
import torch as tr
import torch.nn.functional as F
from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def linear_interpolate_last_dim(x: T, n: int, align_corners: bool = True) -> T:
    n_dim = x.ndim
    assert 1 <= n_dim <= 3
    if x.size(-1) == n:
        return x
    if n_dim == 1:
        x = x.view(1, 1, -1)
    elif n_dim == 2:
        x = x.unsqueeze(1)
    x = F.interpolate(x, n, mode="linear", align_corners=align_corners)
    if n_dim == 1:
        x = x.view(-1)
    elif n_dim == 2:
        x = x.squeeze(1)
    return x


def create_wavetable_sweep(
    wt: T,
    sr: int = 44100,
    duration: float = 4.0,
    mod_signal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Render audio by sweeping through wavetable frames.

    wt: tensor [num_frames, frame_length]
    sr: sample rate
    duration: output duration in seconds
    mod_signal: optional modulation signal of length total_samples, values
        in [0, 1] mapped to [0, num_frames-1]. If None, a linear sweep is used.
    """
    wt = wt.detach().cpu().numpy()

    num_frames, frame_len = wt.shape
    total_samples = int(sr * duration)

    if mod_signal is None:
        frame_positions = np.linspace(0, num_frames - 1, total_samples)
    else:
        frame_positions = np.clip(mod_signal, 0.0, 1.0) * (num_frames - 1)

    output = np.zeros(total_samples)

    for i, pos in enumerate(frame_positions):
        idx_low = int(np.floor(pos))
        idx_high = min(idx_low + 1, num_frames - 1)
        frac = pos - idx_low

        # Linear interpolation between frames
        frame = (1 - frac) * wt[idx_low] + frac * wt[idx_high]

        # Wrap inside frame length
        sample_index = i % frame_len
        output[i] = frame[sample_index]

    # Normalize to avoid clipping
    # output /= np.max(np.abs(output) + 1e-8)
    if np.abs(output).max() > 1.0:
        log.warning("wavetable sweep is clipping")

    return output


def loudness_normalize(
    audio: np.ndarray, sr: int, target_lufs: float = -16
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize audio to target LUFS.

    audio: numpy array
    sr: sample rate
    target_lufs: desired loudness (e.g. -16, -14, -12)
    """

    meter = pyln.Meter(sr)  # ITU-R BS.1770

    loudness = meter.integrated_loudness(audio)

    # Compute gain
    gain = target_lufs - loudness

    # Apply gain
    normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)

    # Optional: clipping protection
    # peak = np.max(np.abs(normalized_audio))
    # if peak > 1.0:
    #     normalized_audio = normalized_audio / peak

    return normalized_audio, loudness, gain


def make_mod_signal(
    n_samples: int,
    sr: float,
    freq: float,
    phase: float = 0.0,
    shape: str = "cos",
    exp: float = 1.0,
) -> T:
    assert n_samples > 0
    assert 0.0 < freq < sr / 2.0
    assert -2 * tr.pi <= phase <= 2 * tr.pi
    assert shape in {"cos", "rect_cos", "inv_rect_cos", "tri", "saw", "rsaw", "sqr"}
    if shape in {"rect_cos", "inv_rect_cos"}:
        # Rectified sine waves have double the frequency
        freq /= 2.0
        phase /= 2.0
    assert exp > 0
    argument = tr.cumsum(2 * tr.pi * tr.full((n_samples,), freq) / sr, dim=0) + phase
    saw = tr.remainder(argument, 2 * tr.pi) / (2 * tr.pi)

    if shape == "cos":
        mod_sig = (tr.cos(argument + tr.pi) + 1.0) / 2.0
    elif shape == "rect_cos":
        mod_sig = tr.abs(tr.cos(argument + (tr.pi / 2.0)))
    elif shape == "inv_rect_cos":
        mod_sig = -tr.abs(tr.cos(argument)) + 1.0
    elif shape == "sqr":
        cos = tr.cos(argument + tr.pi)
        sqr = tr.sign(cos)
        mod_sig = (sqr + 1.0) / 2.0
    elif shape == "saw":
        mod_sig = saw
    elif shape == "rsaw":
        # mod_sig = tr.roll(1.0 - saw, 1)  # TODO(cm)
        mod_sig = 1.0 - saw
    elif shape == "tri":
        tri = 2 * saw
        mod_sig = tr.where(tri > 1.0, 2.0 - tri, tri)
    else:
        raise ValueError("Unsupported shape")

    if exp != 1.0:
        mod_sig = mod_sig**exp
    return mod_sig


def find_corners(mod_sig: T) -> Tuple[T, T]:
    assert mod_sig.ndim == 2
    m_r = mod_sig[:, 1:]
    m_l = mod_sig[:, :-1]
    diff = m_r - m_l
    diff_r = diff[:, 1:]
    diff_l = diff[:, :-1]
    diff_pos_l = (diff_l > 0) * diff_l
    diff_neg_l = (diff_l < 0) * diff_l
    top_corners = diff_pos_l * (diff_r + 1e-16)
    top_corners = -tr.floor(top_corners).long()
    bottom_corners = diff_neg_l * (diff_r + 1e-16)
    bottom_corners = -tr.floor(bottom_corners).long()
    tmp = tr.zeros_like(mod_sig)
    tmp[:, 1:-1] = top_corners
    top_corners = tmp
    tmp = tr.zeros_like(mod_sig)
    tmp[:, 1:-1] = bottom_corners
    bottom_corners = tmp
    return top_corners, bottom_corners


def make_quasi_periodic(
    mod_sig: T,
    randomness: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[T, List[float]]:
    assert mod_sig.ndim == 1
    orig_size = mod_sig.size(0)

    top_corners, bottom_corners = find_corners(mod_sig.unsqueeze(0))
    if top_corners.sum() > bottom_corners.sum():
        corners = top_corners
    else:
        corners = bottom_corners
    corners = corners.squeeze(0)
    corner_indices = (corners == 1).nonzero(as_tuple=True)[0]
    corner_indices = [c.item() for c in corner_indices]
    if len(corner_indices) < 2:
        return mod_sig, []

    boundaries = [0] + corner_indices + [orig_size - 1]
    gaps = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]

    gen = tr.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    n_gaps = len(gaps)
    assert n_gaps % 2 == 0
    half = n_gaps // 2
    scales = [1.0 + randomness] * half + [1.0 - randomness] * half
    perm = tr.randperm(n_gaps, generator=gen).tolist()
    scales = [scales[p] for p in perm]
    scaled_gaps = [g * s for g, s in zip(gaps, scales)]
    total_scaled = sum(scaled_gaps)
    total_orig = sum(gaps)
    norm_gaps = [g / total_scaled for g in scaled_gaps]

    new_boundaries = [0]
    for g in norm_gaps:
        new_boundaries.append(new_boundaries[-1] + g * total_orig)
    new_boundaries = [int(round(b)) for b in new_boundaries]
    new_boundaries[-1] = orig_size - 1

    sections = []
    for i in range(len(boundaries) - 1):
        old_start, old_end = boundaries[i], boundaries[i + 1]
        section = mod_sig[old_start : old_end + 1]
        new_len = new_boundaries[i + 1] - new_boundaries[i] + 1
        new_section = linear_interpolate_last_dim(
            section, max(2, new_len), align_corners=True
        )
        if i < len(boundaries) - 2:
            new_section = new_section[:-1]
        sections.append(new_section)

    new_mod_sig = tr.cat(sections, dim=0)
    new_mod_sig = new_mod_sig[:orig_size]
    return new_mod_sig, norm_gaps
