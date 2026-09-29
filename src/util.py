import glob
import logging
import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import torch.nn.functional as F
from torch import Tensor as T, nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ReadOnlyTensorDict(nn.Module):
    def __init__(self, data: Dict[str | int, T], persistent: bool = True):
        super().__init__()
        self.persistent = persistent
        self.keys = set(data.keys())
        for k, v in data.items():
            self.register_buffer(f"tensor_{k}", v, persistent=persistent)

    def __getitem__(self, key: str | int) -> T:
        return self.get_buffer(f"tensor_{key}")

    def __contains__(self, key: str | int) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Iterator[str | int]:
        return iter(self.keys)

    def keys(self) -> Iterator[str | int]:
        return iter(self.keys)

    def values(self) -> Iterator[T]:
        for k in self.keys:
            yield self[k]

    def items(self) -> Iterator[Tuple[str | int, T]]:
        for k in self.keys:
            yield k, self[k]


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
    # peak = np.max(np.abs(normalized_audio))\n    # if peak > 1.0:
    #     normalized_audio = normalized_audio / peak

    return normalized_audio, loudness, gain


def parse_amount(mod_sig: str) -> Tuple[str, float, str]:
    """Split a mod signal name around its last number, which is the amount, e.g.
    "amp_1.00hz_0.10" -> ("amp_1.00hz_", 0.10, "") and
    "freq_0.25hz" -> ("freq_", 0.25, "hz").
    Strips optional trailing __phase_... if present."""
    mod_sig = re.sub(r"__phase_\d+_\d+$", "", mod_sig)
    match = re.match(r"^(.*?)(\d+(?:\.\d+)?)(\D*)$", mod_sig)
    assert match is not None, f"Could not find an amount in {mod_sig}"
    return match.group(1), float(match.group(2)), match.group(3)


def find_variants(
    samples_dir: str, wt_name: str, mod_sig: str, suffix: str = ""
) -> List[str]:
    """Find all samples of wt_name whose mod signal matches mod_sig apart from
    its amount, including mod_sig itself so that the trivial self distance is
    also measured (not every distance function is guaranteed to return 0).
    Supports both unphased and phase-shifted variants."""
    prefix, _, unit = parse_amount(mod_sig)
    clean_suffix = suffix[:-4] if suffix.endswith(".wav") else suffix
    pattern = os.path.join(samples_dir, f"{wt_name}__{prefix}*{unit}*.wav")
    paths = []
    for path in sorted(glob.glob(pattern)):
        filename = os.path.basename(path)
        name = filename[:-4] if filename.endswith(".wav") else filename
        name = re.sub(r"__phase_\d+_\d+$", "", name)
        if clean_suffix and name.endswith(clean_suffix):
            name = name[: -len(clean_suffix)]
        # Robustly strip any trailing LUFS normalization tag (e.g. _-18lufs or _-18.0lufs)
        name = re.sub(r"_[+-]?\d+(?:\.\d+)?lufs$", "", name)
        variant = name[len(f"{wt_name}__") :]
        try:
            variant_prefix, _, variant_unit = parse_amount(variant)
        except AssertionError:
            continue
        if (variant_prefix, variant_unit) != (prefix, unit):
            continue
        paths.append(path)

    def _sort_key(p: str) -> Tuple[float, int]:
        fname = os.path.basename(p)
        pm = re.search(r"__phase_(\d+)_\d+", fname)
        pidx = int(pm.group(1)) if pm else 0
        cname = re.sub(r"__phase_\d+_\d+", "", fname)
        if clean_suffix and cname.endswith(f"{clean_suffix}.wav"):
            cname = cname[: -len(f"{clean_suffix}.wav")]
        elif cname.endswith(".wav"):
            cname = cname[:-4]
        cname = re.sub(r"_[+-]?\d+(?:\.\d+)?lufs$", "", cname)
        cname = cname[len(f"{wt_name}__") :]
        try:
            _, amt, _ = parse_amount(cname)
        except AssertionError:
            amt = 0.0
        return (amt, pidx)

    paths.sort(key=_sort_key)
    return paths


def format_sig_figs(val: float, precision: int = 3) -> str:
    """Format a float to a fixed number of significant figures, padding trailing zeros."""
    if val is None or pd.isna(val):
        return ""
    if val >= 1.0 or val >= 0.9995:
        return f"{1.0:.{precision}f}"
    s = f"{val:.{precision}g}"
    if "e" in s or "E" in s:
        return f"{val:.{precision}f}"
    parts = s.split(".")
    if len(parts) == 1:
        needed = precision - len(parts[0])
        return parts[0] + "." + "0" * max(0, needed)
    sig_digits = (
        len(parts[1].lstrip("0"))
        if parts[0] == "0"
        else len(parts[0].lstrip("0")) + len(parts[1])
    )
    needed = precision - sig_digits
    if needed > 0:
        s += "0" * needed
    return s
