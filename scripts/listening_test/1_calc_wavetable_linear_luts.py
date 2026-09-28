import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch as tr
from sklearn.isotonic import IsotonicRegression
from torch import Tensor as T

from features import SpectralCentroid
from paths import DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def compute_warmth_curve(frame_batch: T, eps: float = 1e-8) -> T:
    """
    warmth = odd harmonic power ratio (excluding DC)
    """
    fft = tr.fft.rfft(frame_batch)
    power = tr.abs(fft) ** 2

    odd_power = power[:, 1::2].sum(dim=1)
    total_power = power[:, 1:].sum(dim=1)

    warmth = odd_power / (total_power + eps)

    return warmth


def compute_richness_curve(frame_batch: T, eps: float = 1e-8) -> T:
    fft = tr.fft.rfft(frame_batch)
    mag = tr.abs(fft)
    power = mag**2

    freqs = tr.linspace(0, 1, power.shape[1], device=power.device)

    total = power.sum(dim=1, keepdim=True) + eps
    centroid = (power * freqs).sum(dim=1, keepdim=True) / total

    spread = tr.sqrt((power * (freqs - centroid) ** 2).sum(dim=1) / total.squeeze(1))

    k = 7.5
    richness = tr.log(spread * (tr.exp(tr.tensor(k)) - 1) + 1) / k

    return richness


def compute_feature_curve(wt: T, dim: str, sr: int = 44100) -> T:
    """Compute the target feature curve for a given wavetable dimension."""
    dim_lower = dim.lower()
    if "brightness" in dim_lower or "centroid" in dim_lower:
        metric = SpectralCentroid(
            sr, window="flat_top", compress=True, floor=1e-4, scaling="kazazis"
        )
        return metric(wt)
    elif "warmth" in dim_lower:
        return compute_warmth_curve(wt)
    elif "richness" in dim_lower:
        return compute_richness_curve(wt)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


def compute_linear_lut(curve: T) -> np.ndarray:
    """Compute a lookup table (LUT) that warps time so the feature progresses linearly."""
    n = curve.shape[0]
    curve_np = curve.detach().cpu().numpy()
    # Compute consecutive differences to check if curve is already monotonically non-decreasing
    diffs = curve_np[1:] - curve_np[:-1]
    if (diffs >= 0).all():
        # Curve is already monotonic, use it directly
        mono_curve = curve_np
    else:
        # Fit isotonic regression to get the closest monotonically non-decreasing approximation
        ir = IsotonicRegression(increasing=True)
        mono_curve = ir.fit_transform(np.arange(n), curve_np)
    # Create n equally spaced target values spanning the monotonic curve's full range
    targets = np.linspace(mono_curve[0], mono_curve[-1], n)
    # Invert the monotonic curve: for each target value, find the frame position where it occurs
    positions = np.interp(targets, mono_curve, np.arange(n))
    # Normalize positions to [0, 1] to get a LUT that warps time so the feature progresses linearly
    lut = positions / (n - 1)

    return tr.from_numpy(lut).float().cpu().numpy()


def calculate_wavetable_luts(
    wavetable_dir: str,
    save_dir: Optional[str] = None,
    sr: int = 44100,
) -> None:
    if save_dir is None:
        save_dir = wavetable_dir

    os.makedirs(save_dir, exist_ok=True)
    wt_paths = sorted(glob.glob(os.path.join(wavetable_dir, "*.pt")))
    log.info(f"Found {len(wt_paths)} wavetables in {wavetable_dir}")

    for wt_path in wt_paths:
        wt_name = os.path.splitext(os.path.basename(wt_path))[0]
        wt = tr.load(wt_path, weights_only=True)
        log.info(f"wt_name: {wt_name}, wt.shape: {wt.shape}")

        dim = wt_name.split("_")[0]
        curve = compute_feature_curve(wt, dim=dim, sr=sr)
        lut = compute_linear_lut(curve)

        lut_path = os.path.join(save_dir, f"{wt_name}__lut.npy")
        np.save(lut_path, lut)
        log.info(f"Saved LUT ({lut.shape}): {lut_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate linear lookup tables (LUTs) for wavetables."
    )
    parser.add_argument(
        "--wavetable-dir",
        default=os.path.join(DATA_DIR, "wavetables"),
        help="Directory containing wavetable .pt files (default: ../../data/wavetables)",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save .npy LUT files (default: same as wavetable-dir)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample rate (default: 44100)",
    )
    args = parser.parse_args()

    wavetable_dir = args.wavetable_dir
    if not os.path.exists(wavetable_dir):
        raise FileNotFoundError(f"Wavetable directory not found: {wavetable_dir}")

    calculate_wavetable_luts(
        wavetable_dir=wavetable_dir,
        save_dir=args.save_dir,
        sr=args.sr,
    )


if __name__ == "__main__":
    main()
