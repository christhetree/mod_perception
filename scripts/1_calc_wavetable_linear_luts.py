"""
Calculate linear lookup tables (LUTs) for wavetable perceptual feature curves.

This script processes wavetable tensors (.pt files), computes the corresponding
perceptual feature curve (e.g. brightness/spectral centroid, warmth, richness)
across the wavetable frames, and derives a monotonic lookup table (LUT) via isotonic
regression and inverse linear interpolation. The resulting LUTs are saved as .npy
files to normalize and linearize perceptual feature traversal during wavetable synthesis.
"""

import argparse
import glob
import logging
import os
from typing import Optional, Union

import numpy as np
import torch as tr
from sklearn.isotonic import IsotonicRegression
from torch import Tensor as T

from features import SpectralCentroid, compute_richness_curve, compute_warmth_curve
from paths import DATA_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def compute_feature_curve(wt: T, dim: str, sr: int = 44100) -> T:
    """Compute the target feature curve for a given wavetable dimension.

    Parameters
    ----------
    wt : T
        Wavetable tensor of shape (n_frames, frame_size) containing individual
        waveform cycles across time/frames.
    dim : str
        The target perceptual dimension name (e.g., 'brightness', 'centroid',
        'warmth', or 'richness').
    sr : int, default=44100
        Sample rate in Hz used for spectral metric calculations.

    Returns
    -------
    T
        1D tensor of shape (n_frames,) representing the computed feature
        value for each frame in the wavetable.

    Raises
    ------
    ValueError
        If the specified dimension is not supported.
    """
    dim_lower = dim.lower()
    if "brightness" in dim_lower or "centroid" in dim_lower:
        metric = SpectralCentroid(
            sample_rate=sr,
            window="flat_top",
            compress=True,
            floor=1e-4,
            scaling="kazazis",
        )
        return metric(wt)
    elif "warmth" in dim_lower:
        return compute_warmth_curve(wt)
    elif "richness" in dim_lower:
        return compute_richness_curve(wt)
    else:
        raise ValueError(
            f"Unsupported dimension: '{dim}'. Expected 'brightness', 'centroid', "
            "'warmth', or 'richness'."
        )


def compute_linear_lut(curve: Union[T, np.ndarray]) -> np.ndarray:
    """Compute a lookup table (LUT) that warps time so the feature progresses linearly.

    Fits an isotonic regression model to ensure a monotonically non-decreasing
    curve, then inverts the mapping via linear interpolation to find the frame
    positions that produce uniformly spaced feature values across [0, 1].

    Parameters
    ----------
    curve : Union[T, np.ndarray]
        1D feature curve values across wavetable frames of shape (n_frames,).

    Returns
    -------
    np.ndarray
        1D array of shape (n_frames,) with dtype float32 containing normalized
        wavetable lookup indices in the range [0.0, 1.0].
    """
    if isinstance(curve, tr.Tensor):
        curve_np = curve.detach().cpu().numpy()
    else:
        curve_np = np.asarray(curve)

    n = curve_np.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    # Compute consecutive differences to check if curve is already monotonically non-decreasing
    diffs = np.diff(curve_np)
    if (diffs >= 0).all():
        # Curve is already monotonic, use it directly
        mono_curve = curve_np
    else:
        # Fit isotonic regression to get the closest monotonically non-decreasing approximation
        ir = IsotonicRegression(increasing=True)
        mono_curve = ir.fit_transform(np.arange(n), curve_np)

    # Check for flat curve edge case
    if mono_curve[0] == mono_curve[-1]:
        return np.linspace(0.0, 1.0, n, dtype=np.float32)

    # Create n equally spaced target values spanning the monotonic curve's full range
    targets = np.linspace(mono_curve[0], mono_curve[-1], n)
    # Invert the monotonic curve: for each target value, find the frame position where it occurs
    positions = np.interp(targets, mono_curve, np.arange(n))
    # Normalize positions to [0, 1] to get a LUT that warps time so the feature progresses linearly
    lut = positions / (n - 1)

    return lut.astype(np.float32)


def calculate_wavetable_luts(
    wavetable_dir: str,
    save_dir: Optional[str] = None,
    sr: int = 44100,
) -> None:
    """Calculate and save linear LUTs for all wavetables in a directory.

    Loads each .pt wavetable file, computes its corresponding feature curve
    based on the dimension in the filename prefix, generates a linearized LUT,
    and saves the output as a NumPy .npy file.

    Parameters
    ----------
    wavetable_dir : str
        Directory containing wavetable .pt files.
    save_dir : Optional[str], default=None
        Directory to save the resulting .npy LUT files. Defaults to `wavetable_dir`.
    sr : int, default=44100
        Sample rate in Hz.
    """
    if save_dir is None:
        save_dir = wavetable_dir

    os.makedirs(save_dir, exist_ok=True)
    wt_paths = sorted(glob.glob(os.path.join(wavetable_dir, "*.pt")))
    log.info(f"Found {len(wt_paths)} wavetables in {wavetable_dir}")

    if not wt_paths:
        log.warning(f"No .pt wavetable files found in {wavetable_dir}")
        return

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
    """Command-line entry point to calculate linear LUTs for wavetables."""
    parser = argparse.ArgumentParser(
        description="Calculate linear lookup tables (LUTs) for wavetables."
    )
    parser.add_argument(
        "--wavetable-dir",
        default=os.path.join(DATA_DIR, "wavetables"),
        help=f"Directory containing wavetable .pt files (default: {DATA_DIR}/wavetables)",
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
