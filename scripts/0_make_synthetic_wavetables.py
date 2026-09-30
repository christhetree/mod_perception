"""Generate synthetic wavetables with isolated perceptual feature sweeps.

This script constructs three synthetic wavetables (brightness, warmth, and richness)
in the frequency domain with controlled Gaussian spectral envelopes, harmonic power
distributions, and fixed random phase spectra. The resulting wavetable tensors are
saved as .pt files to the output directory.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch as tr
from torch import Tensor as T

from paths import DATA_DIR

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def create_synthetic_wavetable(
    n_pos: int = 256,
    n_samples: int = 1024,
    centroids: Optional[np.ndarray] = None,
    sigmas: Optional[np.ndarray] = None,
    warmths: Optional[np.ndarray] = None,
    centroid_correction: bool = False,
    seed: int = 42,
) -> T:
    """Create a synthetic wavetable by synthesizing frames in the frequency domain.

    Constructs each single-cycle frame using a Gaussian spectral envelope with
    per-position control over centroid position, spectral width (sigma), and
    odd-to-even harmonic warmth power ratio. Fixed random phases are shared across
    all wavetable frames to ensure phase consistency.

    Parameters
    ----------
    n_pos : int, default=256
        Number of wavetable positions/frames.
    n_samples : int, default=1024
        Number of audio samples per single-cycle waveform frame.
    centroids : Optional[np.ndarray], default=None
        Center bin index for Gaussian envelope per position of shape (n_pos,).
    sigmas : Optional[np.ndarray], default=None
        Gaussian standard deviation width (in FFT bins) per position of shape (n_pos,).
    warmths : Optional[np.ndarray], default=None
        Target odd harmonic power ratio per position of shape (n_pos,).
    centroid_correction : bool, default=False
        Whether to iteratively adjust Gaussian center to correct for DC bin clipping.
    seed : int, default=42
        Random seed for initial harmonic phase spectrum.

    Returns
    -------
    T
        Synthesized wavetable tensor of shape (n_pos, n_samples) with dtype float32,
        peak-normalized to [-1.0, 1.0].
    """
    n_bins = n_samples // 2 + 1
    bins = np.arange(n_bins, dtype=np.float64)

    if centroids is None:
        centroids = np.full(n_pos, 40.0)
    if sigmas is None:
        sigmas = np.full(n_pos, 20.0)
    if warmths is None:
        warmths = np.full(n_pos, 0.5)

    rng = np.random.RandomState(seed)
    phases = rng.uniform(0, 2 * np.pi, n_bins)
    phases[0] = 0.0

    odd_mask = np.zeros(n_bins, dtype=bool)
    even_mask = np.zeros(n_bins, dtype=bool)
    odd_mask[1::2] = True
    even_mask[2::2] = True

    frames = np.zeros((n_pos, n_samples))
    for p in range(n_pos):
        center = centroids[p]
        if centroid_correction:
            for _ in range(10):
                envelope = np.exp(-0.5 * ((bins - center) / sigmas[p]) ** 2)
                envelope[0] = 0.0
                power = envelope**2
                actual = (bins * power).sum() / (power.sum() + 1e-12)
                center += centroids[p] - actual

        envelope = np.exp(-0.5 * ((bins - center) / sigmas[p]) ** 2)
        envelope[0] = 0.0

        odd_power = (envelope[odd_mask] ** 2).sum()
        even_power = (envelope[even_mask] ** 2).sum()

        # Scale even harmonics to enforce warmth: odd / (odd + scale^2 * even) = w
        w = warmths[p]
        if even_power > 0 and odd_power > 0:
            even_scale = np.sqrt(odd_power * (1.0 - w) / (w * even_power))
        else:
            even_scale = 1.0

        amplitudes = envelope.copy()
        amplitudes[even_mask] *= even_scale

        spectrum = amplitudes * np.exp(1j * phases)
        frame = np.fft.irfft(spectrum, n=n_samples)

        peak = np.abs(frame).max()
        if peak > 0:
            frame /= peak
        frames[p] = frame

    return tr.from_numpy(frames).float()


def create_synthetic_centroid_sweep(
    n_pos: int = 256,
    n_samples: int = 1024,
    centroid_range: Tuple[float, float] = (0.4, 16.5),
    sigma: float = 2.5,
    target_warmth: float = 0.5,
    seed: int = 42,
) -> T:
    """Create a synthetic brightness (spectral centroid) sweep wavetable.

    Parameters
    ----------
    n_pos : int, default=256
        Number of wavetable positions/frames.
    n_samples : int, default=1024
        Number of samples per frame.
    centroid_range : Tuple[float, float], default=(0.4, 16.5)
        Geometrically spaced centroid range across frames.
    sigma : float, default=2.5
        Gaussian envelope spectral width.
    target_warmth : float, default=0.5
        Constant warmth ratio across frames.
    seed : int, default=42
        Random seed for harmonic phases.

    Returns
    -------
    T
        Wavetable tensor of shape (n_pos, n_samples).
    """
    return create_synthetic_wavetable(
        n_pos=n_pos,
        n_samples=n_samples,
        centroids=np.geomspace(centroid_range[0], centroid_range[1], n_pos),
        sigmas=np.full(n_pos, sigma),
        warmths=np.full(n_pos, target_warmth),
        seed=seed,
    )


create_synthetic_brightness_sweep = create_synthetic_centroid_sweep


def create_synthetic_warmth_sweep(
    n_pos: int = 256,
    n_samples: int = 1024,
    target_centroid: float = 36.0,
    sigma: float = 80.0,
    warmth_range: Tuple[float, float] = (0.001, 0.999),
    seed: int = 42,
) -> T:
    """Create a synthetic warmth (odd harmonic power ratio) sweep wavetable.

    Parameters
    ----------
    n_pos : int, default=256
        Number of wavetable positions/frames.
    n_samples : int, default=1024
        Number of samples per frame.
    target_centroid : float, default=36.0
        Constant centroid bin index across frames.
    sigma : float, default=80.0
        Gaussian envelope spectral width.
    warmth_range : Tuple[float, float], default=(0.001, 0.999)
        Linearly spaced warmth range across frames.
    seed : int, default=42
        Random seed for harmonic phases.

    Returns
    -------
    T
        Wavetable tensor of shape (n_pos, n_samples).
    """
    return create_synthetic_wavetable(
        n_pos=n_pos,
        n_samples=n_samples,
        centroids=np.full(n_pos, target_centroid),
        sigmas=np.full(n_pos, sigma),
        warmths=np.linspace(warmth_range[0], warmth_range[1], n_pos),
        seed=seed,
    )


def create_synthetic_richness_sweep(
    n_pos: int = 256,
    n_samples: int = 1024,
    target_centroid: float = 11.3,
    sigma_range: Tuple[float, float] = (1.0, 9.8),
    target_warmth: float = 0.5,
    seed: int = 42,
) -> T:
    """Create a synthetic richness (spectral spread) sweep wavetable.

    Parameters
    ----------
    n_pos : int, default=256
        Number of wavetable positions/frames.
    n_samples : int, default=1024
        Number of samples per frame.
    target_centroid : float, default=11.3
        Constant centroid bin index across frames.
    sigma_range : Tuple[float, float], default=(1.0, 9.8)
        Geometrically spaced spectral spread range across frames.
    target_warmth : float, default=0.5
        Constant warmth ratio across frames.
    seed : int, default=42
        Random seed for harmonic phases.

    Returns
    -------
    T
        Wavetable tensor of shape (n_pos, n_samples).
    """
    return create_synthetic_wavetable(
        n_pos=n_pos,
        n_samples=n_samples,
        centroids=np.full(n_pos, target_centroid),
        sigmas=np.geomspace(sigma_range[0], sigma_range[1], n_pos),
        warmths=np.full(n_pos, target_warmth),
        centroid_correction=True,
        seed=seed,
    )


def make_synthetic_wavetables(
    save_dir: str,
    n_pos: int = 256,
    n_samples: int = 1024,
    seed: int = 42,
) -> dict[str, T]:
    """Generate and save the 3 synthetic wavetables to disk.

    Parameters
    ----------
    save_dir : str
        Directory to save generated .pt wavetable files.
    n_pos : int, default=256
        Number of positions per wavetable.
    n_samples : int, default=1024
        Number of samples per single-cycle waveform frame.
    seed : int, default=42
        Random seed for harmonic phase spectrum.

    Returns
    -------
    dict[str, T]
        Dictionary mapping wavetable names to their generated Tensor representations.
    """
    os.makedirs(save_dir, exist_ok=True)

    wavetables = {
        f"brightness_synthetic__{n_pos}_{n_samples}": create_synthetic_brightness_sweep(
            n_pos=n_pos, n_samples=n_samples, seed=seed
        ),
        f"warmth_synthetic__{n_pos}_{n_samples}": create_synthetic_warmth_sweep(
            n_pos=n_pos, n_samples=n_samples, seed=seed
        ),
        f"richness_synthetic__{n_pos}_{n_samples}": create_synthetic_richness_sweep(
            n_pos=n_pos, n_samples=n_samples, seed=seed
        ),
    }

    for name, wt in wavetables.items():
        save_path = os.path.join(save_dir, f"{name}.pt")
        tr.save(wt, save_path)
        log.info(f"Saved synthetic wavetable ({wt.shape}): {save_path}")

    return wavetables


def main() -> None:
    """Command-line entry point to create synthetic wavetables."""
    parser = argparse.ArgumentParser(
        description="Generate 3 synthetic wavetables (brightness, warmth, richness)."
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(DATA_DIR, "wavetables"),
        help=f"Directory to save .pt wavetable files (default: {DATA_DIR}/wavetables)",
    )
    parser.add_argument(
        "--n-pos",
        type=int,
        default=256,
        help="Number of wavetable positions/frames (default: 256)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1024,
        help="Number of samples per waveform frame (default: 1024)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for harmonic phases (default: 42)",
    )
    args = parser.parse_args()

    make_synthetic_wavetables(
        save_dir=args.save_dir,
        n_pos=args.n_pos,
        n_samples=args.n_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
