"""Generate audio stimulus sweeps and modulation signal plots for the listening test.

This script synthesizes audio stimuli across multiple wavetables using amplitude,
frequency, and irregularity modulations. The modulation trajectories are warped
using precomputed perceptual lookup tables (LUTs), rendered into wavetable sweeps,
loudness-normalized, windowed with fade-in/out, and saved as WAV audio files.
In addition, publication-ready plots of the modulation signals are rendered and saved.
"""

import argparse
import glob
import logging
import os
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as tr
import torchaudio
from torch import Tensor as T

import util
from modulations import make_mod_signal, make_quasi_periodic
from paths import DATA_DIR, OUT_DIR

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

SERIES_COLOR = "#2a78d6"
AXIS_COLOR = "#52514e"

# Modulation amounts are displayed as (scale, unit), e.g. an irregularity amount
# of 0.125 is displayed as 12.5%
MOD_SIG_AMOUNT_UNITS = {
    "amp": (1.0, ""),
    "freq": (1.0, " Hz"),
    "reg": (100.0, "%"),
}
# Wide enough for the five modulation signals of one modulation type in a row.
# The height is just enough for the 2:1 plots and their labels, since the width
# is what determines how large the plots end up
FIG_SIZE = (12.0, 1.8)
# Height / width of each plotting area, i.e. each plot is twice as wide as tall
BOX_ASPECT = 0.5
DPI = 300
FONT_SIZE = 12


def make_mod_sig(
    mod_type: str,
    amount: float,
    n_samples: int,
    sr: int,
    mod_freq: float = 1.0,
    amp_center_val: float = 0.5,
    reg_seed: int = 42,
) -> T:
    """Generate a modulation signal tensor for a stimulus.

    Parameters
    ----------
    mod_type : str
        Type of modulation: 'freq' (frequency/rate), 'amp' (amplitude/depth),
        or 'reg' (temporal irregularity/randomness).
    amount : float
        Modulation intensity (rate in Hz for 'freq', depth in [0, 1] for 'amp',
        or jitter randomness in [0, 1] for 'reg').
    n_samples : int
        Number of audio samples to generate.
    sr : int
        Sample rate in Hz.
    mod_freq : float, default=1.0
        Fixed base modulation frequency in Hz for 'amp' and 'reg' modulations.
    amp_center_val : float, default=0.5
        Baseline wavetable position center for amplitude modulation.
    reg_seed : int, default=42
        Random seed for generating quasi-periodic irregularity intervals.

    Returns
    -------
    T
        1D tensor of shape (n_samples,) containing the modulation trajectory in [0, 1].

    Raises
    ------
    ValueError
        If `mod_type` is not one of ('freq', 'amp', 'reg').
    """
    if mod_type == "freq":
        return make_mod_signal(n_samples, sr, amount, shape="cos")
    elif mod_type == "amp":
        mod_sig = make_mod_signal(
            n_samples, sr, mod_freq, shape="cos", phase=tr.pi / 2
        )
        mod_sig = amp_center_val + (mod_sig - 0.5) * amount
        log.info(
            f"amp={amount:.2f} mod_sig min={mod_sig.min():.4f} "
            f"max={mod_sig.max():.4f} mean={mod_sig.mean():.4f}"
        )
        return mod_sig
    elif mod_type == "reg":
        mod_sig = make_mod_signal(n_samples, sr, mod_freq, shape="cos")
        mod_sig, norm_gaps = make_quasi_periodic(
            mod_sig, randomness=amount, seed=reg_seed
        )
        log.info(f"r={amount:.2f} intervals: {[f'{g:.2f}' for g in norm_gaps]}")
        return mod_sig
    else:
        raise ValueError(f"Unsupported mod_type: {mod_type}")


def format_amounts(
    mod_type: str,
    amounts: Sequence[float],
    max_decimals: int = 6,
) -> List[str]:
    """Format modulation amounts with appropriate scale, unit, and minimal decimal precision.

    Parameters
    ----------
    mod_type : str
        Modulation type ('amp', 'freq', or 'reg').
    amounts : Sequence[float]
        Sequence of modulation amounts to format.
    max_decimals : int, default=6
        Maximum number of decimal places to check for exact representation.

    Returns
    -------
    List[str]
        List of formatted strings with units.
    """
    scale, unit = MOD_SIG_AMOUNT_UNITS[mod_type]
    vals = [a * scale for a in amounts]
    n_decimals = max_decimals
    for d in range(max_decimals + 1):
        if all(abs(v - round(v, d)) < 1e-9 for v in vals):
            n_decimals = d
            break
    return [f"{v:.{n_decimals}f}{unit}" for v in vals]


def plot_mod_signals(
    mod_type: str,
    amounts: Sequence[float],
    n_samples: int,
    sr: int,
    mod_freq: float = 1.0,
    amp_center_val: float = 0.5,
    reg_seed: int = 42,
    save_dir: str = "",
    fig_size: Tuple[float, float] = FIG_SIZE,
    dpi: int = DPI,
) -> None:
    """Plot modulation trajectories of one modulation type side-by-side.

    The lowest amount corresponds to the reference stimulus in the listening test,
    and subsequent amounts are labeled incrementally.

    Parameters
    ----------
    mod_type : str
        Modulation type ('amp', 'freq', or 'reg').
    amounts : Sequence[float]
        Sequence of modulation amount values.
    n_samples : int
        Total number of samples in each signal.
    sr : int
        Sample rate in Hz.
    mod_freq : float, default=1.0
        Modulation frequency in Hz for 'amp' and 'reg'.
    amp_center_val : float, default=0.5
        Center value for amplitude modulation.
    reg_seed : int, default=42
        Random seed for irregularity modulation.
    save_dir : str, default=""
        Directory where the plot SVG will be saved. If empty, saving is skipped.
    fig_size : Tuple[float, float], default=FIG_SIZE
        Figure size (width, height) in inches.
    dpi : int, default=DPI
        Resolution of the saved figure in dots per inch.
    """
    amounts_sorted = sorted(amounts)
    mod_sigs = [
        make_mod_sig(
            mod_type, amount, n_samples, sr, mod_freq, amp_center_val, reg_seed
        )
        for amount in amounts_sorted
    ]
    amount_labels = format_amounts(mod_type, amounts_sorted)
    dur_sec = n_samples / sr
    t = np.arange(n_samples) / sr

    # Constrained layout keeps the shared x label tight against the plots
    fig, axs = plt.subplots(
        1,
        len(mod_sigs),
        figsize=fig_size,
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axs = axs[0]
    for idx, (ax, mod_sig) in enumerate(zip(axs, mod_sigs)):
        ax.set_box_aspect(BOX_ASPECT)  # 2:1 plotting area, not a 2:1 figure
        ax.plot(t, mod_sig.numpy(), color=SERIES_COLOR, linewidth=1.5)
        label = "Reference" if idx == 0 else f"Amount {idx}"
        ax.set_title(f"{label} ({amount_labels[idx]})", fontsize=FONT_SIZE)
        ax.set_xlim(0.0, dur_sec)
        ax.set_xticks(np.arange(0.0, dur_sec + 1e-6, 1.0))
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=FONT_SIZE)
        ax.grid(True, color=AXIS_COLOR, alpha=0.15, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axs[0].set_ylabel("Wavetable position", fontsize=FONT_SIZE)
    fig.supxlabel("Time (s)", fontsize=FONT_SIZE)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"mod_sigs__{mod_type}.svg"
        plt.savefig(os.path.join(save_dir, save_name), dpi=dpi)
        log.info(f"Saved {save_name} ({len(mod_sigs)} signals, {dpi} dpi)")
    plt.close(fig)


def render_and_save_sweep(
    wt: T,
    wt_name: str,
    stim_tag: str,
    mod_sig: T,
    lut: Optional[np.ndarray],
    period_sec: float,
    n_phases: int,
    sr: int,
    sweep_dur_sec: float,
    target_lufs: float,
    fade_in: np.ndarray,
    fade_out: np.ndarray,
    fade_samples: int,
    save_dir: str,
) -> None:
    """Warp a modulation signal via LUT and render phase-shifted audio sweeps to disk.

    Parameters
    ----------
    wt : T
        Wavetable tensor of shape (n_frames, frame_size).
    wt_name : str
        Name prefix of the wavetable.
    stim_tag : str
        Stimulus identifier tag (e.g. 'freq_0.25hz', 'amp_1.00hz_0.10').
    mod_sig : T
        Base 1D modulation signal tensor of shape (n_samples,).
    lut : Optional[np.ndarray]
        Linearized lookup table array, or None if no LUT warping is applied.
    period_sec : float
        Modulation period duration in seconds.
    n_phases : int
        Number of uniformly spaced phase shifts to render.
    sr : int
        Audio sample rate in Hz.
    sweep_dur_sec : float
        Total duration of the synthesized sweep in seconds.
    target_lufs : float
        Target integrated loudness in LUFS for normalization.
    fade_in : np.ndarray
        Linear fade-in window array.
    fade_out : np.ndarray
        Linear fade-out window array.
    fade_samples : int
        Number of fade-in and fade-out samples at boundaries.
    save_dir : str
        Directory to save generated WAV audio files.
    """
    if lut is not None:
        mod_sig_warped = np.interp(mod_sig.numpy(), np.linspace(0, 1, len(lut)), lut)
    else:
        mod_sig_warped = mod_sig.numpy()

    period_samples = int(round(period_sec * sr))
    for phase_idx in range(n_phases):
        shift = int(round(phase_idx * period_samples / n_phases))
        rolled_mod_sig = np.roll(mod_sig_warped, shift)

        sweep = util.create_wavetable_sweep(
            wt, sr=sr, duration=sweep_dur_sec, mod_signal=rolled_mod_sig
        )
        sweep_norm, loudness, gain = util.loudness_normalize(sweep, sr, target_lufs)

        sweep_norm[:fade_samples] *= fade_in
        sweep_norm[-fade_samples:] *= fade_out

        phase_suffix = f"__phase_{phase_idx}_{n_phases}" if n_phases > 1 else ""
        save_name = f"{wt_name}__{stim_tag}_{target_lufs}lufs{phase_suffix}.wav"
        save_path = os.path.join(save_dir, save_name)
        torchaudio.save(
            save_path,
            tr.tensor(sweep_norm).unsqueeze(0).expand(2, -1).float(),
            sr,
        )
        log.info(f"Saved {save_name} (loudness={loudness:.1f}, gain={gain:.1f}dB)")


def generate_stimuli(
    wavetable_dir: str,
    stimuli_save_dir: str,
    figures_save_dir: Optional[str] = None,
    sr: int = 44100,
    sweep_dur_sec: float = 4.0,
    target_lufs: float = -18.0,
    fade_samples: int = 256,
    n_phases: int = 1,
) -> None:
    """Generate audio sweeps for all wavetables and render modulation signal figures.

    Parameters
    ----------
    wavetable_dir : str
        Directory containing wavetable .pt files and optional .npy LUTs.
    stimuli_save_dir : str
        Directory to save generated WAV audio files.
    figures_save_dir : Optional[str], default=None
        Directory to save modulation signal plots. If None, uses `{OUT_DIR}/figures`.
    sr : int, default=44100
        Audio sample rate in Hz.
    sweep_dur_sec : float, default=4.0
        Duration in seconds for each audio sweep.
    target_lufs : float, default=-18.0
        Target loudness in LUFS for audio normalization.
    fade_samples : int, default=256
        Number of fade-in/out samples applied to audio edges.
    n_phases : int, default=1
        Number of phase shifts to render per modulation.
    """
    if figures_save_dir is None:
        figures_save_dir = os.path.join(OUT_DIR, "figures")

    freq_vals = [0.25, 0.5, 1.0, 2.0, 4.0]
    amp_freq = 1.0
    amp_center_val = 0.5
    amp_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    reg_freq = 1.0
    reg_seed = 42
    reg_vals = [0.0, 0.125, 0.25, 0.375, 0.5]

    n_samples = int(sr * sweep_dur_sec)
    wt_paths = sorted(glob.glob(os.path.join(wavetable_dir, "*.pt")))
    log.info(f"Found {len(wt_paths)} wavetables in {wavetable_dir}")

    fade_in = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)

    os.makedirs(stimuli_save_dir, exist_ok=True)
    os.makedirs(figures_save_dir, exist_ok=True)

    # 1. Render and save modulation trajectory plots
    for mod_type, amounts, mod_freq in [
        ("amp", amp_vals, amp_freq),
        ("freq", freq_vals, 1.0),
        ("reg", reg_vals, reg_freq),
    ]:
        plot_mod_signals(
            mod_type,
            amounts,
            n_samples,
            sr,
            mod_freq=mod_freq,
            amp_center_val=amp_center_val,
            reg_seed=reg_seed,
            save_dir=figures_save_dir,
        )

    # 2. Render and save audio sweeps across wavetables and modulations
    mod_configs = [
        ("freq", freq, 1.0 / freq, f"freq_{freq:.2f}hz") for freq in freq_vals
    ] + [
        ("amp", amp, 1.0 / amp_freq, f"amp_{amp_freq:.2f}hz_{amp:.2f}")
        for amp in amp_vals
    ] + [
        ("reg", reg, 1.0 / reg_freq, f"reg_{reg_freq:.2f}hz_{reg:.3f}")
        for reg in reg_vals
    ]

    for wt_path in wt_paths:
        wt_name = os.path.splitext(os.path.basename(wt_path))[0]
        wt = tr.load(wt_path, weights_only=True)
        lut_path = os.path.join(wavetable_dir, f"{wt_name}__lut.npy")
        if os.path.exists(lut_path):
            lut = np.load(lut_path)
            log.info(
                f"Processing wavetable: {wt_name} (shape={wt.shape}, lut={lut.shape})"
            )
        else:
            lut = None
            log.warning(
                f"No LUT file found at {lut_path}, skipping warping for {wt_name}"
            )

        for mod_type, amount, period_sec, stim_tag in mod_configs:
            if mod_type == "freq":
                mod_sig = make_mod_sig("freq", amount, n_samples, sr)
            elif mod_type == "amp":
                mod_sig = make_mod_sig(
                    "amp",
                    amount,
                    n_samples,
                    sr,
                    mod_freq=amp_freq,
                    amp_center_val=amp_center_val,
                )
            elif mod_type == "reg":
                mod_sig = make_mod_sig(
                    "reg",
                    amount,
                    n_samples,
                    sr,
                    mod_freq=reg_freq,
                    reg_seed=reg_seed,
                )
            else:
                continue

            render_and_save_sweep(
                wt=wt,
                wt_name=wt_name,
                stim_tag=stim_tag,
                mod_sig=mod_sig,
                lut=lut,
                period_sec=period_sec,
                n_phases=n_phases,
                sr=sr,
                sweep_dur_sec=sweep_dur_sec,
                target_lufs=target_lufs,
                fade_in=fade_in,
                fade_out=fade_out,
                fade_samples=fade_samples,
                save_dir=stimuli_save_dir,
            )


def main() -> None:
    """Command-line entry point to generate audio stimuli and modulation plots."""
    parser = argparse.ArgumentParser(
        description="Synthesize audio stimuli and plot modulation signals for listening test."
    )
    parser.add_argument(
        "--wavetable-dir",
        default=os.path.join(DATA_DIR, "wavetables"),
        help=f"Directory containing wavetable .pt and .npy files (default: {DATA_DIR}/wavetables)",
    )
    parser.add_argument(
        "--stimuli-dir",
        default=os.path.join(OUT_DIR, "stimuli"),
        help=f"Directory to save generated WAV stimuli (default: {OUT_DIR}/stimuli)",
    )
    parser.add_argument(
        "--figures-dir",
        default=os.path.join(OUT_DIR, "figures"),
        help=f"Directory to save modulation signal plots (default: {OUT_DIR}/figures)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Sweep duration in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-18.0,
        help="Target integrated LUFS (default: -18.0)",
    )
    parser.add_argument(
        "--n-phases",
        type=int,
        default=1,
        help="Number of phases to synthesize per modulation (default: 1)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.wavetable_dir):
        raise FileNotFoundError(
            f"Wavetable directory not found: {args.wavetable_dir}"
        )

    generate_stimuli(
        wavetable_dir=args.wavetable_dir,
        stimuli_save_dir=args.stimuli_dir,
        figures_save_dir=args.figures_dir,
        sr=args.sr,
        sweep_dur_sec=args.duration,
        target_lufs=args.target_lufs,
        n_phases=args.n_phases,
    )


if __name__ == "__main__":
    main()
