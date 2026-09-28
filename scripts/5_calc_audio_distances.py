import argparse
import glob
import logging
import os
import re
from typing import Dict, Sequence, Tuple, Union

import pandas as pd
import torch as tr
import torchaudio
from auraloss.freq import MultiResolutionSTFTLoss
from torch import Tensor as T
from torch import nn

from losses import (
    ClapEmbeddingLoss,
    EncodecEmbeddingLoss,
    JTFSTLoss,
    LogMSSLoss,
    MFCCDistance,
    PANNsEmbeddingLoss,
    Scat1DLoss,
    VGGishEmbeddingLoss,
)
from paths import OUT_DIR
from util import find_variants, parse_amount

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def load_audio(path: str, sr: int) -> T:
    """Load an audio file, verify sample rate, and return a mono (1, 1, N) tensor.

    Parameters
    ----------
    path : str
        Path to the audio WAV file.
    sr : int
        Expected sample rate.

    Returns
    -------
    T
        Audio tensor with shape (1, 1, n_samples).
    """
    audio, audio_sr = torchaudio.load(path)
    assert audio_sr == sr, f"Expected sr={sr}, got {audio_sr} for {path}"
    # The samples are mono duplicated across both channels
    audio = audio[:1, :]
    return audio.unsqueeze(0)


def resolve_loss_fn(
    entry: Union[nn.Module, Tuple[str, nn.Module]],
) -> Tuple[str, nn.Module]:
    """Normalize a loss entry into a (name, loss_module) pair.

    Parameters
    ----------
    entry : Union[nn.Module, Tuple[str, nn.Module]]
        Either a loss module instance or a (name, loss_module) tuple.

    Returns
    -------
    Tuple[str, nn.Module]
        Tuple of (name, loss_fn), using the class name if not explicitly specified.
    """
    if isinstance(entry, tuple):
        name, loss_fn = entry
        return name, loss_fn
    return entry.__class__.__name__, entry


def get_phase_info(path: str) -> Tuple[int, int, str]:
    """Extract phase metadata from an audio file path.

    Parameters
    ----------
    path : str
        Audio file path to parse.

    Returns
    -------
    Tuple[int, int, str]
        Tuple of (phase_idx, n_phases, phase_str). Returns (0, 1, '') if no phase tag is found.
    """
    m = re.search(r"__phase_(\d+)_(\d+)", os.path.basename(path))
    if m:
        p_idx = int(m.group(1))
        n_p = int(m.group(2))
        return p_idx, n_p, f"__phase_{p_idx}_{n_p}"
    return 0, 1, ""


def extract_stimulus_tag(
    variant_name: str,
    wt_name: str,
    phase_str: str,
    target_lufs: Union[int, float] = -18,
) -> str:
    """Extract the base stimulus tag from a variant filename.

    Strips the wavetable prefix, phase suffix, and LUFS loudness tag.

    Parameters
    ----------
    variant_name : str
        Basename of the variant audio file (without .wav extension).
    wt_name : str
        Wavetable name prefix.
    phase_str : str
        Phase substring (e.g. '__phase_0_24') or empty string.
    target_lufs : Union[int, float], default=-18
        Target LUFS value used in filename formatting.

    Returns
    -------
    str
        Extracted stimulus tag (e.g. 'amp_1.00hz_0.50').
    """
    core = variant_name[len(f"{wt_name}__") :]
    if phase_str and core.endswith(phase_str):
        core = core[: -len(phase_str)]
    lufs_tag = f"_{target_lufs}lufs"
    if core.endswith(lufs_tag):
        return core[: -len(lufs_tag)]
    return re.sub(r"_[+-]?\d+(?:\.\d+)?lufs$", "", core)


def find_reference_path(
    samples_dir: str,
    wt_name: str,
    mod_sig: str,
    target_lufs: Union[int, float],
    phase_idx: int,
    n_phases: int,
) -> str:
    """Find the path to the reference audio sample for a given wavetable, modulation, and phase.

    Parameters
    ----------
    samples_dir : str
        Directory containing synthesized audio files.
    wt_name : str
        Wavetable name prefix.
    mod_sig : str
        Reference modulation signature (e.g. 'amp_1.00hz_0.10').
    target_lufs : Union[int, float]
        Target LUFS loudness level.
    phase_idx : int
        Phase index of the reference audio.
    n_phases : int
        Total number of phases.

    Returns
    -------
    str
        Path to the matching reference WAV file.

    Raises
    ------
    FileNotFoundError
        If no matching reference audio file is found in samples_dir.
    """
    # 1. Exact match with n_phases (supporting int or float formatting)
    for lufs_str in [f"{target_lufs}", f"{float(target_lufs):.1f}"]:
        path_exact = os.path.join(
            samples_dir,
            f"{wt_name}__{mod_sig}_{lufs_str}lufs__phase_{phase_idx}_{n_phases}.wav",
        )
        if os.path.exists(path_exact):
            return path_exact

    # 2. Glob with any n_phases
    matches = sorted(
        glob.glob(
            os.path.join(
                samples_dir,
                f"{wt_name}__{mod_sig}*lufs*__phase_{phase_idx}_*.wav",
            )
        )
    )
    if matches:
        return matches[0]

    # 3. Fallback to unphased file if phase_idx == 0
    if phase_idx == 0:
        for lufs_str in [f"{target_lufs}", f"{float(target_lufs):.1f}"]:
            path_unphased = os.path.join(
                samples_dir, f"{wt_name}__{mod_sig}_{lufs_str}lufs.wav"
            )
            if os.path.exists(path_unphased):
                return path_unphased

        matches_unphased = [
            p
            for p in sorted(
                glob.glob(os.path.join(samples_dir, f"{wt_name}__{mod_sig}*lufs*.wav"))
            )
            if "__phase_" not in os.path.basename(p)
        ]
        if matches_unphased:
            return matches_unphased[0]

    raise FileNotFoundError(
        f"Missing reference audio for wt='{wt_name}', ref='{mod_sig}', phase={phase_idx} in {samples_dir}"
    )


def compute_distances(
    loss_fns: Sequence[Union[nn.Module, Tuple[str, nn.Module]]],
    wavetables: Sequence[str],
    mod_sig_references: Sequence[str],
    samples_dir: str,
    save_path: str,
    sr: int = 44100,
    target_lufs: Union[int, float] = -18,
    ref_match_phase: bool = False,
) -> pd.DataFrame:
    """Compute distances between reference and variant audio across loss functions and save to TSV.

    Parameters
    ----------
    loss_fns : Sequence[Union[nn.Module, Tuple[str, nn.Module]]]
        Loss function(s) or (name, loss_fn) pairs to evaluate.
    wavetables : Sequence[str]
        List of wavetable names to evaluate.
    mod_sig_references : Sequence[str]
        Reference modulation signals (e.g. 'amp_1.00hz_0.10', 'freq_0.25hz', 'reg_1.00hz_0.000').
    samples_dir : str
        Directory containing synthesized audio samples.
    save_path : str
        Path to output TSV file.
    sr : int, default=44100
        Sample rate.
    target_lufs : Union[int, float], default=-18
        Loudness normalization level used in filenames.
    ref_match_phase : bool, default=False
        If True, compare each variant at phase_n against the reference audio with
        the corresponding phase_n.
        If False, always compare each variant against the reference audio at phase 0.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all computed distance records.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    suffix = f"_{target_lufs}lufs.wav"

    loss_entries = [resolve_loss_fn(entry) for entry in loss_fns]
    loss_names = [name for name, _ in loss_entries]
    assert len(set(loss_names)) == len(
        loss_names
    ), f"Loss function names must be unique, got {loss_names}"

    audio_cache: Dict[str, T] = {}

    def get_audio(path: str) -> T:
        if path not in audio_cache:
            audio_cache[path] = load_audio(path, sr)
        return audio_cache[path]

    rows = []

    for loss_name, loss_fn in loss_entries:
        for wt_name in wavetables:
            for mod_sig in mod_sig_references:
                _, ref_amount, _ = parse_amount(mod_sig)
                variant_paths = find_variants(samples_dir, wt_name, mod_sig, suffix)
                log.info(
                    f"{loss_name} | {wt_name} | {mod_sig}: found {len(variant_paths)} samples"
                )

                for variant_path in variant_paths:
                    variant_name = os.path.basename(variant_path)
                    if variant_name.endswith(".wav"):
                        variant_name = variant_name[:-4]

                    phase_idx, n_phases, phase_str = get_phase_info(variant_path)
                    stim_tag = extract_stimulus_tag(
                        variant_name, wt_name, phase_str, target_lufs
                    )
                    _, amount, _ = parse_amount(stim_tag)
                    variant_mod_sig = f"{stim_tag}{phase_str}"

                    # Determine reference phase based on ref_match_phase boolean option
                    target_ref_phase = phase_idx if ref_match_phase else 0
                    ref_path = find_reference_path(
                        samples_dir=samples_dir,
                        wt_name=wt_name,
                        mod_sig=mod_sig,
                        target_lufs=target_lufs,
                        phase_idx=target_ref_phase,
                        n_phases=n_phases,
                    )

                    ref_audio = get_audio(ref_path)
                    audio = get_audio(variant_path)
                    assert (
                        audio.shape == ref_audio.shape
                    ), f"Shape mismatch: {audio.shape} vs {ref_audio.shape}"

                    with tr.no_grad():
                        dist = loss_fn(audio, ref_audio).item()

                    is_reference = (amount == ref_amount) and (
                        phase_idx == target_ref_phase
                    )

                    rows.append(
                        {
                            "loss_fn": loss_name,
                            "wavetable": wt_name,
                            "mod_type": mod_sig.split("_", 1)[0],
                            "reference": mod_sig,
                            "ref_amount": ref_amount,
                            "mod_sig": variant_mod_sig,
                            "amount": amount,
                            "phase": phase_idx,
                            "ref_phase": target_ref_phase,
                            "is_reference": is_reference,
                            "distance": dist,
                        }
                    )
                    log.info(
                        f"  {variant_mod_sig} vs ref_phase_{target_ref_phase}: "
                        f"{dist:.6g}"
                    )

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, sep="\t")
    log.info(f"Saved {len(df)} distances to {save_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute audio loss distances across stimulus variants and phases."
    )
    parser.add_argument(
        "--samples-dir",
        default=os.path.join(OUT_DIR, "stimuli"),
        help=f"Directory containing audio samples (default: {OUT_DIR}/stimuli)",
    )
    parser.add_argument(
        "--save-dir",
        default=OUT_DIR,
        help=f"Directory to save distance TSV (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Explicit TSV save path (default: {save-dir}/audio_distances.tsv)",
    )
    parser.add_argument(
        "--ref-match-phase",
        action="store_true",
        default=False,
        help=(
            "If True, compare each variant at phase_n against reference at phase_n. "
            "If False (default), always compare against reference at phase 0."
        ),
    )
    args = parser.parse_args()

    samples_dir = args.samples_dir
    save_dir = args.save_dir
    tsv_path = args.save_path or os.path.join(save_dir, "audio_distances.tsv")
    ref_match_phase = args.ref_match_phase
    sr = 44100
    target_lufs = -18

    loss_fns = [
        (
            "mss_log_lin",
            MultiResolutionSTFTLoss(
                fft_sizes=[64, 128, 256, 512, 1024, 2048],
                hop_sizes=[16, 32, 64, 128, 256, 512],
                win_lengths=[64, 128, 256, 512, 1024, 2048],
                w_sc=0.0,
                w_phs=0.0,
                w_lin_mag=1.0,
                w_log_mag=1.0,
                mag_distance="L1",
            ),
        ),
        (
            "mss_rev",
            LogMSSLoss(
                fft_sizes=[67, 127, 257, 509, 1021, 2053],
                hop_sizes=[33, 63, 128, 254, 510, 1026],
                win_lengths=[67, 127, 257, 509, 1021, 2053],
                window="flat_top",
                log_mag_eps=1.0,
                gamma=1.0,
                p=2,
            ),
        ),
        ("mfcc", MFCCDistance(sr=sr)),
        (
            "scat1d",
            Scat1DLoss(
                shape=176400,
                J=12,
                Q1=8,
                Q2=2,
                T=None,
                max_order=2,
                p=2,
                use_rho_log1p=True,
            ),
        ),
        (
            "jtfs",
            JTFSTLoss(
                shape=176400,
                J=12,
                Q1=8,
                Q2=2,
                J_fr=5,
                Q_fr=2,
                T=None,
                F=None,
                format_="joint",
                p=2,
                use_rho_log1p=True,
            ),
        ),
        ("vggish", VGGishEmbeddingLoss(in_sr=sr)),
        ("clap", ClapEmbeddingLoss(use_cuda=False, in_sr=sr)),
        (
            "encodec48k",
            EncodecEmbeddingLoss(in_sr=sr, model_id="facebook/encodec_48khz"),
        ),
        (
            "panns_wavegram_logmel",
            PANNsEmbeddingLoss(variant="wavegram-logmel", in_sr=sr),
        ),
    ]
    wavetables = [
        "brightness_real__harmonics__synced_sines__256_1024",
        "brightness_synthetic__256_1024",
        "richness_real__filter__acid_saw__46_1024__inverted",
        "richness_synthetic__256_1024",
        "warmth_real__vintage__logue_saw__166_1024",
        "warmth_synthetic__256_1024",
    ]
    mod_sig_references = [
        "amp_1.00hz_0.10",
        "freq_0.25hz",
        "reg_1.00hz_0.000",
    ]

    os.makedirs(save_dir, exist_ok=True)

    compute_distances(
        loss_fns=loss_fns,
        wavetables=wavetables,
        mod_sig_references=mod_sig_references,
        samples_dir=samples_dir,
        save_path=tsv_path,
        sr=sr,
        target_lufs=target_lufs,
        ref_match_phase=ref_match_phase,
    )
