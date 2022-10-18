import numpy as np
import librosa
import torch
from torchlibrosa.stft import STFT


def load_audio(
    audio_path: str,
    mono: bool,
    sample_rate: float,
    offset: float = 0.0,
    duration: float = None,
) -> np.ndarray:
    r"""Load audio.

    Args:
        audio_path: str
        mono: bool
        sample_rate: float
    """
    audio, _ = librosa.core.load(
        audio_path, sr=sample_rate, mono=mono, offset=offset, duration=duration
    )
    # (audio_samples,) | (channels_num, audio_samples)

    if audio.ndim == 1:
        audio = audio[None, :]
        # (1, audio_samples,)

    return audio


def load_random_segment(
    audio_path: str,
    random_state: int,
    segment_seconds: float,
    mono: bool,
    sample_rate: int,
) -> np.ndarray:
    r"""Randomly select an audio segment from a recording."""

    duration = librosa.get_duration(filename=audio_path)

    start_time = random_state.uniform(0.0, duration - segment_seconds)

    audio = load_audio(
        audio_path=audio_path,
        mono=mono,
        sample_rate=sample_rate,
        offset=start_time,
        duration=segment_seconds,
    )
    # (channels_num, audio_samples)

    return audio


def wave_to_spectrogram(wave, stft: STFT):
    N, C, L = wave.shape
    x = wave.reshape(-1, L)

    real, imag = stft(x)
    real, imag = real.reshape(N, C, *real.shape[2:]), imag.reshape(N, C, *imag.shape[2:])
    return real, imag


def magphase(real, imag):
    r"""Calculate magnitude and phase from real and imag part of signals.

    Args:
        real: tensor, real part of signals
        imag: tensor, imag part of signals

    Returns:
        mag: tensor, magnitude of signals
        cos: tensor, cosine of phases of signals
        sin: tensor, sine of phases of signals
    """
    mag = (real ** 2 + imag ** 2) ** 0.5
    mag = torch.clamp(mag, 1e-8, np.inf)
    cos = real / mag
    sin = imag / mag
    return mag, cos, sin
    