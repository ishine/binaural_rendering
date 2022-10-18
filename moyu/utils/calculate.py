import numpy as np
import torch
from torchlibrosa.stft import STFT, magphase


def float32_to_int16(x: np.float32) -> np.int16:

    x = np.clip(x, a_min=-1, a_max=1)

    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: np.int16) -> np.float32:

    return (x / 32767.0).astype(np.float32)


def magnitude_to_db(x: float) -> float:
    eps = 1e-10
    return 20.0 * np.log10(max(x, eps))


def db_to_magnitude(x: float) -> float:
    return 10.0 ** (x / 20)


def get_pitch_shift_factor(shift_pitch: float) -> float:
    r"""The factor of the audio length to be scaled."""
    return 2 ** (shift_pitch / 12)


def calculate_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    s_true = ref
    s_artif = est - ref
    sdr = 10.0 * (
        np.log10(np.clip(np.mean(s_true ** 2), 1e-8, np.inf))
        - np.log10(np.clip(np.mean(s_artif ** 2), 1e-8, np.inf))
    )
    return sdr


def calculate_lsd(x, y, 
        stft=STFT(n_fft=2048, hop_length=480, pad_mode="reflect", center=True, window="hann")):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    # x, y: channels, samples
    X, *_ = magphase(*stft(x))
    Y, *_ = magphase(*stft(y))

    a = torch.log10(X ** 2)
    b = torch.log10(Y ** 2)

    lsd = torch.mean(torch.sqrt(torch.mean((a - b) ** 2, dim=-1)))
    return lsd
