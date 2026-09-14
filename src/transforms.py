"""IQ -> log-magnitude spectrogram, shared identically by training and serving."""
from __future__ import annotations

import torch

from . import config


def compute_log_magnitude_spectrogram(
    iq: torch.Tensor,
    n_fft: int = config.N_FFT,
    win_length: int = config.WIN_LENGTH,
    hop_length: int = config.HOP_LENGTH,
    eps: float = 1e-6,
) -> torch.Tensor:
    """iq: (2, N) float32 tensor of [I, Q] rows -> (1, n_fft, num_frames)
    normalized log-magnitude spectrogram, DC-centered on the frequency axis.
    """
    complex_signal = torch.complex(iq[0], iq[1])
    window = torch.hann_window(win_length, dtype=iq.dtype, device=iq.device)
    spec = torch.stft(
        complex_signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        onesided=False,
        return_complex=True,
    )  # (n_fft, num_frames) complex

    log_mag = torch.log10(spec.abs() + eps)
    log_mag = torch.fft.fftshift(log_mag, dim=0)  # center DC bin on the frequency axis
    log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + eps)  # per-sample normalization
    return log_mag.unsqueeze(0)  # (1, n_fft, num_frames)
