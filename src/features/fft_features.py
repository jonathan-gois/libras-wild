"""
Features de frequência via FFT com janela deslizante adaptativa.

Estratégia do Paper 2:
- Janela deslizante de tamanho W, passo S
- Para cada janela aplica FFT em cada dimensão (225 features)
- Extrai as K frequências dominantes (magnitude e fase)
- Também computa FFT global sobre toda a sequência

Retorna vetor de features estático para classificadores clássicos.
"""

import numpy as np


def sliding_fft(sequence: np.ndarray,
                window_size: int = 16,
                step: int = 8,
                n_top_freqs: int = 5) -> np.ndarray:
    """
    sequence: (T, F)
    Retorna features de frequência concatenadas:
      - FFT global: mag e fase das n_top_freqs dominantes -> 2 * n_top_freqs * F
      - Sliding FFT: mean e std das mags por freq -> 2 * (window_size//2) * F

    Total (default): 2*5*225 + 2*(8)*225 = 2250 + 3600 = 5850
    """
    T, F = sequence.shape
    feats = []

    # ---- 1. FFT global ----
    global_fft = np.fft.rfft(sequence, axis=0)  # (T//2+1, F)
    mag_global  = np.abs(global_fft)             # magnitude
    phase_global = np.angle(global_fft)          # fase

    # Top-K frequências dominantes (por magnitude somada sobre features)
    importance = mag_global.sum(axis=1)          # (n_freqs,)
    top_k = np.argsort(importance)[::-1][:n_top_freqs]
    top_k = np.sort(top_k)

    feats.append(mag_global[top_k].flatten())    # (n_top_freqs * F,)
    feats.append(phase_global[top_k].flatten())  # (n_top_freqs * F,)

    # ---- 2. Sliding window FFT ----
    n_freqs_half = window_size // 2 + 1
    window_mags = []

    start = 0
    while start + window_size <= T:
        segment = sequence[start:start + window_size]
        win_fft = np.fft.rfft(segment, axis=0)   # (n_freqs_half, F)
        window_mags.append(np.abs(win_fft))
        start += step

    if len(window_mags) == 0:
        # Sequência muito curta: padding
        segment = np.zeros((window_size, F))
        segment[:T] = sequence
        win_fft = np.fft.rfft(segment, axis=0)
        window_mags.append(np.abs(win_fft))

    window_mags = np.stack(window_mags)  # (n_windows, n_freqs_half, F)
    feats.append(window_mags.mean(axis=0).flatten())  # mean over windows
    feats.append(window_mags.std(axis=0).flatten())   # std  over windows

    return np.concatenate(feats)


def batch_fft_features(sequences: list,
                        window_size: int = 16,
                        step: int = 8,
                        n_top_freqs: int = 5) -> np.ndarray:
    """
    sequences: lista de (T_i, 225)
    Retorna (N, D_fft)
    """
    return np.stack([
        sliding_fft(s, window_size, step, n_top_freqs)
        for s in sequences
    ])
