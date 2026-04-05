"""
Features cinemáticas: velocidade e aceleração dos landmarks.

Dado uma sequência (T, 225), calcula:
- velocidade  = diff de 1ª ordem
- aceleração  = diff de 2ª ordem

Retorna estatísticas resumidas por landmark (para classificadores clássicos)
e a sequência completa (para modelos sequenciais).
"""

import numpy as np


def compute_velocity(sequence: np.ndarray) -> np.ndarray:
    """(T, 225) -> (T-1, 225)"""
    return np.diff(sequence, n=1, axis=0)


def compute_acceleration(sequence: np.ndarray) -> np.ndarray:
    """(T, 225) -> (T-2, 225)"""
    return np.diff(sequence, n=2, axis=0)


def kinematic_stats(sequence: np.ndarray) -> np.ndarray:
    """
    Calcula estatísticas resumidas de velocidade e aceleração.

    sequence: (T, 225)
    Retorna vetor 1-D com:
      - mean, std, max, min da velocidade  -> 4 * 225 = 900
      - mean, std, max, min da aceleração  -> 4 * 225 = 900
    Total: 1800 features
    """
    vel = compute_velocity(sequence)   # (T-1, 225)
    acc = compute_acceleration(sequence)  # (T-2, 225)

    feats = []
    for arr in (vel, acc):
        feats.append(np.mean(np.abs(arr), axis=0))
        feats.append(np.std(arr, axis=0))
        feats.append(np.max(np.abs(arr), axis=0))
        feats.append(np.min(np.abs(arr), axis=0))

    return np.concatenate(feats)  # (1800,)


def batch_kinematic_stats(sequences: list) -> np.ndarray:
    """
    sequences: lista de arrays (T_i, 225)
    Retorna (N, 1800)
    """
    return np.stack([kinematic_stats(s) for s in sequences])


def pad_sequence(sequence: np.ndarray, max_len: int,
                 pad_value: float = 0.0) -> np.ndarray:
    """Pad ou trunca sequência para max_len frames."""
    T, F = sequence.shape
    if T >= max_len:
        return sequence[:max_len]
    pad = np.full((max_len - T, F), pad_value, dtype=sequence.dtype)
    return np.concatenate([sequence, pad], axis=0)


def build_kinematic_sequences(sequences: list,
                               max_len: int = 200) -> np.ndarray:
    """
    Para modelos sequenciais: concatena posição + velocidade + aceleração
    em cada frame.

    Retorna (N, max_len, 225*3) com padding.
    velocidade e aceleração são zero-padded no início para alinhar frames.
    """
    result = []
    for seq in sequences:
        T, F = seq.shape
        vel = np.zeros_like(seq)
        acc = np.zeros_like(seq)

        if T > 1:
            vel[1:] = np.diff(seq, n=1, axis=0)
        if T > 2:
            acc[2:] = np.diff(seq, n=2, axis=0)

        combined = np.concatenate([seq, vel, acc], axis=1)  # (T, F*3)
        combined = pad_sequence(combined, max_len)
        result.append(combined)

    return np.stack(result)  # (N, max_len, F*3)
