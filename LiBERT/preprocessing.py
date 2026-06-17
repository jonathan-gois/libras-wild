"""Normalização e suavização de landmarks — portado de run_v3.py (normalize_v3, kalman_v3)
para que o LiBERT use exatamente o mesmo pré-processamento do pipeline MINDS-Libras."""

import numpy as np

from config import POSE_N, LHAND_N, RHAND_N, KALMAN_Q, KALMAN_R, USE_Z_NORM


def normalize_v3(seq: np.ndarray, norm_z: bool = True) -> np.ndarray:
    """Centraliza no quadril, escala pela distância entre ombros (fix B1: normaliza z também)."""
    T = seq.shape[0]
    xyz = seq.reshape(T, 75, 3).copy()
    hip = (xyz[:, 23, :2] + xyz[:, 24, :2]) / 2
    sd = np.linalg.norm(xyz[:, 11, :2] - xyz[:, 12, :2], axis=1).mean() + 1e-6

    xyz[:, :, 0] = (xyz[:, :, 0] - hip[:, 0:1]) / sd
    xyz[:, :, 1] = (xyz[:, :, 1] - hip[:, 1:2]) / sd
    if norm_z:
        xyz[:, :, 2] = xyz[:, :, 2] / sd
    return xyz.reshape(T, 225)


def kalman_v3(seq: np.ndarray, Q: float, R: float) -> np.ndarray:
    """Random-walk Kalman smoothing (fix B3: init no primeiro frame não-zero)."""
    norms = np.linalg.norm(seq, axis=1)
    first_valid = int(np.argmax(norms > 1e-6))

    out = np.empty_like(seq)
    x = seq[first_valid].copy()
    P = np.ones(seq.shape[1])

    for t in range(len(seq)):
        P = P + Q
        K = P / (P + R)
        x = x + K * (seq[t] - x)
        P = (1 - K) * P
        out[t] = x
    return out


def preprocess(seq: np.ndarray) -> np.ndarray:
    """Pipeline completo: normalize_v3 -> kalman_v3, igual ao MINDS-Libras v3."""
    return kalman_v3(normalize_v3(seq, norm_z=USE_Z_NORM), KALMAN_Q, KALMAN_R).astype(np.float32)
