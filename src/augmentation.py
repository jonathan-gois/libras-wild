"""
Data augmentation para sequências de landmarks MediaPipe.

Aumentações implementadas:
  1. Mirror (espelhar eixo X) — simula sinalizadores canhotos
  2. Time stretch — resampling temporal (velocidades diferentes)
  3. Gaussian noise — pequena perturbação nos landmarks
  4. Temporal crop — corte aleatório de início/fim
  5. Coordinate dropout — zerar landmark aleatório (simula oclusão)
"""

import numpy as np
from scipy.interpolate import interp1d

# ── índices de landmarks que precisam ser espelhados ─────────────────
# MediaPipe Holistic: pose(0-32) + left_hand(33-53) + right_hand(54-74)
# No espelho: left_hand ↔ right_hand, e coordenada X = 1 - X
POSE_MIRROR_PAIRS = [
    (1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),
    (15,16),(17,18),(19,20),(21,22),(23,24),(25,26),
    (27,28),(29,30),(31,32),
]
LHAND_OFFSET = 33
RHAND_OFFSET = 54


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Espelha a sequência no eixo X.
    seq: (T, 225) — flat [pose(99), lhand(63), rhand(63)]
    """
    seq = seq.copy()
    T = seq.shape[0]
    xyz = seq.reshape(T, 75, 3)

    # Inverter coordenada X de todos os landmarks
    xyz[:, :, 0] = 1.0 - xyz[:, :, 0]

    # Trocar landmarks pares esquerdo/direito na pose
    for (a, b) in POSE_MIRROR_PAIRS:
        xyz[:, a, :], xyz[:, b, :] = xyz[:, b, :].copy(), xyz[:, a, :].copy()

    # Trocar mão esquerda ↔ direita (índices 33-53 ↔ 54-74)
    lh = xyz[:, LHAND_OFFSET:LHAND_OFFSET+21, :].copy()
    rh = xyz[:, RHAND_OFFSET:RHAND_OFFSET+21, :].copy()
    xyz[:, LHAND_OFFSET:LHAND_OFFSET+21, :] = rh
    xyz[:, RHAND_OFFSET:RHAND_OFFSET+21, :] = lh

    return xyz.reshape(T, 225)


def time_stretch(seq: np.ndarray,
                 rate: float = None,
                 rate_range: tuple = (0.7, 1.3)) -> np.ndarray:
    """
    Reamosta temporalmente a sequência.
    rate < 1 → mais lento (mais frames), rate > 1 → mais rápido (menos frames).
    """
    T, F = seq.shape
    if rate is None:
        rate = np.random.uniform(*rate_range)
    new_T = max(10, int(T / rate))
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, new_T)
    interp = interp1d(old_t, seq, axis=0, kind="linear")
    return interp(new_t).astype(np.float32)


def add_noise(seq: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """Adiciona ruído gaussiano nos landmarks."""
    noise = np.random.normal(0, sigma, seq.shape).astype(np.float32)
    return seq + noise


def temporal_crop(seq: np.ndarray,
                  crop_ratio: float = None,
                  ratio_range: tuple = (0.85, 1.0)) -> np.ndarray:
    """Corta aleatoriamente início e/ou fim da sequência."""
    T = seq.shape[0]
    if crop_ratio is None:
        crop_ratio = np.random.uniform(*ratio_range)
    keep = max(10, int(T * crop_ratio))
    max_start = T - keep
    start = np.random.randint(0, max_start + 1)
    return seq[start:start + keep]


def coordinate_dropout(seq: np.ndarray,
                        drop_prob: float = 0.05) -> np.ndarray:
    """Zera landmarks individuais com probabilidade drop_prob."""
    seq = seq.copy()
    T, F = seq.shape
    n_landmarks = 75
    mask = np.random.rand(T, n_landmarks) > drop_prob  # (T, 75)
    mask_flat = np.repeat(mask, 3, axis=1)  # (T, 225)
    seq *= mask_flat
    return seq


def augment_sequence(seq: np.ndarray,
                     mirror: bool = False,
                     stretch: bool = True,
                     noise: bool = True,
                     crop: bool = True,
                     dropout: bool = True) -> np.ndarray:
    """Aplica augmentações compostas a uma sequência."""
    if mirror:
        seq = mirror_sequence(seq)
    if crop and np.random.rand() < 0.5:
        seq = temporal_crop(seq)
    if stretch and np.random.rand() < 0.5:
        seq = time_stretch(seq)
    if noise and np.random.rand() < 0.7:
        seq = add_noise(seq)
    if dropout and np.random.rand() < 0.3:
        seq = coordinate_dropout(seq)
    return seq


def build_augmented_dataset(sequences: list, labels: np.ndarray,
                             mirror_all: bool = True,
                             n_extra: int = 2) -> tuple:
    """
    Gera dataset aumentado.

    mirror_all : duplica todo o dataset espelhando (simula canhotos)
    n_extra    : quantas versões aumentadas adicionais por amostra original

    Retorna (sequences_aug, labels_aug)
    """
    new_seqs = list(sequences)
    new_labels = list(labels)

    # 1. Mirror de toda a base
    if mirror_all:
        for s, l in zip(sequences, labels):
            new_seqs.append(mirror_sequence(s))
            new_labels.append(l)

    # 2. Augmentação aleatória adicional
    for _ in range(n_extra):
        for s, l in zip(sequences, labels):
            aug = augment_sequence(s.copy())
            new_seqs.append(aug)
            new_labels.append(l)

    return new_seqs, np.array(new_labels)
