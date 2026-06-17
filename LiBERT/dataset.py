"""Dataset de pré-treino: carrega o cache de janelas e aplica masking por spans contíguos
a cada amostra (masking dinâmico — recalculado a cada época, não fixo no cache)."""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from config import CACHE_DIR, MASK_RATIO, MASK_SPAN_MIN, MASK_SPAN_MAX, VAL_FRACTION, SEED


def span_mask(T: int, ratio: float, span_min: int, span_max: int, rng: np.random.Generator) -> np.ndarray:
    """Marca ~`ratio` dos T frames em spans contíguos (estilo SpanBERT/wav2vec2)."""
    mask = np.zeros(T, dtype=bool)
    target = int(T * ratio)
    attempts = 0
    while mask.sum() < target and attempts < 10 * T:
        span_len = int(rng.integers(span_min, span_max + 1))
        start = int(rng.integers(0, max(1, T - span_len + 1)))
        mask[start:start + span_len] = True
        attempts += 1
    return mask


class WildPretrainDataset(Dataset):
    def __init__(self, split: str = "train", seed: int = SEED):
        windows = np.load(CACHE_DIR / "windows.npy", mmap_mode="r")
        video_ids = np.load(CACHE_DIR / "video_ids.npy")

        rng = np.random.default_rng(seed)
        unique_videos = np.unique(video_ids)
        rng.shuffle(unique_videos)
        n_val = max(1, int(len(unique_videos) * VAL_FRACTION))
        val_videos = set(unique_videos[:n_val].tolist())

        if split == "train":
            keep = np.array([vid not in val_videos for vid in video_ids])
        elif split == "val":
            keep = np.array([vid in val_videos for vid in video_ids])
        else:
            raise ValueError(f"split inválido: {split}")

        self.indices = np.nonzero(keep)[0]
        self.windows = windows
        self.split = split
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 1))
        print(f"[{split}] {len(self.indices)} janelas | {len(unique_videos) - n_val if split=='train' else n_val} vídeos")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        w = np.asarray(self.windows[self.indices[idx]])  # (T, 225)
        T = w.shape[0]
        mask = span_mask(T, MASK_RATIO, MASK_SPAN_MIN, MASK_SPAN_MAX, self.rng)
        return torch.from_numpy(w.copy()), torch.from_numpy(mask)
