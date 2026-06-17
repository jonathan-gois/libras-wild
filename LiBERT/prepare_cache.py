"""Pré-processa todos os vídeos de dataset/wild (normalize_v3 + kalman_v3) e monta
um cache de janelas deslizantes (WINDOW frames, STRIDE passo) para o pré-treino do LiBERT.

Como nenhum vídeo do wild tem menos de WINDOW frames, todas as janelas são 100% reais
(sem necessidade de padding/attention mask nesta etapa).

Uso: env/bin/python LiBERT/prepare_cache.py
"""

import pickle
import numpy as np
from pathlib import Path

from config import WILD_DIR, CACHE_DIR, WINDOW, STRIDE
from preprocessing import preprocess


def window_starts(T: int, window: int, stride: int) -> list[int]:
    starts = list(range(0, T - window + 1, stride))
    last = T - window
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    video_dirs = sorted(p.parent for p in WILD_DIR.glob("*/landmarks.pkl"))
    print(f"{len(video_dirs)} vídeos com landmarks encontrados em {WILD_DIR}")

    all_windows = []
    video_ids = []        # índice do vídeo de origem de cada janela (p/ split train/val sem leakage)
    video_names = []

    for vid_idx, vdir in enumerate(video_dirs):
        with open(vdir / "landmarks.pkl", "rb") as f:
            seq, fps = pickle.load(f)
        seq = preprocess(seq)
        T = seq.shape[0]
        for s in window_starts(T, WINDOW, STRIDE):
            all_windows.append(seq[s:s + WINDOW])
            video_ids.append(vid_idx)
        video_names.append(vdir.name)
        if (vid_idx + 1) % 20 == 0 or vid_idx == len(video_dirs) - 1:
            print(f"  [{vid_idx+1}/{len(video_dirs)}] {vdir.name}: T={T} frames, "
                  f"{len(window_starts(T, WINDOW, STRIDE))} janelas (total acumulado: {len(all_windows)})")

    windows = np.stack(all_windows, axis=0).astype(np.float32)   # (N, WINDOW, 225)
    video_ids = np.array(video_ids, dtype=np.int32)

    np.save(CACHE_DIR / "windows.npy", windows)
    np.save(CACHE_DIR / "video_ids.npy", video_ids)
    with open(CACHE_DIR / "video_names.pkl", "wb") as f:
        pickle.dump(video_names, f)

    print(f"\nCache salvo em {CACHE_DIR}")
    print(f"windows.npy: shape={windows.shape}, {windows.nbytes / 1e9:.2f} GB")
    print(f"vídeos: {len(video_names)} | janelas: {len(windows)}")


if __name__ == "__main__":
    main()
