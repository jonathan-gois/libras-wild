"""Carrega o MINDS-Libras (1.560 amostras, 20 classes), pré-processa com o mesmo
normalize_v3+kalman_v3 do LiBERT, e reamostra cada sequência para WINDOW frames
via interpolação linear (não trunca/duplica — evita o bug D2 do pipeline DL original)."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from config import PROJECT_ROOT, WINDOW
from preprocessing import preprocess

MINDS_PKLS = [
    PROJECT_ROOT / "data" / "processed_data_full.pkl",
    PROJECT_ROOT / "data" / "processed_data_s03040709.pkl",
]


def resample(seq: np.ndarray, target_len: int = WINDOW) -> np.ndarray:
    """Interpolação linear por canal de T frames para target_len frames."""
    T = seq.shape[0]
    if T == target_len:
        return seq
    t_old = np.linspace(0, T - 1, T)
    t_new = np.linspace(0, T - 1, target_len)
    out = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(t_new, t_old, seq[:, c])
    return out


def load_minds():
    dfs = []
    for path in MINDS_PKLS:
        df = pickle.load(open(path, "rb"))
        dfs.append(df[["filename", "class", "landmarks"]])
    full = pd.concat(dfs, ignore_index=True)

    classes = sorted(full["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X = np.empty((len(full), WINDOW, 225), dtype=np.float32)
    for i, lm in enumerate(full["landmarks"]):
        seq = preprocess(np.asarray(lm, dtype=np.float32))
        X[i] = resample(seq, WINDOW)
    y = full["class"].map(class_to_idx).to_numpy(dtype=np.int64)

    print(f"MINDS-Libras: {len(full)} amostras, {len(classes)} classes")
    return X, y, classes
