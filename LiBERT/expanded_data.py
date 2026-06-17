"""Carrega V-Librasil (1.364 classes, 3 sinalizadores) e libras_ufop (55 classes
sintéticas `ufop_N`) a partir dos índices gerados pelas issues #2 e #3, aplicando o
mesmo preprocess()+resample() usado pelo MINDS-Libras em minds_data.py. Os .pkl
individuais desses dois datasets ainda estão em formato bruto (landmarks, fps)."""

import pickle
import unicodedata
import re

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, WINDOW
from preprocessing import preprocess
from minds_data import resample

V_LIBRASIL_INDEX = PROJECT_ROOT / "data" / "v_librasil_landmarks" / "index.csv"
UFOP_INDEX = PROJECT_ROOT / "data" / "libras_ufop_landmarks" / "index.csv"


def normalize_class_name(s: str) -> str:
    """Minúsculo, sem acento, espaços colapsados — usado só para detectar colisões
    de glosa entre datasets, não como nome final de classe."""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _resolve_pkl_path(raw_path: str) -> "Path":
    from pathlib import Path
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_v_librasil():
    """Retorna (X (N,200,225) float32, class_names list[str] len N) — uma entrada por vídeo,
    usando o texto original da glosa (não normalizado) como nome de classe."""
    df = pd.read_csv(V_LIBRASIL_INDEX)
    df = df[df["status"] == "ok"].reset_index(drop=True)

    X = np.empty((len(df), WINDOW, 225), dtype=np.float32)
    class_names = []
    for i, row in df.iterrows():
        path = _resolve_pkl_path(row["pkl_path"])
        landmarks, _fps = pickle.load(open(path, "rb"))
        seq = preprocess(np.asarray(landmarks, dtype=np.float32))
        X[i] = resample(seq, WINDOW)
        class_names.append(row["class"])

    print(f"V-Librasil: {len(df)} amostras, {df['class'].nunique()} classes, "
          f"{df['user_id'].nunique()} sinalizadores")
    return X, class_names


def load_libras_ufop():
    """Retorna (X (N,200,225) float32, class_names list[str] len N) — nomes sintéticos
    `ufop_<label>` já que não há mapeamento label->texto disponível."""
    df = pd.read_csv(UFOP_INDEX)

    X = np.empty((len(df), WINDOW, 225), dtype=np.float32)
    class_names = []
    for i, row in df.iterrows():
        path = _resolve_pkl_path(row["pkl_path"])
        landmarks, _fps = pickle.load(open(path, "rb"))
        seq = preprocess(np.asarray(landmarks, dtype=np.float32))
        X[i] = resample(seq, WINDOW)
        class_names.append(f"ufop_{row['label']}")

    print(f"libras_ufop: {len(df)} amostras, {df['label'].nunique()} classes")
    return X, class_names
