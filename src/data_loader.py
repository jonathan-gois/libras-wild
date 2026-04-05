"""
Carrega e pré-processa o MINDS-Libras (processed_data.pkl).

Retorna:
  sequences : lista de N arrays (T_i, 225)  — landmarks brutos
  labels    : array (N,) de inteiros
  class_names: lista de 20 strings
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_dataset(pkl_path: str):
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

    le = LabelEncoder()
    y = le.fit_transform(df["class"].values)

    sequences = [np.array(row, dtype=np.float32) for row in df["landmarks"]]

    return sequences, y, list(le.classes_)


def normalize_sequences(sequences: list) -> list:
    """
    Normalização por sequência: centraliza e escala cada landmark pelo
    bounding-box do corpo (robustez a distância da câmera).

    Para cada sequência:
      1. Separa coordenadas pose (primeiros 33*3 = 99 valores)
      2. Calcula centro dos quadris (landmarks 23 e 24 do pose)
      3. Subtrai o centro e divide pela distância ombro-a-ombro
    """
    normed = []
    for seq in sequences:
        seq = seq.copy()
        T = seq.shape[0]
        # Reshape para (T, 75, 3)
        xyz = seq.reshape(T, 75, 3)

        # Landmarks 23 e 24 = quadris (pose), offsets globais 0+23 e 0+24
        hip_center = (xyz[:, 23, :2] + xyz[:, 24, :2]) / 2  # (T, 2)

        # Ombros: landmarks 11 e 12
        shoulder_dist = np.linalg.norm(
            xyz[:, 11, :2] - xyz[:, 12, :2], axis=1, keepdims=True
        ).mean() + 1e-6  # escalar

        # Centralizar x,y
        xyz[:, :, 0] -= hip_center[:, 0:1]
        xyz[:, :, 1] -= hip_center[:, 1:2]
        xyz[:, :, :2] /= shoulder_dist

        normed.append(xyz.reshape(T, 225))
    return normed
