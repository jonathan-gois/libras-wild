"""
PyTorch Dataset para o pipeline multimodal.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from features.gei import compute_gei
from features.kinematics import kinematic_stats, build_kinematic_sequences
from features.fft_features import sliding_fft


class LibrasDataset(Dataset):
    """
    Pré-computa e armazena todas as features em RAM.

    Parâmetros
    ----------
    sequences  : lista de (T_i, 225) — landmarks normalizados
    labels     : array (N,)
    max_len    : comprimento máximo para padding das sequências temporais
    gei_H, gei_W : resolução da imagem GEI
    fft_window, fft_step, fft_top_k : parâmetros do FFT deslizante
    """

    def __init__(self, sequences, labels,
                 max_len=200,
                 gei_H=64, gei_W=48,
                 fft_window=16, fft_step=8, fft_top_k=5):

        self.labels = torch.tensor(labels, dtype=torch.long)

        print("  Calculando GEI...", flush=True)
        gei_list = [compute_gei(s, gei_H, gei_W) for s in sequences]
        self.gei = torch.tensor(np.stack(gei_list), dtype=torch.float32)

        print("  Calculando features cinemáticas (stats)...", flush=True)
        kin_list = [kinematic_stats(s) for s in sequences]
        kin_arr  = np.stack(kin_list)

        print("  Calculando features FFT...", flush=True)
        fft_list = [sliding_fft(s, fft_window, fft_step, fft_top_k)
                    for s in sequences]
        fft_arr  = np.stack(fft_list)

        static = np.concatenate([kin_arr, fft_arr], axis=1)
        self.static = torch.tensor(static, dtype=torch.float32)

        print("  Construindo sequências temporais (pos+vel+acc)...", flush=True)
        seq_arr = build_kinematic_sequences(sequences, max_len)
        self.seq = torch.tensor(seq_arr, dtype=torch.float32)

        # Comprimentos reais antes do padding (para pack_padded_sequence)
        self.lengths = torch.tensor(
            [min(s.shape[0], max_len) for s in sequences],
            dtype=torch.long
        )

        print(f"  Dataset criado: {len(self.labels)} amostras", flush=True)
        print(f"    GEI shape:    {tuple(self.gei.shape)}")
        print(f"    Seq shape:    {tuple(self.seq.shape)}")
        print(f"    Static shape: {tuple(self.static.shape)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.gei[idx],
            self.seq[idx],
            self.static[idx],
            self.lengths[idx],
            self.labels[idx],
        )
