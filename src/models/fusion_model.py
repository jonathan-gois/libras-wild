"""
Modelo de fusão multimodal:

  GEI branch   (64×48 → CNN → embedding)
+ Temporal branch (pos+vel+acc → BiLSTM → embedding)
+ Static branch (kinematic_stats + FFT stats → MLP → embedding)
   ↓
Concat → Dropout → FC → 20 classes

Também expõe um extractor de features planas para uso com
classificadores clássicos (sklearn).
"""

import torch
import torch.nn as nn
import numpy as np


class GEIEncoder(nn.Module):
    """Mini-CNN para a imagem GEI (64×48 flat)."""

    def __init__(self, input_dim: int = 3072, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):  # (B, 3072)
        return self.net(x)  # (B, embed_dim)


class TemporalEncoder(nn.Module):
    """BiLSTM sobre a sequência (pos + vel + acc)."""

    def __init__(self, input_size: int = 675,  # 225 * 3
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 embed_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x, lengths=None):  # x: (B, T, 675)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, (h, _) = self.lstm(packed)
        else:
            out, (h, _) = self.lstm(x)

        # Pega último estado oculto das duas direções
        h_fwd = h[-2]  # (B, hidden_size)
        h_bwd = h[-1]  # (B, hidden_size)
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # (B, hidden_size*2)
        return self.proj(h_cat)  # (B, embed_dim)


class StaticEncoder(nn.Module):
    """MLP para features estáticas (kinematic stats + FFT)."""

    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):  # (B, input_dim)
        return self.net(x)


class FusionModel(nn.Module):
    """
    Modelo de fusão: GEI + Temporal + Estático → 20 classes.
    """

    def __init__(self,
                 gei_dim: int = 3072,
                 temporal_input: int = 675,
                 static_dim: int = 7650,
                 n_classes: int = 20,
                 gei_embed: int = 128,
                 temporal_hidden: int = 256,
                 temporal_embed: int = 256,
                 static_embed: int = 128,
                 n_lstm_layers: int = 2):
        super().__init__()

        self.gei_enc = GEIEncoder(gei_dim, gei_embed)
        self.temporal_enc = TemporalEncoder(
            temporal_input, temporal_hidden, n_lstm_layers, temporal_embed
        )
        self.static_enc = StaticEncoder(static_dim, static_embed)

        total_embed = gei_embed + temporal_embed + static_embed
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(total_embed, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, gei, seq, static, lengths=None):
        g = self.gei_enc(gei)
        t = self.temporal_enc(seq, lengths)
        s = self.static_enc(static)
        fused = torch.cat([g, t, s], dim=1)
        return self.classifier(fused)
