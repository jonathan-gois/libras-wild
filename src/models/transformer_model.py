"""
Modelo Transformer multimodal para reconhecimento de Libras.

Arquitetura:
  - GEI branch:      Linear(3072→512) + LayerNorm + ReLU → embed(128)
  - Temporal branch: Positional Encoding + Transformer Encoder → CLS token → embed(256)
  - Static branch:   MLP(kinematic_stats + FFT) → embed(128)
  - Fusion:          Concat(512) → Dropout → FC → 20 classes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 300, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTemporalEncoder(nn.Module):
    """
    Transformer Encoder com token [CLS] para classificação.
    Input: (B, T, input_size) → output: (B, embed_dim)
    """

    def __init__(self,
                 input_size: int = 675,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 embed_dim: int = 256):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # [CLS] token learnable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN (mais estável)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x, lengths=None):  # x: (B, T, input_size)
        B, T, _ = x.shape
        x = self.input_proj(x)       # (B, T, d_model)
        x = self.pos_enc(x)

        # Prepend [CLS]
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)           # (B, T+1, d_model)

        # Máscara de padding (True = ignorar)
        if lengths is not None:
            T_padded = T + 1  # +1 pelo CLS
            mask = torch.ones(B, T_padded, dtype=torch.bool, device=x.device)
            mask[:, 0] = False  # CLS sempre visível
            for i, l in enumerate(lengths):
                mask[i, 1:l+1] = False  # frames reais visíveis
        else:
            mask = None

        out = self.transformer(x, src_key_padding_mask=mask)  # (B, T+1, d_model)
        cls_out = out[:, 0]  # Pega o [CLS] token
        return self.proj(cls_out)  # (B, embed_dim)


class GEIEncoder(nn.Module):
    def __init__(self, input_dim: int = 3072, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class StaticEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerFusionModel(nn.Module):
    """
    Modelo de fusão multimodal com Transformer.
    GEI + Transformer(pos+vel+acc) + Static(kinematic_stats+FFT) → 20 classes
    """

    def __init__(self,
                 gei_dim: int = 3072,
                 temporal_input: int = 675,
                 static_dim: int = 8100,
                 n_classes: int = 20,
                 gei_embed: int = 128,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 temporal_embed: int = 256,
                 static_embed: int = 128,
                 dropout: float = 0.1):
        super().__init__()

        self.gei_enc = GEIEncoder(gei_dim, gei_embed)
        self.temporal_enc = TransformerTemporalEncoder(
            input_size=temporal_input,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            embed_dim=temporal_embed,
            dropout=dropout,
        )
        self.static_enc = StaticEncoder(static_dim, static_embed)

        total = gei_embed + temporal_embed + static_embed
        self.classifier = nn.Sequential(
            nn.LayerNorm(total),
            nn.Dropout(0.4),
            nn.Linear(total, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, gei, seq, static, lengths=None):
        g = self.gei_enc(gei)
        t = self.temporal_enc(seq, lengths)
        s = self.static_enc(static)
        fused = torch.cat([g, t, s], dim=1)
        return self.classifier(fused)
