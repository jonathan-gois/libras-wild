"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) para sequências de landmarks.

Arquitetura híbrida eficiente para CPU:
  - SpatialGraphConv: processa relações anatômicas entre landmarks em cada frame
  - Transformer temporal: processa a evolução ao longo do tempo
  - Input: (B, T, 75, 9) onde 9 = pos(3) + vel(3) + acc(3)

Referência: Yan et al. "Spatial Temporal Graph Convolutional Networks for
Skeleton-Based Action Recognition" (AAAI 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Definição do grafo anatômico (75 landmarks MediaPipe Holistic) ────────────
# Pose: 0-32  |  Mão esquerda: 33-53  |  Mão direita: 54-74

POSE_CONNECTIONS = [
    (0,1),(0,4),(1,2),(2,3),(3,7),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(11,23),(12,14),(12,24),(13,15),(14,16),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
    (17,19),(18,20),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(27,31),(28,30),(28,32),(29,31),(30,32),
]

HAND_CONNECTIONS_BASE = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

LH = 33   # offset mão esquerda
RH = 54   # offset mão direita

LHAND_CONNECTIONS = [(a+LH, b+LH) for a,b in HAND_CONNECTIONS_BASE]
RHAND_CONNECTIONS = [(a+RH, b+RH) for a,b in HAND_CONNECTIONS_BASE]

# Conexões cruzadas: pulso na pose → raiz da mão
CROSS_CONNECTIONS = [(15, LH), (16, RH)]

ALL_CONNECTIONS = (POSE_CONNECTIONS + LHAND_CONNECTIONS +
                   RHAND_CONNECTIONS + CROSS_CONNECTIONS)

N_NODES = 75


def build_adjacency(n_nodes: int = N_NODES,
                    connections: list = ALL_CONNECTIONS) -> torch.Tensor:
    """
    Constrói matriz de adjacência normalizada (D^-1 A) para o grafo do corpo.
    Inclui auto-conexões (self-loops).
    """
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    np.fill_diagonal(A, 1.0)          # self-loop
    for a, b in connections:
        A[a, b] = 1.0
        A[b, a] = 1.0
    D = A.sum(axis=1, keepdims=True)
    D = np.where(D > 0, D, 1.0)
    A_norm = A / D                    # normalização por grau
    return torch.from_numpy(A_norm)   # (75, 75)


# ── SpatialGraphConv ─────────────────────────────────────────────────────────

class SpatialGraphConv(nn.Module):
    """
    Convolução espacial no grafo: para cada nó, agrega informação dos vizinhos
    ponderada pela adjacência e projeta com um Linear.

    Input:  (B, T, V, C_in)
    Output: (B, T, V, C_out)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 A: torch.Tensor, residual: bool = True):
        super().__init__()
        self.register_buffer("A", A)                    # (V, V) fixo
        self.fc      = nn.Linear(in_channels, out_channels, bias=False)
        self.bn      = nn.BatchNorm1d(out_channels)
        self.act     = nn.GELU()
        self.residual = (nn.Linear(in_channels, out_channels, bias=False)
                         if residual and in_channels != out_channels else
                         (nn.Identity() if residual else None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, C)
        B, T, V, C = x.shape

        # Agrega vizinhos: x_agg[b,t,v,:] = sum_w A[v,w] * x[b,t,w,:]
        x_agg = torch.einsum("vw,btwc->btvc", self.A, x)   # (B, T, V, C)

        # Projeção linear por nó
        out = self.fc(x_agg)                                # (B, T, V, C_out)

        # BatchNorm1d espera (B, C, *) → reshape
        B2, T2, V2, Co = out.shape
        out = out.reshape(B2 * T2 * V2, Co)
        out = self.bn(out).reshape(B2, T2, V2, Co)
        out = self.act(out)

        if self.residual is not None:
            out = out + self.residual(x)

        return out


# ── STGCNEncoder: GCN espacial → Transformer temporal ───────────────────────

class STGCNEncoder(nn.Module):
    """
    Codificador ST-GCN híbrido:
      1. Compute pos + vel (diferença) como canais de entrada
      2. Duas camadas de SpatialGraphConv
      3. Pooling médio sobre os nós (V → 1)
      4. Transformer Encoder para modelagem temporal
      5. Token [CLS] → projeção final

    Input:  seq  (B, T, 225)   — landmarks flat (pos)
    Output: embed (B, embed_dim)
    """

    def __init__(self,
                 n_nodes:    int = 75,
                 in_coords:  int = 3,     # x,y,z
                 gcn_dims:   tuple = (64, 128),
                 d_model:    int = 128,
                 nhead:      int = 4,
                 num_layers: int = 2,
                 embed_dim:  int = 256,
                 dropout:    float = 0.1,
                 max_len:    int = 201):
        super().__init__()
        self.n_nodes   = n_nodes
        self.in_coords = in_coords

        A = build_adjacency()

        # Canais de entrada: pos + vel → 2 * in_coords
        in_ch = in_coords * 2

        # Camadas de grafo espacial
        gcn_layers = []
        prev_ch = in_ch
        for ch in gcn_dims:
            gcn_layers.append(SpatialGraphConv(prev_ch, ch, A, residual=True))
            prev_ch = ch
        self.gcn = nn.Sequential(*gcn_layers)

        # Projeção para d_model após pooling de nós
        self.node_proj = nn.Linear(prev_ch, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding (learnable)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer temporal (Pre-LN)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Projeção final
        self.proj = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def _compute_vel(self, pos: torch.Tensor) -> torch.Tensor:
        """Velocidade por diferença finita, mantendo T."""
        vel = torch.diff(pos, dim=1)                # (B, T-1, V, C)
        vel = torch.cat([vel[:, :1, :, :], vel], dim=1)  # (B, T, V, C)
        return vel

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # seq: (B, T, 225) or (B, T, 675) [pos+vel+acc] — use only positions
        B, T, F = seq.shape
        V, C = self.n_nodes, self.in_coords

        pos = seq[:, :, :V*C].view(B, T, V, C)  # (B, T, V, 3)
        vel = self._compute_vel(pos)          # (B, T, V, 3)
        x   = torch.cat([pos, vel], dim=-1)  # (B, T, V, 6)

        # Downsample temporal: a cada 4 frames → T_ds = T/4
        # Reduz Transformer de T=200 (501ms) para T=50 (135ms)
        stride = 4
        x = x[:, ::stride, :, :]             # (B, T_ds, V, 6)
        T_ds = x.shape[1]
        if lengths is not None:
            lengths_ds = (lengths / stride).long().clamp(min=1)

        # GCN espacial
        x = self.gcn(x)                      # (B, T_ds, V, gcn_dims[-1])

        # Pooling médio sobre nós
        x = x.mean(dim=2)                    # (B, T_ds, gcn_dims[-1])

        # Projeção → d_model
        x = self.node_proj(x)                # (B, T_ds, d_model)

        # Padding mask (frames além do comprimento real após downsample)
        if lengths is not None:
            pad_mask = torch.arange(T_ds, device=seq.device).unsqueeze(0) >= lengths_ds.unsqueeze(1)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
            pad_mask = torch.cat([cls_mask, pad_mask], dim=1)
        else:
            pad_mask = None

        # CLS token + positional encoding
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)             # (B, T_ds+1, d_model)
        x   = x + self.pos_emb[:, : T_ds + 1, :]

        # Transformer temporal
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x[:, 0, :])   # CLS token

        return self.proj(x)          # (B, embed_dim)
