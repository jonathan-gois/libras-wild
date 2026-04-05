"""
Modelo v5: ST-GCN + MobileNetV3-Small + FFT MLP

Fluxo:
  GEI (3072) → MobileNetV3-Small (pré-treinado ImageNet) → proj(128) ──┐
  seq (T×225) → STGCNEncoder (GCN espacial + Transformer temporal) ────┤→ Fusion FC → 20
  FFT+KIN (8100) → MLP(256→128) ──────────────────────────────────────┘

Diferenças sobre v3/v4:
  - GEI processado por CNN pré-treinada (não por MLP flat)
  - Temporal processado por ST-GCN que respeita anatomia do grafo corporal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from .stgcn import STGCNEncoder


class GEIEncoderCNN(nn.Module):
    """
    MobileNetV3-Small pré-treinado aplicado ao GEI.

    Modos:
      - raw_gei=True : recebe GEI flat (3072) → processa CNN completa (lento)
      - raw_gei=False: recebe features pré-extraídas (576) → só projeta (rápido)

    O modo pré-extraído (padrão) é ativado quando train_v5 chama
    precompute_mobilenet_gei() antes do loop de treino.
    """
    def __init__(self, embed_dim: int = 128, freeze_features: bool = False):
        super().__init__()
        mv3 = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        self.features = mv3.features    # usado só na pré-extração
        self.avgpool  = mv3.avgpool

        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(576, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, gei: torch.Tensor) -> torch.Tensor:
        # Aceita tanto GEI flat (3072) quanto features pré-extraídas (576)
        if gei.shape[1] == 576:
            # Modo rápido: features já extraídas pela MobileNet
            return self.proj(gei)

        # Modo completo (lento, para compatibilidade)
        B = gei.shape[0]
        img = gei.view(B, 1, 64, 48).expand(-1, 3, -1, -1)
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=img.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225],
                            device=img.device).view(1,3,1,1)
        img  = (img - mean) / std
        img  = F.interpolate(img, size=(112, 112),
                             mode="bilinear", align_corners=False)
        feat = self.features(img)
        feat = self.avgpool(feat).flatten(1)
        return self.proj(feat)


class StaticEncoderV5(nn.Module):
    """Encoder MLP para FFT + Kinematic stats."""
    def __init__(self, input_dim: int, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionModelV5(nn.Module):
    """
    Modelo de fusão multimodal v5.

    Args:
        gei_embed      : dimensão do embedding GEI (saída CNN)
        temporal_embed : dimensão do embedding ST-GCN
        static_dim     : dimensão do vetor estático (FFT + KIN)
        static_embed   : dimensão do embedding estático
        n_classes      : número de classes (20 para MINDS-Libras)
        gcn_dims       : canais das camadas de grafo
        d_model        : dimensão interna do Transformer temporal
        nhead          : número de cabeças de atenção
        num_layers     : número de camadas do Transformer
        dropout        : taxa de dropout
        freeze_cnn     : congela pesos da MobileNet durante treino
    """
    def __init__(self,
                 gei_embed:      int   = 128,
                 temporal_embed: int   = 256,
                 static_dim:     int   = 8100,
                 static_embed:   int   = 128,
                 n_classes:      int   = 20,
                 gcn_dims:       tuple = (64, 128),
                 d_model:        int   = 128,
                 nhead:          int   = 4,
                 num_layers:     int   = 2,
                 dropout:        float = 0.1,
                 freeze_cnn:     bool  = False):
        super().__init__()

        self.gei_encoder      = GEIEncoderCNN(embed_dim=gei_embed,
                                              freeze_features=freeze_cnn)
        self.temporal_encoder = STGCNEncoder(
            gcn_dims=gcn_dims,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            embed_dim=temporal_embed,
            dropout=dropout,
        )
        self.static_encoder   = StaticEncoderV5(static_dim, static_embed, dropout)

        fusion_dim = gei_embed + temporal_embed + static_embed
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, gei: torch.Tensor,
                seq: torch.Tensor,
                static: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        g = self.gei_encoder(gei)
        t = self.temporal_encoder(seq, lengths)
        s = self.static_encoder(static)
        return self.classifier(torch.cat([g, t, s], dim=1))
