"""LiBERT: transformer encoder pré-treinado via masked span reconstruction sobre
sequências de landmarks de pose (225-dim/frame). Token = 1 frame inteiro;
saída inclui o embedding [CLS] (usado depois para fine-tuning/matching)."""

import torch
import torch.nn as nn

from config import FEAT_DIM, D_MODEL, N_LAYERS, N_HEADS, FFN_DIM, DROPOUT, WINDOW


class LiBERT(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT, max_len=WINDOW):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.recon_head = nn.Linear(d_model, feat_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, feat_dim). mask: (B, T) bool, True = frame mascarado.
        Retorna hidden states (B, T+1, d_model), posição 0 = [CLS]."""
        B, T, _ = x.shape
        h = self.input_proj(x)
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), self.mask_token.to(h.dtype), h)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos_embed[:, :T + 1]
        h = self.encoder(h)
        return self.norm(h)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """Pré-treino: reconstrói os frames mascarados. Retorna (recon, cls_embedding)."""
        h = self.encode(x, mask)
        cls_emb = h[:, 0]
        frame_h = h[:, 1:]
        recon = self.recon_head(frame_h)
        return recon, cls_emb

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inferência: embedding [CLS] sem masking (para fine-tuning/matching)."""
        h = self.encode(x, mask=None)
        return h[:, 0]
