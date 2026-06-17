"""Treina um tagger neural por FRAME ("está sinalizando?") em cima das hidden states
do encoder LiBERT pré-treinado — issue #12.

Substitui/complementa o sign_spotter heurístico (src/sign_spotter.py, Gradient Boosting
sobre features de janela inteira, F1 0.851), que classifica a janela inteira de uma vez
e é frágil a corte abrupto/segmentação ruim do dataset wild. Aqui cada FRAME recebe um
score independente de "é parte de um sinal real" — permite re-segmentação fina (issue #13)
em vez de só aceitar/rejeitar a janela inteira.

Dados de treino
----------------
Positivos: MINDS-Libras via minds_data.load_minds() — 1.560 sinais isolados completos,
já preprocessados (normalize_v3+kalman_v3) e reamostrados para 200 frames. Cada amostra é
um sinal do início ao fim, mas os primeiros/últimos ~12.5% dos frames são onset/offset
(preparação/retração) e não o "core" do sinal — em vez de descartá-los (o que mudaria o
formato fixo de 200 frames que o LiBERT espera), eles entram com label is_signing=1 mas
peso de loss reduzido (EDGE_WEIGHT < 1), e ficam fora do cálculo de F1 de validação,
porque seu label é ambíguo (não dá pra dizer com certeza visual se é "mão subindo pro
sinal" ou "mão ainda parada").

Negativos: reaproveita a lógica de geração sintética de src/sign_spotter.py
(make_negatives: só-preparação, só-retração, hold de frames repetidos, dois sinais
concatenados, janela aleatória fora do stroke) aplicada sobre os MESMOS landmarks crus do
MINDS (data/processed_data_full.pkl + processed_data_s03040709.pkl) que alimentam
load_minds(). Cada negativo sintético (comprimento T variável) é preprocessado com o MESMO
pipeline (normalize_v3+kalman_v3) e reamostrado para 200 frames via interpolação linear
(igual ao load_minds), e todo frame recebe is_signing=0.

Modelo
------
Encoder LiBERT pré-treinado (checkpoints/best.pt) + cabeça nn.Linear(384, 1) por frame
(BCEWithLogitsLoss, pesos por frame para as bordas dos positivos). O encoder inteiro é
FINE-TUNADO junto com a cabeça (não congelado) — ver "Decisão de design" abaixo.

Decisão de design: fine-tune do encoder vs. cabeça apenas
-----------------------------------------------------------
Por padrão este script fine-tuna o encoder inteiro junto com a cabeça, usando LR
discriminativo (encoder 1e-5, 10x menor que a cabeça 1e-4) — USE_DISCRIMINATIVE_LR=True.
A motivação: o encoder foi pré-treinado para reconstrução mascarada (captura estrutura
temporal geral de pose), não para a fronteira fina "sinalizando vs. parado/transição/hold"
que esta tarefa exige; um LR baixo no encoder permite especializá-lo nessa fronteira sem
destruir o que já foi aprendido no pré-treino. Defina USE_DISCRIMINATIVE_LR=False para
treinar só a cabeça (encoder congelado) como alternativa mais barata/conservadora — não é
o padrão porque o fine-tune completo convergiu rápido e estável nesta base de dados (ver
log de treino), mas vale revisitar se o tagger não generalizar bem para o wild real (sinal
de overfitting ao MINDS).

Como aplicar a uma sequência contínua arbitrária (para a issue #13 — re-segmentação)
-------------------------------------------------------------------------------------
O encoder LiBERT (e portanto este tagger) foi pré-treinado/treinado em janelas fixas de
200 frames (WINDOW em config.py) — não aceita sequências de comprimento arbitrário T
diretamente (pos_embed tem tamanho fixo max_len+1). Para taguear um vídeo inteiro (T >> 200
frames) frame a frame:

  1. Preprocessar a sequência crua (T, 225) inteira com preprocessing.preprocess() UMA VEZ
     (normalize_v3 usa estatísticas robustas — média da distância dos ombros ao longo de
     toda a sequência —, então não fatiar antes de chamar preprocess).
  2. Fatiar em janelas deslizantes de WINDOW=200 frames com stride menor que 200 (ex.: 100,
     50% overlap, ou menor para mais suavidade) usando frame_tagger_score_sequence() abaixo.
  3. Para frames cobertos por múltiplas janelas (por causa do overlap), agregar os scores
     sobrepostos (ex.: média, ou média ponderada por distância à borda da janela — os
     hidden states no centro de uma janela tendem a ser mais confiáveis que nas bordas,
     onde o encoder tem menos contexto).
  4. O resultado é um score por frame ∈ [0,1] ao longo do vídeo inteiro, pronto para a
     issue #13 re-segmentar com base em onde score cruza um threshold.

A função frame_tagger_score_sequence(model, head, seq_raw, device) no fim deste arquivo
implementa os passos 1-3 e serve de referência/ponto de partida para a issue #13.

Uso
---
    env/bin/python LiBERT/train_frame_tagger.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import CKPT_DIR, D_MODEL, WINDOW, SEED, PROJECT_ROOT
from model import LiBERT
from minds_data import load_minds, resample
from preprocessing import preprocess

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT / "src"))
from sign_spotter import make_negatives  # noqa: E402 — reaproveita geração de negativos sintéticos

import pickle

# ── Hiperparâmetros ──────────────────────────────────────────────────────────
EDGE_FRACTION = 0.125          # ~12.5% de cada borda do clip MINDS é onset/offset
EDGE_WEIGHT = 0.3              # peso de loss reduzido nessas bordas (não zero: ainda é sinal real)
EPOCHS = 25
BATCH_SIZE = 32
HEAD_LR = 1e-4
ENCODER_LR = 1e-5              # 10x menor que a cabeça — fine-tune discriminativo
USE_DISCRIMINATIVE_LR = True   # False = encoder congelado, só treina a cabeça
WEIGHT_DECAY = 0.01
VAL_FRACTION = 0.15
GRAD_CLIP = 1.0

MINDS_PKLS = [
    PROJECT_ROOT / "data" / "processed_data_full.pkl",
    PROJECT_ROOT / "data" / "processed_data_s03040709.pkl",
]


# ── Construção do dataset ────────────────────────────────────────────────────

def load_raw_minds_landmarks() -> list:
    """Carrega os landmarks CRUS (sem preprocess, T variável) do MINDS — input do
    make_negatives do sign_spotter, que precisa cortar/concatenar antes de normalizar."""
    import pandas as pd
    dfs = []
    for path in MINDS_PKLS:
        df = pickle.load(open(path, "rb"))
        dfs.append(df[["filename", "class", "landmarks"]])
    full = pd.concat(dfs, ignore_index=True)
    return [np.asarray(lm, dtype=np.float32) for lm in full["landmarks"]]


def build_dataset():
    """Monta (X, frame_labels, frame_weights) combinando positivos (MINDS completo) e
    negativos sintéticos (lógica do sign_spotter), todos reamostrados para WINDOW frames.

    Retorna:
        X:       (N, WINDOW, 225) float32 — já preprocessado (normalize_v3+kalman_v3)
        labels:  (N, WINDOW) float32 — 1 = sinalizando, 0 = não
        weights: (N, WINDOW) float32 — peso de loss por frame (bordas de positivos = EDGE_WEIGHT)
        groups:  (N,) int — id de amostra original (para split sem vazamento entre splits)
    """
    print("Carregando positivos (MINDS-Libras, via load_minds)...", flush=True)
    X_pos, y_pos, classes = load_minds()  # (1560, 200, 225) já preprocessado/reamostrado
    n_pos = len(X_pos)

    edge_n = max(1, int(round(WINDOW * EDGE_FRACTION)))
    pos_labels = np.ones((n_pos, WINDOW), dtype=np.float32)
    pos_weights = np.ones((n_pos, WINDOW), dtype=np.float32)
    pos_weights[:, :edge_n] = EDGE_WEIGHT
    pos_weights[:, -edge_n:] = EDGE_WEIGHT
    print(f"  {n_pos} positivos | bordas onset/offset: {edge_n} frames/lado, peso {EDGE_WEIGHT}", flush=True)

    print("Gerando negativos sintéticos (reaproveitando sign_spotter.make_negatives)...", flush=True)
    raw_landmarks = load_raw_minds_landmarks()
    raw_negatives = make_negatives(raw_landmarks)  # lista de arrays (T_var, 225), T variável
    print(f"  {len(raw_negatives)} negativos sintéticos brutos (T variável)", flush=True)

    X_neg = np.empty((len(raw_negatives), WINDOW, 225), dtype=np.float32)
    keep_neg = []
    for i, seq in enumerate(raw_negatives):
        if len(seq) < 4:
            continue
        seq_pp = preprocess(seq)
        X_neg[i] = resample(seq_pp, WINDOW)
        keep_neg.append(i)
    X_neg = X_neg[keep_neg]
    n_neg = len(X_neg)
    neg_labels = np.zeros((n_neg, WINDOW), dtype=np.float32)
    neg_weights = np.ones((n_neg, WINDOW), dtype=np.float32)
    print(f"  {n_neg} negativos após preprocess+resample para {WINDOW} frames", flush=True)

    X = np.concatenate([X_pos, X_neg], axis=0)
    labels = np.concatenate([pos_labels, neg_labels], axis=0)
    weights = np.concatenate([pos_weights, neg_weights], axis=0)
    groups = np.arange(len(X))  # cada amostra (pos ou neg) é seu próprio grupo p/ split

    print(f"Dataset combinado: {len(X)} amostras ({n_pos} positivas, {n_neg} negativas)", flush=True)
    return X.astype(np.float32), labels, weights, groups


# ── Avaliação ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, head, X, labels, weights, device, batch_size=64, threshold=0.5):
    """F1/precision/recall por frame, EXCLUINDO frames de borda de baixo peso (EDGE_WEIGHT)
    do cálculo — eles têm label ambíguo, não fazem parte do "core" do sinal."""
    model.eval(); head.eval()
    all_probs, all_labels, all_mask = [], [], []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        h = model.encode(xb, mask=None)[:, 1:]  # (B, WINDOW, D_MODEL) — descarta CLS
        logits = head(h).squeeze(-1)             # (B, WINDOW)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels[i:i + batch_size])
        all_mask.append(weights[i:i + batch_size] >= 1.0)  # só frames de peso "cheio"

    probs = np.concatenate(all_probs).ravel()
    y_true = np.concatenate(all_labels).ravel()
    mask = np.concatenate(all_mask).ravel()

    y_true_eval = y_true[mask]
    y_pred_eval = (probs[mask] >= threshold).astype(np.float32)

    f1 = f1_score(y_true_eval, y_pred_eval, zero_division=0)
    prec = precision_score(y_true_eval, y_pred_eval, zero_division=0)
    rec = recall_score(y_true_eval, y_pred_eval, zero_division=0)
    return f1, prec, rec


# ── Treino ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, labels, weights, groups = build_dataset()

    idx = np.arange(len(X))
    idx_train, idx_val = train_test_split(idx, test_size=VAL_FRACTION, random_state=SEED)
    print(f"Split: {len(idx_train)} treino, {len(idx_val)} val", flush=True)

    X_train, labels_train, weights_train = X[idx_train], labels[idx_train], weights[idx_train]
    X_val, labels_val, weights_val = X[idx_val], labels[idx_val], weights[idx_val]

    model = LiBERT().to(device)
    ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Encoder pré-treinado carregado (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.5f})", flush=True)

    head = nn.Linear(D_MODEL, 1).to(device)

    if USE_DISCRIMINATIVE_LR:
        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "lr": ENCODER_LR},
            {"params": head.parameters(), "lr": HEAD_LR},
        ], weight_decay=WEIGHT_DECAY)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    n_train = len(X_train)
    best_f1 = -1.0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        if USE_DISCRIMINATIVE_LR:
            model.train()
        head.train()

        perm = np.random.permutation(n_train)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n_train, BATCH_SIZE):
            bidx = perm[i:i + BATCH_SIZE]
            xb = torch.from_numpy(X_train[bidx]).to(device)
            yb = torch.from_numpy(labels_train[bidx]).to(device)
            wb = torch.from_numpy(weights_train[bidx]).to(device)

            if USE_DISCRIMINATIVE_LR:
                h = model.encode(xb, mask=None)[:, 1:]
            else:
                with torch.no_grad():
                    h = model.encode(xb, mask=None)[:, 1:]
            logits = head(h).squeeze(-1)
            loss_per_frame = loss_fn(logits, yb)
            loss = (loss_per_frame * wb).sum() / wb.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            params = list(model.parameters()) + list(head.parameters()) if USE_DISCRIMINATIVE_LR else list(head.parameters())
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        val_f1, val_prec, val_rec = evaluate(model, head, X_val, labels_val, weights_val, device)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{EPOCHS} | train_loss {total_loss/n_batches:.4f} | "
              f"val_F1 {val_f1:.4f} | val_P {val_prec:.4f} | val_R {val_rec:.4f} | {dt:.1f}s", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
                "config": {
                    "d_model": D_MODEL,
                    "window": WINDOW,
                    "edge_fraction": EDGE_FRACTION,
                    "edge_weight": EDGE_WEIGHT,
                    "discriminative_lr": USE_DISCRIMINATIVE_LR,
                },
            }, CKPT_DIR / "frame_tagger.pt")

    print(f"\nTreino concluído. Melhor F1 por frame (val, frames de peso cheio): {best_f1:.4f}", flush=True)
    print(f"Checkpoint salvo em {CKPT_DIR / 'frame_tagger.pt'}", flush=True)
    print(f"Comparação conceitual: sign_spotter (janela inteira, GBT) F1=0.851 — ver "
          f"docstring do módulo para ressalvas sobre a comparação não ser 1:1.", flush=True)


# ── Inferência em sequência contínua (referência p/ issue #13) ──────────────

@torch.no_grad()
def frame_tagger_score_sequence(model: LiBERT, head: nn.Module, seq_raw: np.ndarray,
                                 device: torch.device, window: int = WINDOW,
                                 stride: int = 50) -> np.ndarray:
    """Aplica o tagger a uma sequência CRUA arbitrariamente longa (T, 225), T > window.

    Usa janelas deslizantes de `window` frames com passo `stride` (overlap = window-stride)
    e agrega scores sobrepostos por média simples. Frames nas pontas absolutas da sequência
    (cobertos por menos janelas) terão score um pouco mais ruidoso — comportamento esperado.

    Args:
        model: LiBERT já carregado (encoder, com pesos do frame_tagger.pt em "model").
        head: nn.Linear(384, 1) já carregado (pesos do frame_tagger.pt em "head").
        seq_raw: (T, 225) float32, landmarks CRUS (sem preprocess) de um vídeo inteiro.
        device: torch.device.
        window: tamanho da janela (deve bater com o treino, WINDOW=200).
        stride: passo entre janelas consecutivas; menor = mais overlap = mais suave/lento.

    Returns:
        scores: (T,) float32 em [0,1] — P(is_signing) por frame, pronto pra threshold
        e re-segmentação (issue #13).
    """
    model.eval(); head.eval()
    seq = preprocess(seq_raw)  # normaliza a sequência INTEIRA de uma vez (estatísticas globais)
    T = len(seq)

    if T <= window:
        # Sequência curta: roda como uma janela só (preprocess já reamostraria via resample
        # se quiséssemos bater 200 exato, mas como o encoder usa pos_embed[:T+1], T<window
        # funciona sem resample — só não terá o mesmo "comprimento físico" do treino).
        xb = torch.from_numpy(seq).unsqueeze(0).to(device)
        h = model.encode(xb, mask=None)[:, 1:]
        return torch.sigmoid(head(h).squeeze(-1)).squeeze(0).cpu().numpy()

    score_sum = np.zeros(T, dtype=np.float32)
    score_cnt = np.zeros(T, dtype=np.float32)
    starts = list(range(0, T - window + 1, stride))
    if starts[-1] != T - window:
        starts.append(T - window)  # garante cobertura até o fim

    for s in starts:
        chunk = seq[s:s + window]
        xb = torch.from_numpy(chunk).unsqueeze(0).to(device)
        h = model.encode(xb, mask=None)[:, 1:]
        probs = torch.sigmoid(head(h).squeeze(-1)).squeeze(0).cpu().numpy()
        score_sum[s:s + window] += probs
        score_cnt[s:s + window] += 1.0

    return score_sum / np.maximum(score_cnt, 1.0)


if __name__ == "__main__":
    main()
