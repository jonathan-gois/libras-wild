"""Fase 2 — calibração do embedding [CLS]: fine-tune do LiBERT pré-treinado com uma
cabeça de classificação (cross-entropy) sobre o MINDS-Libras (20 classes). A cabeça é
descartada ao final; o que importa é o embedding [CLS] resultante, usado para montar
um banco de protótipos (média por classe) para o matching open-set no wild.

Uso: env/bin/python LiBERT/calibrate.py
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from config import CKPT_DIR, SEED, D_MODEL
from model import LiBERT
from minds_data import load_minds

EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-4
VAL_FRACTION = 0.15


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    X, y, classes = load_minds()
    n_classes = len(classes)

    idx = np.arange(len(X))
    idx_train, idx_val = train_test_split(
        idx, test_size=VAL_FRACTION, stratify=y, random_state=SEED)

    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    train_ds = TensorDataset(X_t[idx_train], y_t[idx_train])
    val_ds = TensorDataset(X_t[idx_val], y_t[idx_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LiBERT().to(device)
    ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Encoder pré-treinado carregado (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.5f})")

    head = nn.Linear(D_MODEL, n_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=LR, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train(); head.train()
        train_loss, train_correct, train_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            cls_emb = model.encode(xb, mask=None)[:, 0]
            logits = head(cls_emb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(yb)
            train_correct += (logits.argmax(-1) == yb).sum().item()
            train_n += len(yb)

        model.eval(); head.eval()
        val_correct, val_n = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                cls_emb = model.encode(xb, mask=None)[:, 0]
                logits = head(cls_emb)
                val_correct += (logits.argmax(-1) == yb).sum().item()
                val_n += len(yb)

        train_acc = train_correct / train_n
        val_acc = val_correct / val_n
        print(f"epoch {epoch:3d}/{EPOCHS} | train_loss {train_loss/train_n:.4f} | "
              f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_DIR / "calibrated.pt")

    print(f"\nMelhor val_acc: {best_val_acc:.4f}")

    # Recarrega o melhor checkpoint calibrado e monta o banco de protótipos
    model.load_state_dict(torch.load(CKPT_DIR / "calibrated.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        all_emb = []
        for i in range(0, len(X_t), BATCH_SIZE):
            xb = X_t[i:i + BATCH_SIZE].to(device)
            all_emb.append(model.encode(xb, mask=None)[:, 0].cpu())
        all_emb = torch.cat(all_emb).numpy()

    prototypes = {}
    for ci, cname in enumerate(classes):
        prototypes[cname] = all_emb[y == ci].mean(axis=0)

    # Threshold de referência: similaridade da amostra ao protótipo da própria classe (positivos)
    # vs ao protótipo mais próximo de uma classe errada (distratores) — só para diagnóstico.
    def cos(a, b):
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    pos_sims, neg_sims = [], []
    proto_mat = np.stack([prototypes[c] for c in classes])
    for i in range(len(all_emb)):
        sims = np.array([cos(all_emb[i], proto_mat[ci]) for ci in range(n_classes)])
        true_ci = y[i]
        pos_sims.append(sims[true_ci])
        wrong = sims.copy(); wrong[true_ci] = -1
        neg_sims.append(wrong.max())

    pos_sims, neg_sims = np.array(pos_sims), np.array(neg_sims)
    print(f"\nSimilaridade ao protótipo correto:  média {pos_sims.mean():.3f}  p10 {np.percentile(pos_sims,10):.3f}")
    print(f"Similaridade ao melhor distrator:    média {neg_sims.mean():.3f}  p90 {np.percentile(neg_sims,90):.3f}")
    suggested_threshold = float((np.percentile(pos_sims, 10) + np.percentile(neg_sims, 90)) / 2)
    print(f"Threshold sugerido (ponto médio p10/p90): {suggested_threshold:.3f}")

    with open(CKPT_DIR / "prototypes.pkl", "wb") as f:
        pickle.dump({
            "prototypes": prototypes,
            "classes": classes,
            "suggested_threshold": suggested_threshold,
        }, f)
    print(f"\nProtótipos salvos em {CKPT_DIR / 'prototypes.pkl'}")


if __name__ == "__main__":
    main()
