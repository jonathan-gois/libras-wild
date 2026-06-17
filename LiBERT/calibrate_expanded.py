"""Fase 2 expandida (issue #5) — recalibração do banco de protótipos com vocabulário
muito maior: MINDS-Libras (20 classes) + V-Librasil (1.364 classes, 3 sinalizadores)
+ libras_ufop (55 classes sintéticas `ufop_N`). Mesma lógica de calibrate.py
(fine-tune de classificação sobre o encoder pré-treinado, depois protótipos = média
de embedding por classe), mas:

  - classes de glosa idêntica entre MINDS-Libras e V-Librasil (ex: "amarelo") são
    mescladas num único protótipo/classe;
  - classes com < 2 amostras vão inteiras pro treino (não dá pra estratificar
    val com elas) — documentado no log;
  - épocas/lr ajustados para o volume ~5x maior de dados e vocabulário ~70x maior.

Não sobrescreve os artefatos da Fase 2 original (20 classes): salva em
checkpoints/calibrated_expanded.pt e checkpoints/prototypes_expanded.pkl.

Uso: env/bin/python LiBERT/calibrate_expanded.py
"""

import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from config import CKPT_DIR, SEED, D_MODEL
from model import LiBERT
from minds_data import load_minds
from expanded_data import load_v_librasil, load_libras_ufop, normalize_class_name

EPOCHS = 25
BATCH_SIZE = 64
LR = 1e-4
VAL_FRACTION = 0.15
OLD_THRESHOLD = 0.624


def build_combined_dataset():
    """Carrega os 3 datasets, resolve colisões de nome de classe (MINDS <-> V-Librasil)
    e devolve (X, y, classes, collision_report)."""
    X_minds, y_minds_idx, minds_classes = load_minds()

    X_v, v_class_names = load_v_librasil()
    X_ufop, ufop_class_names = load_libras_ufop()

    # --- Resolve nomes finais de classe -------------------------------------------
    # Mapa normalizado -> nome final de classe (usa o nome do MINDS-Libras quando colide).
    minds_norm_to_name = {normalize_class_name(c): c for c in minds_classes}

    final_class_set = list(minds_classes)  # preserva ordem/identidade do MINDS-Libras
    final_class_index = {c: i for i, c in enumerate(final_class_set)}

    collisions = {}  # nome_final -> set de nomes originais de V-Librasil que colidiram
    v_final_names = []
    for raw_name in v_class_names:
        norm = normalize_class_name(raw_name)
        if norm in minds_norm_to_name:
            final_name = minds_norm_to_name[norm]
            collisions.setdefault(final_name, set()).add(raw_name)
        else:
            final_name = raw_name
            if final_name not in final_class_index:
                final_class_index[final_name] = len(final_class_set)
                final_class_set.append(final_name)
        v_final_names.append(final_name)

    # libras_ufop: nomes sintéticos, nunca colidem.
    for raw_name in ufop_class_names:
        if raw_name not in final_class_index:
            final_class_index[raw_name] = len(final_class_set)
            final_class_set.append(raw_name)

    classes = final_class_set
    n_classes = len(classes)

    # --- Monta X, y combinados -----------------------------------------------------
    X_parts = [X_minds, X_v, X_ufop]
    y_parts = [
        np.array([final_class_index[minds_classes[i]] for i in y_minds_idx], dtype=np.int64),
        np.array([final_class_index[n] for n in v_final_names], dtype=np.int64),
        np.array([final_class_index[n] for n in ufop_class_names], dtype=np.int64),
    ]
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    collision_report = {k: sorted(v) for k, v in collisions.items()}
    return X, y, classes, collision_report


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, y, classes, collisions = build_combined_dataset()
    n_classes = len(classes)

    print(f"\nDataset combinado: {len(X)} amostras, {n_classes} classes "
          f"(MINDS 20 + V-Librasil ~1364 + libras_ufop 55 - {len(collisions)} colisões)")
    print(f"\nColisões de glosa MINDS-Libras <-> V-Librasil ({len(collisions)}):")
    for final_name, raw_names in sorted(collisions.items()):
        print(f"  '{final_name}'  <-  V-Librasil: {raw_names}")

    # --- Split treino/val estratificado por classe (manual) ------------------------
    # sklearn.train_test_split com stratify global exige n_classes <= test_size*N e
    # <= (1-test_size)*N, o que falha aqui (1432 classes, muitas com só 2-3 amostras).
    # Em vez disso, decide o split classe a classe: só entra amostra em val se a
    # classe tiver amostras suficientes para garantir >=1 em cada lado do split
    # (round(n*VAL_FRACTION) >= 1, ou seja n >= ceil(1/VAL_FRACTION)).
    rng = np.random.RandomState(SEED)
    counts = Counter(y.tolist())
    min_n_for_split = int(np.ceil(1 / VAL_FRACTION))  # = 7 para VAL_FRACTION=0.15

    idx_train_list, idx_val_list = [], []
    n_rare_classes = 0
    n_rare_samples = 0
    for label, n in counts.items():
        idx_c = np.where(y == label)[0]
        if n < min_n_for_split:
            idx_train_list.append(idx_c)
            n_rare_classes += 1
            n_rare_samples += n
            continue
        idx_c = idx_c.copy()
        rng.shuffle(idx_c)
        n_val_c = max(1, int(round(n * VAL_FRACTION)))
        idx_val_list.append(idx_c[:n_val_c])
        idx_train_list.append(idx_c[n_val_c:])

    idx_train = np.concatenate(idx_train_list)
    idx_val = np.concatenate(idx_val_list) if idx_val_list else np.array([], dtype=np.int64)

    print(f"\nClasses com < {min_n_for_split} amostras (foram inteiras pro treino, sem entrar "
          f"no split de val): {n_rare_classes} classes, {n_rare_samples} amostras")
    print(f"Split: {len(idx_train)} treino / {len(idx_val)} val "
          f"(de {len(X)} amostras totais, {n_classes} classes)")

    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    train_ds = TensorDataset(X_t[idx_train], y_t[idx_train])
    val_ds = TensorDataset(X_t[idx_val], y_t[idx_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LiBERT().to(device)
    ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"\nEncoder pré-treinado carregado (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.5f})")

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
        val_acc = val_correct / val_n if val_n else float("nan")
        print(f"epoch {epoch:3d}/{EPOCHS} | train_loss {train_loss/train_n:.4f} | "
              f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

        if val_n and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_DIR / "calibrated_expanded.pt")

    print(f"\nMelhor val_acc: {best_val_acc:.4f}")

    # Recarrega o melhor checkpoint calibrado e monta o banco de protótipos
    model.load_state_dict(torch.load(CKPT_DIR / "calibrated_expanded.pt", map_location=device))
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
    proto_norms = np.linalg.norm(proto_mat, axis=1) + 1e-8
    for i in range(len(all_emb)):
        emb = all_emb[i]
        sims = (proto_mat @ emb) / (proto_norms * (np.linalg.norm(emb) + 1e-8))
        true_ci = y[i]
        pos_sims.append(sims[true_ci])
        wrong = sims.copy(); wrong[true_ci] = -1
        neg_sims.append(wrong.max())

    pos_sims, neg_sims = np.array(pos_sims), np.array(neg_sims)
    print(f"\nSimilaridade ao protótipo correto:  média {pos_sims.mean():.3f}  p10 {np.percentile(pos_sims,10):.3f}")
    print(f"Similaridade ao melhor distrator:    média {neg_sims.mean():.3f}  p90 {np.percentile(neg_sims,90):.3f}")
    suggested_threshold = float((np.percentile(pos_sims, 10) + np.percentile(neg_sims, 90)) / 2)
    print(f"Threshold sugerido (ponto médio p10/p90): {suggested_threshold:.3f}")
    print(f"Threshold antigo (20 classes, só MINDS-Libras): {OLD_THRESHOLD:.3f}")
    print(f"Delta: {suggested_threshold - OLD_THRESHOLD:+.3f}")

    with open(CKPT_DIR / "prototypes_expanded.pkl", "wb") as f:
        pickle.dump({
            "prototypes": prototypes,
            "classes": classes,
            "suggested_threshold": suggested_threshold,
            "old_threshold": OLD_THRESHOLD,
            "collisions": collisions,
            "n_rare_classes_no_val_split": n_rare_classes,
            "sources": {"minds_libras": 20, "v_librasil": "~1364", "libras_ufop": 55},
        }, f)
    print(f"\nProtótipos salvos em {CKPT_DIR / 'prototypes_expanded.pkl'}")
    print(f"Vocabulário final: {n_classes} classes")


if __name__ == "__main__":
    main()
