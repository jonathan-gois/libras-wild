"""
Treinamento v2: BiLSTM + Data Augmentation (mirror + noise + stretch).

Melhorias sobre v1:
  - Mirror augmentation (duplica dataset simulando canhotos)
  - Augmentação aleatória (noise, stretch, crop) no training set
  - OneCycleLR scheduler (mais agressivo)
  - Label smoothing aumentado
  - Mixup no espaço de features

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python3 -u src/train_v2.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import json, time
from pathlib import Path

from data_loader import load_dataset, normalize_sequences
from dataset import LibrasDataset
from augmentation import build_augmented_dataset
from models.fusion_model import FusionModel

CFG = dict(
    pkl_path     = "data/processed_data.pkl",
    results_dir  = "results",
    tag          = "v2_bilstm_aug",
    n_folds      = 10,
    epochs       = 80,
    batch_size   = 32,
    lr           = 2e-3,
    weight_decay = 1e-4,
    max_len      = 200,
    gei_H        = 64,
    gei_W        = 48,
    fft_window   = 16,
    fft_step     = 8,
    fft_top_k    = 5,
    gei_embed        = 128,
    temporal_hidden  = 256,
    temporal_embed   = 256,
    static_embed     = 128,
    n_lstm_layers    = 2,
    patience         = 20,
    seed             = 42,
    mirror_aug       = True,   # duplica com espelho
    n_extra_aug      = 1,      # 1 versão extra por amostra
    label_smoothing  = 0.15,
    mixup_alpha      = 0.2,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def mixup_batch(gei, seq, static, lengths, labels, alpha=0.2):
    """Mixup nos embeddings de entrada."""
    if alpha <= 0:
        return gei, seq, static, lengths, labels, None
    lam = np.random.beta(alpha, alpha)
    B = gei.size(0)
    perm = torch.randperm(B, device=gei.device)
    gei2   = lam * gei   + (1 - lam) * gei[perm]
    seq2   = lam * seq   + (1 - lam) * seq[perm]
    static2 = lam * static + (1 - lam) * static[perm]
    return gei2, seq2, static2, lengths, labels, (labels[perm], lam)


def train_epoch(model, loader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for gei, seq, static, lengths, labels in loader:
        gei, seq, static = gei.to(DEVICE), seq.to(DEVICE), static.to(DEVICE)
        lengths, labels = lengths.to(DEVICE), labels.to(DEVICE)

        gei2, seq2, static2, lengths2, lab2, mix = mixup_batch(
            gei, seq, static, lengths, labels, CFG["mixup_alpha"]
        )
        optimizer.zero_grad()
        logits = model(gei2, seq2, static2, lengths2)
        if mix is not None:
            lab_b, lam = mix
            loss = lam * criterion(logits, lab2) + (1 - lam) * criterion(logits, lab_b)
        else:
            loss = criterion(logits, lab2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(lab2)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(lab2)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    all_preds, all_labels = [], []
    for gei, seq, static, lengths, labels in loader:
        gei, seq, static = gei.to(DEVICE), seq.to(DEVICE), static.to(DEVICE)
        lengths, labels = lengths.to(DEVICE), labels.to(DEVICE)
        logits = model(gei, seq, static, lengths)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        n += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / n, correct / n, all_preds, all_labels


def run_cv(sequences, labels, class_names):
    Path(CFG["results_dir"]).mkdir(exist_ok=True)

    # ── Augmentação só no dataset de treino (por fold) ──
    # Para o dataset base pré-computado usamos os originais
    print("Pré-computando features do dataset base...", flush=True)
    base_dataset = LibrasDataset(
        sequences, labels,
        max_len=CFG["max_len"], gei_H=CFG["gei_H"], gei_W=CFG["gei_W"],
        fft_window=CFG["fft_window"], fft_step=CFG["fft_step"],
        fft_top_k=CFG["fft_top_k"],
    )

    # Augmented dataset (treino aumentado)
    print("Gerando dataset aumentado...", flush=True)
    aug_seqs, aug_labels = build_augmented_dataset(
        sequences, labels,
        mirror_all=CFG["mirror_aug"],
        n_extra=CFG["n_extra_aug"],
    )
    print(f"  Dataset base: {len(sequences)} → aumentado: {len(aug_seqs)}", flush=True)
    aug_dataset = LibrasDataset(
        aug_seqs, aug_labels,
        max_len=CFG["max_len"], gei_H=CFG["gei_H"], gei_W=CFG["gei_W"],
        fft_window=CFG["fft_window"], fft_step=CFG["fft_step"],
        fft_top_k=CFG["fft_top_k"],
    )

    gei_dim     = base_dataset.gei.shape[1]
    static_dim  = base_dataset.static.shape[1]
    temporal_in = base_dataset.seq.shape[2]

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True,
                          random_state=CFG["seed"])
    fold_accs = []
    all_preds_cv, all_labels_cv = [], []

    # Índices de augmentação: os primeiros N são os originais; os N+1..2N são mirror; etc.
    N = len(sequences)

    for fold, (train_idx_base, val_idx) in enumerate(
        skf.split(np.zeros(N), labels)
    ):
        print(f"\n{'='*55}", flush=True)
        print(f"Fold {fold+1}/{CFG['n_folds']}  (val={len(val_idx)})", flush=True)

        # Para treino: usar índices do aug_dataset correspondentes
        # Original: índices train_idx_base
        # Mirror:   N + train_idx_base
        # Extra:    2N + train_idx_base, 3N + train_idx_base, ...
        train_idx_aug = list(train_idx_base)
        if CFG["mirror_aug"]:
            train_idx_aug += (N + train_idx_base).tolist()
        for k in range(CFG["n_extra_aug"]):
            offset = N * (2 + k) if CFG["mirror_aug"] else N * (1 + k)
            train_idx_aug += (offset + train_idx_base).tolist()

        train_loader = DataLoader(aug_dataset, batch_size=CFG["batch_size"],
                                   sampler=SubsetRandomSampler(train_idx_aug))
        val_loader   = DataLoader(base_dataset, batch_size=CFG["batch_size"],
                                   sampler=SubsetRandomSampler(val_idx))

        model = FusionModel(
            gei_dim=gei_dim, temporal_input=temporal_in, static_dim=static_dim,
            n_classes=len(class_names),
            gei_embed=CFG["gei_embed"],
            temporal_hidden=CFG["temporal_hidden"],
            temporal_embed=CFG["temporal_embed"],
            static_embed=CFG["static_embed"],
            n_lstm_layers=CFG["n_lstm_layers"],
        ).to(DEVICE)

        steps_per_epoch = len(train_loader)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                      weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CFG["lr"],
            steps_per_epoch=steps_per_epoch,
            epochs=CFG["epochs"], pct_start=0.1,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

        best_val_acc = 0.0
        best_state   = None
        no_improve   = 0

        for epoch in range(1, CFG["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler)
            va_loss, va_acc, _, _ = eval_epoch(model, val_loader, criterion)

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  E{epoch:3d} tr={tr_acc:.3f} va={va_acc:.3f} best={best_val_acc:.3f} "
                      f"({time.time()-t0:.1f}s)", flush=True)

            if no_improve >= CFG["patience"]:
                print(f"  Early stop epoch {epoch}", flush=True)
                break

        model.load_state_dict(best_state)
        _, final_acc, preds, true_labels = eval_epoch(model, val_loader, criterion)
        fold_accs.append(final_acc)
        all_preds_cv.extend(preds)
        all_labels_cv.extend(true_labels)
        print(f"  Fold {fold+1} acc: {final_acc:.4f}", flush=True)

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n{'='*55}", flush=True)
    print(f"[{CFG['tag']}] CV: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%", flush=True)
    print(f"Fold accs: {[f'{a*100:.1f}' for a in fold_accs]}", flush=True)

    report = classification_report(
        all_labels_cv, all_preds_cv,
        target_names=class_names, digits=4
    )
    print("\n", report, flush=True)

    tag = CFG["tag"]
    results = {"tag": tag, "mean_acc": mean_acc, "std_acc": std_acc,
               "fold_accs": fold_accs, "config": CFG}
    with open(f"{CFG['results_dir']}/{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{CFG['results_dir']}/{tag}_report.txt", "w") as f:
        f.write(f"CV: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%\n\n{report}")

    print(f"Salvo em {CFG['results_dir']}/{tag}.json", flush=True)
    return mean_acc, std_acc


def main():
    set_seed(CFG["seed"])
    seqs, labels, classes = load_dataset(CFG["pkl_path"])
    print(f"{len(seqs)} amostras, {len(classes)} classes", flush=True)
    seqs = normalize_sequences(seqs)
    run_cv(seqs, labels, classes)


if __name__ == "__main__":
    main()
