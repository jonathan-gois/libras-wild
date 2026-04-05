"""
Treinamento v3: Transformer + Data Augmentation.

Melhorias sobre v2:
  - TransformerTemporalEncoder (4 camadas, 8 cabeças) no lugar do BiLSTM
  - d_model=256, pre-LayerNorm (mais estável)
  - Token [CLS] para classificação
  - Label smoothing 0.15 + OneCycleLR

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python3 -u src/train_v3.py
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
from models.transformer_model import TransformerFusionModel

CFG = dict(
    pkl_path     = "data/processed_data.pkl",
    results_dir  = "results",
    tag          = "v3_transformer_aug",
    n_folds      = 10,
    epochs       = 80,
    batch_size   = 32,
    lr           = 1e-3,
    weight_decay = 1e-4,
    max_len      = 200,
    gei_H        = 64,
    gei_W        = 48,
    fft_window   = 16,
    fft_step     = 8,
    fft_top_k    = 5,
    gei_embed    = 128,
    d_model      = 256,
    nhead        = 8,
    num_layers   = 4,
    temporal_embed = 256,
    static_embed   = 128,
    dropout        = 0.1,
    patience       = 20,
    seed           = 42,
    mirror_aug     = True,
    n_extra_aug    = 1,
    label_smoothing = 0.15,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def train_epoch(model, loader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for gei, seq, static, lengths, labels in loader:
        gei, seq, static = gei.to(DEVICE), seq.to(DEVICE), static.to(DEVICE)
        lengths, labels = lengths.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(gei, seq, static, lengths)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
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

    print("Pré-computando features...", flush=True)
    base_dataset = LibrasDataset(
        sequences, labels,
        max_len=CFG["max_len"], gei_H=CFG["gei_H"], gei_W=CFG["gei_W"],
        fft_window=CFG["fft_window"], fft_step=CFG["fft_step"],
        fft_top_k=CFG["fft_top_k"],
    )

    print("Gerando dataset aumentado...", flush=True)
    aug_seqs, aug_labels = build_augmented_dataset(
        sequences, labels,
        mirror_all=CFG["mirror_aug"],
        n_extra=CFG["n_extra_aug"],
    )
    print(f"  {len(sequences)} → {len(aug_seqs)} amostras", flush=True)
    aug_dataset = LibrasDataset(
        aug_seqs, aug_labels,
        max_len=CFG["max_len"], gei_H=CFG["gei_H"], gei_W=CFG["gei_W"],
        fft_window=CFG["fft_window"], fft_step=CFG["fft_step"],
        fft_top_k=CFG["fft_top_k"],
    )

    gei_dim     = base_dataset.gei.shape[1]
    static_dim  = base_dataset.static.shape[1]
    temporal_in = base_dataset.seq.shape[2]
    N = len(sequences)

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True,
                          random_state=CFG["seed"])
    fold_accs = []
    all_preds_cv, all_labels_cv = [], []

    for fold, (train_idx_base, val_idx) in enumerate(
        skf.split(np.zeros(N), labels)
    ):
        print(f"\n{'='*55}", flush=True)
        print(f"Fold {fold+1}/{CFG['n_folds']}", flush=True)

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

        model = TransformerFusionModel(
            gei_dim=gei_dim,
            temporal_input=temporal_in,
            static_dim=static_dim,
            n_classes=len(class_names),
            gei_embed=CFG["gei_embed"],
            d_model=CFG["d_model"],
            nhead=CFG["nhead"],
            num_layers=CFG["num_layers"],
            temporal_embed=CFG["temporal_embed"],
            static_embed=CFG["static_embed"],
            dropout=CFG["dropout"],
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
    seqs = normalize_sequences(seqs)
    run_cv(seqs, labels, classes)


if __name__ == "__main__":
    main()
