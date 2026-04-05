"""
Treinamento do modelo de fusão multimodal (GEI + BiLSTM + FFT).

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python src/train.py

Resultados salvos em results/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import json, time
from pathlib import Path

from data_loader import load_dataset, normalize_sequences
from dataset import LibrasDataset
from models.fusion_model import FusionModel

# ──────────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────────
CFG = dict(
    pkl_path  = "data/processed_data.pkl",
    results_dir = "results",
    n_folds   = 10,
    epochs    = 60,
    batch_size = 32,
    lr        = 1e-3,
    weight_decay = 1e-4,
    max_len   = 200,
    gei_H     = 64,
    gei_W     = 48,
    fft_window = 16,
    fft_step   = 8,
    fft_top_k  = 5,
    # Modelo
    gei_embed       = 128,
    temporal_hidden = 256,
    temporal_embed  = 256,
    static_embed    = 128,
    n_lstm_layers   = 2,
    patience        = 15,   # early stopping
    seed            = 42,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, n = 0, 0, 0
    for gei, seq, static, lengths, labels in loader:
        gei, seq, static = gei.to(DEVICE), seq.to(DEVICE), static.to(DEVICE)
        lengths, labels = lengths.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(gei, seq, static, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

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
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        n += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / n, correct / n, all_preds, all_labels


def run_cv(dataset: LibrasDataset, labels: np.ndarray, class_names: list):
    Path(CFG["results_dir"]).mkdir(exist_ok=True)

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True,
                          random_state=CFG["seed"])
    fold_accs = []
    all_preds_cv, all_labels_cv = [], []

    # dimensões do modelo
    gei_dim    = dataset.gei.shape[1]
    static_dim = dataset.static.shape[1]
    temporal_in = dataset.seq.shape[2]

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{CFG['n_folds']}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")

        train_loader = DataLoader(
            dataset,
            batch_size=CFG["batch_size"],
            sampler=SubsetRandomSampler(train_idx),
        )
        val_loader = DataLoader(
            dataset,
            batch_size=CFG["batch_size"],
            sampler=SubsetRandomSampler(val_idx),
        )

        model = FusionModel(
            gei_dim       = gei_dim,
            temporal_input = temporal_in,
            static_dim    = static_dim,
            n_classes     = len(class_names),
            gei_embed     = CFG["gei_embed"],
            temporal_hidden = CFG["temporal_hidden"],
            temporal_embed  = CFG["temporal_embed"],
            static_embed    = CFG["static_embed"],
            n_lstm_layers   = CFG["n_lstm_layers"],
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG["epochs"]
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = 0.0
        best_state  = None
        no_improve  = 0

        for epoch in range(1, CFG["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
            va_loss, va_acc, _, _ = eval_epoch(model, val_loader, criterion)
            scheduler.step()

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve  = 0
            else:
                no_improve += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | "
                      f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                      f"va_loss={va_loss:.4f} va_acc={va_acc:.3f} | "
                      f"best={best_val_acc:.3f} | {time.time()-t0:.1f}s")

            if no_improve >= CFG["patience"]:
                print(f"  Early stop epoch {epoch}")
                break

        # Avalia com melhor estado
        model.load_state_dict(best_state)
        _, final_acc, preds, true_labels = eval_epoch(model, val_loader, criterion)
        fold_accs.append(final_acc)
        all_preds_cv.extend(preds)
        all_labels_cv.extend(true_labels)
        print(f"  Fold {fold+1} final acc: {final_acc:.4f}")

    # ── Relatório final ──
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n{'='*50}")
    print(f"CV Result: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Fold accs: {[f'{a*100:.1f}%' for a in fold_accs]}")

    report = classification_report(
        all_labels_cv, all_preds_cv,
        target_names=class_names, digits=4
    )
    print("\nClassification Report (agregado):\n", report)

    # Salva resultados
    results = {
        "mean_acc": mean_acc,
        "std_acc":  std_acc,
        "fold_accs": fold_accs,
        "config": CFG,
    }
    with open(f"{CFG['results_dir']}/cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{CFG['results_dir']}/classification_report.txt", "w") as f:
        f.write(f"CV: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%\n\n")
        f.write(report)

    print(f"\nResultados salvos em {CFG['results_dir']}/")
    return mean_acc, std_acc


def main():
    set_seed(CFG["seed"])

    print("Carregando dataset...")
    sequences, labels, class_names = load_dataset(CFG["pkl_path"])
    print(f"  {len(sequences)} amostras, {len(class_names)} classes")
    print(f"  Classes: {class_names}")

    print("Normalizando sequências...")
    sequences = normalize_sequences(sequences)

    print("\nPré-computando features (pode demorar alguns minutos)...")
    dataset = LibrasDataset(
        sequences, labels,
        max_len    = CFG["max_len"],
        gei_H      = CFG["gei_H"],
        gei_W      = CFG["gei_W"],
        fft_window = CFG["fft_window"],
        fft_step   = CFG["fft_step"],
        fft_top_k  = CFG["fft_top_k"],
    )

    print("\nIniciando cross-validation...")
    run_cv(dataset, labels, class_names)


if __name__ == "__main__":
    main()
