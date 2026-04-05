"""
Treinamento v5: ST-GCN + MobileNetV3-Small + FFT MLP.

Diferenças sobre v3/v4:
  - GEI → MobileNetV3-Small (pré-treinado ImageNet, fine-tuned)
  - Temporal → STGCNEncoder (grafo anatômico + Transformer temporal)
  - Sem mirror augmentation (ver análise em RESULTADOS.md)

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python3 -u src/train_v5.py 2>&1 | tee results/train_v5.log
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import json, time
from pathlib import Path

from data_loader import load_dataset, normalize_sequences
from dataset import LibrasDataset
from augmentation import build_augmented_dataset
from models.fusion_v5 import FusionModelV5

CFG = dict(
    pkl_path        = "data/processed_data.pkl",
    results_dir     = "results",
    tag             = "v5_stgcn_mobilenet",
    n_folds         = 10,
    epochs          = 60,
    batch_size      = 16,
    lr              = 5e-4,
    weight_decay    = 1e-4,
    max_len         = 200,
    gei_H           = 64,
    gei_W           = 48,
    fft_window      = 16,
    fft_step        = 8,
    fft_top_k       = 5,
    # ST-GCN
    gcn_dims        = (64, 128),
    d_model         = 128,
    nhead           = 4,
    num_layers      = 2,
    # embeddings
    gei_embed       = 128,
    temporal_embed  = 256,
    static_embed    = 128,
    dropout         = 0.1,
    patience        = 15,
    seed            = 42,
    # augmentação (sem mirror)
    mirror_aug      = False,
    n_extra_aug     = 1,   # 1600 amostras treino (vs 2400) — reduz tempo ~40%
    label_smoothing = 0.15,
    freeze_cnn      = False,   # fine-tune toda a MobileNet
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
        lengths, labels  = lengths.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(gei, seq, static, lengths)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    all_preds, all_labels  = [], []
    for gei, seq, static, lengths, labels in loader:
        gei, seq, static = gei.to(DEVICE), seq.to(DEVICE), static.to(DEVICE)
        lengths, labels  = lengths.to(DEVICE), labels.to(DEVICE)
        logits = model(gei, seq, static, lengths)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        n       += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / n, correct / n, all_preds, all_labels


@torch.no_grad()
def precompute_mobilenet_gei(gei_array: np.ndarray, model: FusionModelV5,
                              batch_size: int = 64) -> np.ndarray:
    """
    Extrai features da MobileNetV3 para todos os GEIs de uma vez.
    Elimina o custo da CNN do loop de treino (3-4x mais rápido).
    Retorna: (N, 576) np.ndarray
    """
    enc = model.gei_encoder
    enc.features.eval()
    enc.avgpool.eval()
    feats = []
    for i in range(0, len(gei_array), batch_size):
        chunk = gei_array[i:i+batch_size]
        if isinstance(chunk, torch.Tensor):
            batch = chunk.to(DEVICE)
        else:
            batch = torch.from_numpy(chunk).to(DEVICE)
        B = batch.shape[0]
        img = batch.view(B, 1, 64, 48).expand(-1, 3, -1, -1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)
        img  = (img - mean) / std
        img  = F.interpolate(img, size=(112, 112), mode="bilinear", align_corners=False)
        f = enc.features(img)
        f = enc.avgpool(f).flatten(1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)   # (N, 576)


def run_cv(sequences, labels, class_names):
    Path(CFG["results_dir"]).mkdir(exist_ok=True)

    print("Pré-computando features base...", flush=True)
    base_dataset = LibrasDataset(
        sequences, labels,
        max_len=CFG["max_len"], gei_H=CFG["gei_H"], gei_W=CFG["gei_W"],
        fft_window=CFG["fft_window"], fft_step=CFG["fft_step"],
        fft_top_k=CFG["fft_top_k"],
    )
    print(f"  GEI shape:    {base_dataset.gei.shape}", flush=True)
    print(f"  Seq shape:    {base_dataset.seq.shape}", flush=True)
    print(f"  Static shape: {base_dataset.static.shape}", flush=True)

    print("Gerando dataset aumentado (sem mirror)...", flush=True)
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

    static_dim  = base_dataset.static.shape[1]
    N = len(sequences)

    # ── Pré-extração única das features MobileNetV3 ─────────────────────────
    print("\nPré-extraindo features MobileNetV3 (one-shot, sem backprop)...", flush=True)
    _tmp_model = FusionModelV5(
        gei_embed=CFG["gei_embed"], temporal_embed=CFG["temporal_embed"],
        static_dim=static_dim, static_embed=CFG["static_embed"],
        n_classes=len(class_names), gcn_dims=CFG["gcn_dims"],
        d_model=CFG["d_model"], nhead=CFG["nhead"], num_layers=CFG["num_layers"],
        dropout=CFG["dropout"],
    ).to(DEVICE)

    t0 = time.time()
    base_gei_feat = precompute_mobilenet_gei(base_dataset.gei, _tmp_model)
    aug_gei_feat  = precompute_mobilenet_gei(aug_dataset.gei, _tmp_model)
    del _tmp_model
    print(f"  {len(base_gei_feat)+len(aug_gei_feat)} GEIs → features (576d) em {time.time()-t0:.1f}s", flush=True)

    # Substitui GEI nos datasets por features pré-extraídas (576d em vez de 3072d)
    base_dataset.gei = base_gei_feat
    aug_dataset.gei  = aug_gei_feat

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
        # n_extra augmentations (mirror=False, so offset starts at N*1)
        for k in range(CFG["n_extra_aug"]):
            offset = N * (1 + k)
            train_idx_aug += (offset + train_idx_base).tolist()

        train_loader = DataLoader(aug_dataset, batch_size=CFG["batch_size"],
                                  sampler=SubsetRandomSampler(train_idx_aug))
        val_loader   = DataLoader(base_dataset, batch_size=CFG["batch_size"],
                                  sampler=SubsetRandomSampler(val_idx))

        model = FusionModelV5(
            gei_embed      = CFG["gei_embed"],
            temporal_embed = CFG["temporal_embed"],
            static_dim     = static_dim,
            static_embed   = CFG["static_embed"],
            n_classes      = len(class_names),
            gcn_dims       = CFG["gcn_dims"],
            d_model        = CFG["d_model"],
            nhead          = CFG["nhead"],
            num_layers     = CFG["num_layers"],
            dropout        = CFG["dropout"],
            freeze_cnn     = CFG["freeze_cnn"],
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
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer,
                                          criterion, scheduler)
            va_loss, va_acc, _, _ = eval_epoch(model, val_loader, criterion)

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  E{epoch:3d} tr={tr_acc:.3f} va={va_acc:.3f} "
                      f"best={best_val_acc:.3f} ({time.time()-t0:.1f}s)", flush=True)

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

    tag     = CFG["tag"]
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
