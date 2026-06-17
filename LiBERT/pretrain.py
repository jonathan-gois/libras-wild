"""Pré-treino do LiBERT via masked span reconstruction sobre o dataset wild.

Uso: env/bin/python LiBERT/pretrain.py
"""

import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    CKPT_DIR, BATCH_SIZE, LR, WEIGHT_DECAY, WARMUP_STEPS, EPOCHS, GRAD_CLIP, SEED,
)
from dataset import WildPretrainDataset
from model import LiBERT


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = __import__("numpy").random.default_rng(SEED + 1000 * worker_id + hash(info.dataset.split) % 1000)


def lr_lambda(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def run_epoch(model, loader, optimizer, scheduler, device, train: bool, scaler=None):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    loss_fn = nn.SmoothL1Loss()

    for x, mask in loader:
        x, mask = x.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            recon, _ = model(x, mask)
            loss = loss_fn(recon[mask], x[mask])

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    torch.manual_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = WildPretrainDataset(split="train")
    val_ds = WildPretrainDataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, worker_init_fn=worker_init_fn,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, worker_init_fn=worker_init_fn,
                             pin_memory=True)

    model = LiBERT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LiBERT: {n_params/1e6:.2f}M parâmetros")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, WARMUP_STEPS, total_steps))

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, scheduler, device, train=False)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{EPOCHS} | train {train_loss:.5f} | val {val_loss:.5f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                   CKPT_DIR / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       CKPT_DIR / "best.pt")

    print(f"Treino concluído. Melhor val loss: {best_val:.5f}")


if __name__ == "__main__":
    main()
