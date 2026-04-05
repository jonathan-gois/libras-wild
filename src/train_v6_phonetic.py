"""
Treinamento v6: modelo paramétrico com dicionário fônico.

Usa as anotações do Qwen2-VL (data/phonetic/dictionary.json) para:
  1. Construir embedding semântico de cada classe via CLIP/SBERT
  2. Treinar com perda combinada: classificação + alinhamento paramétrico
  3. Suporte a zero-shot para sinais novos (descrever → reconhecer)

Arquitetura:
  Landmarks → STGCNEncoder + FFT → visual_embed (512d)
                                        ↓
              ┌─────────────────────────┤
              │  Contrastive loss       │
              │  (alinha visual com     │
              │   embedding textual     │
              │   dos parâmetros)       │
              └─────────────────────────┘
                                        ↓
              Classificador → 20 classes + open-vocabulary

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python3 -u src/train_v6_phonetic.py 2>&1 | tee results/train_v6.log
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import time
from pathlib import Path

from data_loader import load_dataset, normalize_sequences
from dataset import LibrasDataset
from augmentation import build_augmented_dataset
from models.stgcn import STGCNEncoder

DICT_FILE   = Path("data/phonetic/dictionary.json")
RESULTS_DIR = Path("results")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

CFG = dict(
    pkl_path        = "data/processed_data.pkl",
    tag             = "v6_phonetic",
    n_folds         = 10,
    epochs          = 80,
    batch_size      = 32,
    lr              = 5e-4,
    weight_decay    = 1e-4,
    max_len         = 200,
    gei_H           = 64,
    gei_W           = 48,
    fft_window      = 16,
    fft_step        = 8,
    fft_top_k       = 5,
    # Arquitetura
    gcn_dims        = (64, 128),
    d_model         = 128,
    nhead           = 4,
    num_tf_layers   = 2,
    temporal_embed  = 256,
    static_embed    = 128,
    visual_embed    = 512,    # espaço compartilhado visual-textual
    dropout         = 0.1,
    patience        = 20,
    seed            = 42,
    # Augmentação
    mirror_aug      = False,
    n_extra_aug     = 2,
    label_smoothing = 0.1,
    # Perda paramétrica
    contrastive_weight = 0.3,  # peso da loss de alinhamento paramétrico
    temperature        = 0.07, # temperatura da InfoNCE
)


# ── Encoder de texto para embeddings paramétricos ───────────────────────────
def build_text_embeddings(class_names: list, dictionary: dict,
                          embed_dim: int = 512) -> torch.Tensor:
    """
    Constrói embedding de texto para cada classe usando SBERT ou CLIP.
    Se o dicionário estiver disponível, usa a descrição fonológica.
    Senão, usa apenas o nome da classe.

    Retorna: (n_classes, embed_dim) tensor normalizado.
    """
    try:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        use_sbert = True
        print("  Usando SBERT multilingual para embeddings textuais", flush=True)
    except ImportError:
        use_sbert = False
        print("  SBERT não disponível — usando embedding aleatório (instale sentence-transformers)", flush=True)

    descriptions = []
    for cname in class_names:
        entry = dictionary.get(cname, {})
        if entry:
            # Monta descrição fonológica estruturada
            parts = []
            if entry.get("mao_dominante"):
                parts.append(f"mão dominante {entry['mao_dominante']}")
            if entry.get("configuracao_mao_dominante"):
                parts.append(entry["configuracao_mao_dominante"])
            if entry.get("ponto_articulacao"):
                parts.append(f"localização: {entry['ponto_articulacao']}")
            if entry.get("movimento"):
                parts.append(f"movimento: {entry['movimento']}")
            if entry.get("orientacao_palma"):
                parts.append(f"palma {entry['orientacao_palma']}")
            if entry.get("expressao_facial"):
                parts.append(f"expressão: {entry['expressao_facial']}")
            desc = f"Sinal '{cname}': " + ", ".join(parts) if parts else cname
        else:
            desc = f"Sinal de Libras: {cname}"
        descriptions.append(desc)

    if use_sbert:
        raw = sbert.encode(descriptions, normalize_embeddings=True,
                           show_progress_bar=False)
        # Projeta para visual_embed se necessário
        if raw.shape[1] != embed_dim:
            # Projeção linear simples
            proj = nn.Linear(raw.shape[1], embed_dim, bias=False)
            nn.init.orthogonal_(proj.weight)
            with torch.no_grad():
                raw = proj(torch.from_numpy(raw)).numpy()
        emb = torch.from_numpy(raw).float()
    else:
        emb = torch.randn(len(class_names), embed_dim)
        emb = F.normalize(emb, dim=1)

    return emb  # (n_classes, embed_dim)


# ── Modelo v6 ────────────────────────────────────────────────────────────────
class PhoneticFusionModel(nn.Module):
    """
    Modelo de fusão com espaço paramétrico compartilhado.

    visual_embed: projeção do espaço visual para o espaço textual
    classifier: cabeça de classificação no espaço visual
    text_protos: protótipos textuais fixos por classe (não treinados)
    """
    def __init__(self, static_dim: int, n_classes: int,
                 text_protos: torch.Tensor, cfg: dict):
        super().__init__()
        self.n_classes = n_classes

        # Encoder temporal (ST-GCN + Transformer)
        self.temporal = STGCNEncoder(
            gcn_dims=cfg["gcn_dims"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_tf_layers"],
            embed_dim=cfg["temporal_embed"],
            dropout=cfg["dropout"],
        )

        # Encoder estático (FFT + KIN)
        self.static_enc = nn.Sequential(
            nn.Linear(static_dim, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(512, cfg["static_embed"]),
            nn.LayerNorm(cfg["static_embed"]), nn.GELU(),
        )

        # Fusão → espaço visual
        fusion_in = cfg["temporal_embed"] + cfg["static_embed"]
        self.visual_proj = nn.Sequential(
            nn.Linear(fusion_in, cfg["visual_embed"]),
            nn.LayerNorm(cfg["visual_embed"]),
        )

        # Cabeça de classificação
        self.classifier = nn.Linear(cfg["visual_embed"], n_classes)

        # Protótipos textuais (fixos — não são parâmetros treináveis)
        self.register_buffer("text_protos", text_protos)  # (n_classes, visual_embed)
        self.temperature = cfg["temperature"]

    def forward(self, seq: torch.Tensor, static: torch.Tensor,
                lengths: torch.Tensor):
        t_emb = self.temporal(seq, lengths)             # (B, temporal_embed)
        s_emb = self.static_enc(static)                 # (B, static_embed)
        fused = torch.cat([t_emb, s_emb], dim=1)        # (B, fusion_in)
        v_emb = self.visual_proj(fused)                 # (B, visual_embed)
        logits = self.classifier(v_emb)                 # (B, n_classes)
        return logits, F.normalize(v_emb, dim=1)        # logits + embedding normalizado

    @torch.no_grad()
    def zero_shot_predict(self, seq, static, lengths,
                          new_text_protos: torch.Tensor = None) -> torch.Tensor:
        """
        Predição zero-shot: cosine similarity entre visual embed e protótipos textuais.
        new_text_protos: (n_new_classes, visual_embed) — pode ser classes nunca vistas
        """
        _, v_norm = self.forward(seq, static, lengths)
        protos = new_text_protos if new_text_protos is not None else self.text_protos
        protos = F.normalize(protos, dim=1)
        return v_norm @ protos.T  # (B, n_classes)


# ── Perda combinada ───────────────────────────────────────────────────────────
class PhoneticLoss(nn.Module):
    """
    Classificação + InfoNCE paramétrico.

    InfoNCE alinha o embedding visual do sinal com o protótipo textual
    da sua classe (e afasta dos outros) — análogo ao CLIP.
    """
    def __init__(self, n_classes: int, label_smoothing: float,
                 contrastive_weight: float, temperature: float):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.cw = contrastive_weight
        self.tau = temperature

    def infonce(self, v_norm: torch.Tensor, text_protos: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """InfoNCE entre embeddings visuais e protótipos textuais das classes."""
        t_norm = F.normalize(text_protos, dim=1)          # (C, D)
        sim    = v_norm @ t_norm.T / self.tau              # (B, C)
        return F.cross_entropy(sim, labels)

    def forward(self, logits, v_norm, text_protos, labels):
        loss_ce   = self.ce(logits, labels)
        loss_info = self.infonce(v_norm, text_protos, labels)
        return loss_ce + self.cw * loss_info


# ── CV loop ───────────────────────────────────────────────────────────────────
def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def run_cv(sequences, labels, class_names, text_protos: torch.Tensor):
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Pré-computando features...", flush=True)
    base_ds = LibrasDataset(sequences, labels,
                            max_len=CFG["max_len"],
                            gei_H=CFG["gei_H"], gei_W=CFG["gei_W"],
                            fft_window=CFG["fft_window"],
                            fft_step=CFG["fft_step"],
                            fft_top_k=CFG["fft_top_k"])

    aug_seqs, aug_labels = build_augmented_dataset(
        sequences, labels,
        mirror_all=CFG["mirror_aug"], n_extra=CFG["n_extra_aug"])
    print(f"  {len(sequences)} → {len(aug_seqs)} amostras (aug)", flush=True)
    aug_ds = LibrasDataset(aug_seqs, aug_labels,
                           max_len=CFG["max_len"],
                           gei_H=CFG["gei_H"], gei_W=CFG["gei_W"],
                           fft_window=CFG["fft_window"],
                           fft_step=CFG["fft_step"],
                           fft_top_k=CFG["fft_top_k"])

    static_dim = base_ds.static.shape[1]
    N = len(sequences)
    text_protos = text_protos.to(DEVICE)

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True,
                          random_state=CFG["seed"])
    fold_accs = []
    all_preds_cv, all_labels_cv = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), labels)):
        print(f"\n{'='*55}", flush=True)
        print(f"Fold {fold+1}/{CFG['n_folds']}", flush=True)

        train_idx_aug = list(train_idx)
        for k in range(CFG["n_extra_aug"]):
            train_idx_aug += (N * (1 + k) + train_idx).tolist()

        train_loader = DataLoader(aug_ds, batch_size=CFG["batch_size"],
                                  sampler=SubsetRandomSampler(train_idx_aug))
        val_loader   = DataLoader(base_ds, batch_size=CFG["batch_size"],
                                  sampler=SubsetRandomSampler(val_idx))

        model = PhoneticFusionModel(
            static_dim=static_dim, n_classes=len(class_names),
            text_protos=text_protos, cfg=CFG,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CFG["lr"],
            steps_per_epoch=len(train_loader),
            epochs=CFG["epochs"], pct_start=0.1)
        criterion = PhoneticLoss(
            n_classes=len(class_names),
            label_smoothing=CFG["label_smoothing"],
            contrastive_weight=CFG["contrastive_weight"],
            temperature=CFG["temperature"])

        best_acc, best_state, no_imp = 0.0, None, 0

        for epoch in range(1, CFG["epochs"] + 1):
            # ── train ──
            model.train()
            tr_correct, tr_n = 0, 0
            for _, seq, static, lengths, lbls in train_loader:
                seq, static = seq.to(DEVICE), static.to(DEVICE)
                lengths, lbls = lengths.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                logits, v_norm = model(seq, static, lengths)
                loss = criterion(logits, v_norm, model.text_protos, lbls)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                tr_correct += (logits.argmax(1) == lbls).sum().item()
                tr_n       += len(lbls)

            # ── val ──
            model.eval()
            val_preds, val_lbls = [], []
            with torch.no_grad():
                for _, seq, static, lengths, lbls in val_loader:
                    seq, static = seq.to(DEVICE), static.to(DEVICE)
                    lengths, lbls = lengths.to(DEVICE), lbls.to(DEVICE)
                    logits, _ = model(seq, static, lengths)
                    val_preds.extend(logits.argmax(1).cpu().numpy())
                    val_lbls.extend(lbls.cpu().numpy())

            va_acc = (np.array(val_preds) == np.array(val_lbls)).mean()
            tr_acc = tr_correct / tr_n

            if va_acc > best_acc:
                best_acc  = va_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  E{epoch:3d} tr={tr_acc:.3f} va={va_acc:.3f} "
                      f"best={best_acc:.3f}", flush=True)

            if no_imp >= CFG["patience"]:
                print(f"  Early stop E{epoch}", flush=True)
                break

        model.load_state_dict(best_state)
        model.eval()
        fold_preds, fold_lbls = [], []
        with torch.no_grad():
            for _, seq, static, lengths, lbls in val_loader:
                seq, static = seq.to(DEVICE), static.to(DEVICE)
                lengths = lengths.to(DEVICE)
                logits, _ = model(seq, static, lengths)
                fold_preds.extend(logits.argmax(1).cpu().numpy())
                fold_lbls.extend(lbls.cpu().numpy())

        acc = (np.array(fold_preds) == np.array(fold_lbls)).mean()
        fold_accs.append(acc)
        all_preds_cv.extend(fold_preds)
        all_labels_cv.extend(fold_lbls)
        print(f"  Fold {fold+1} acc: {acc:.4f}", flush=True)

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n[{CFG['tag']}] CV: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%", flush=True)

    report = classification_report(all_labels_cv, all_preds_cv,
                                   target_names=class_names, digits=4)
    print(report, flush=True)

    tag = CFG["tag"]
    results = {"tag": tag, "mean_acc": float(mean_acc), "std_acc": float(std_acc),
               "fold_accs": [float(a) for a in fold_accs], "config": CFG}
    with open(RESULTS_DIR / f"{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_DIR / f"{tag}_report.txt", "w") as f:
        f.write(f"CV: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%\n\n{report}")
    print(f"Salvo em {RESULTS_DIR}/{tag}.json", flush=True)


def main():
    set_seed(CFG["seed"])
    seqs, labels, class_names = load_dataset(CFG["pkl_path"])
    seqs = normalize_sequences(seqs)

    # Carrega dicionário fônico (se disponível)
    if DICT_FILE.exists():
        with open(DICT_FILE) as f:
            dictionary = json.load(f)
        print(f"Dicionário fônico: {len(dictionary)} classes", flush=True)
    else:
        print("AVISO: dicionário fônico não encontrado — usando nomes das classes.", flush=True)
        print("       Execute annotate_qwen.py primeiro para melhores resultados.", flush=True)
        dictionary = {}

    # Constrói embeddings textuais
    print("Construindo embeddings paramétricos...", flush=True)
    text_protos = build_text_embeddings(class_names, dictionary,
                                        embed_dim=CFG["visual_embed"])
    print(f"  text_protos: {text_protos.shape}", flush=True)

    run_cv(seqs, labels, class_names, text_protos)


if __name__ == "__main__":
    main()
