"""Pré-anotação de glosas no dataset wild via matching de similaridade de embedding.

O que faz:
    Para cada vídeo de `dataset/wild/<video_id>/` que tem `landmarks.pkl`, carrega o
    landmark contínuo (T, 225) + fps, roda `preprocessing.preprocess()` (normalize_v3 +
    kalman_v3) no VÍDEO INTEIRO (evita o bias de inicialização do Kalman em clipes curtos
    isolados), e só então recorta a fatia correspondente a cada clipe listado em
    `meta.json` usando `t_start`/`t_end` convertidos para índices de frame via fps.
    Cada fatia recortada é reamostrada para WINDOW=200 frames (`minds_data.resample`,
    interpolação linear) e embedada em lote na GPU com `model.embed()` (encoder
    LiBERT calibrado, `checkpoints/calibrated.pt`). O embedding [CLS] de cada clipe é
    comparado via similaridade de cosseno contra os 20 protótipos de classe do
    MINDS-Libras (`checkpoints/prototypes.pkl`); se a maior similaridade for >= ao
    `suggested_threshold` salvo no protótipo, a classe correspondente é proposta.

Como rodar:
    /home/jonathangois/Documentos/Libras_2026/env/bin/python LiBERT/preannotate_wild.py

Saída:
    LiBERT/results/wild_preannotation.csv com colunas:
        video_id, seg_id, clip_path, t_start, t_end, predicted_class, similarity, status
    - predicted_class: nome da classe MINDS-Libras mais similar (sempre preenchido,
      mesmo quando "desconhecido" — é só o argmax, mas a confiança é baixa).
    - similarity: similaridade de cosseno do embedding do clipe contra o protótipo
      de predicted_class (a maior entre as 20 classes).
    - status: "confiante" se similarity >= suggested_threshold (prototypes.pkl),
      caso contrário "desconhecido" — significa que o clipe não bateu com confiança
      suficiente em nenhuma das 20 classes conhecidas e precisa de revisão manual
      (pode ser uma glosa fora do vocabulário do MINDS-Libras, ruído de segmentação,
      etc.).

Não escreve em nenhum banco de dados, Supabase ou API externa — só o CSV local acima.
"""

import json
import pickle

import numpy as np
import pandas as pd
import torch

from config import PROJECT_ROOT, WILD_DIR, CKPT_DIR, LIBERT_DIR, WINDOW
from preprocessing import preprocess
from minds_data import resample
from model import LiBERT

RESULTS_DIR = LIBERT_DIR / "results"
BATCH_SIZE = 256


def load_model(device):
    model = LiBERT().to(device)
    state_dict = torch.load(CKPT_DIR / "calibrated.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_prototypes():
    with open(CKPT_DIR / "prototypes.pkl", "rb") as f:
        data = pickle.load(f)
    classes = data["classes"]
    proto_mat = np.stack([data["prototypes"][c] for c in classes]).astype(np.float32)
    # Normaliza para cosine via produto escalar simples depois.
    proto_norms = np.linalg.norm(proto_mat, axis=1, keepdims=True) + 1e-8
    proto_unit = proto_mat / proto_norms
    threshold = float(data["suggested_threshold"])
    return classes, proto_unit, threshold


def cosine_sim_batch(embeddings: np.ndarray, proto_unit: np.ndarray) -> np.ndarray:
    """embeddings: (N, D). proto_unit: (C, D) já normalizado. Retorna (N, C)."""
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    emb_unit = embeddings / emb_norms
    return emb_unit @ proto_unit.T


def iter_video_dirs():
    for d in sorted(WILD_DIR.iterdir()):
        if d.is_dir() and (d / "landmarks.pkl").exists():
            yield d


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} (cuda available: {torch.cuda.is_available()})")

    model = load_model(device)
    classes, proto_unit, threshold = load_prototypes()
    print(f"protótipos: {len(classes)} classes, threshold sugerido = {threshold:.4f}")

    video_dirs = list(iter_video_dirs())
    print(f"vídeos com landmarks.pkl: {len(video_dirs)}")

    rows = []
    n_clips_total = 0
    n_videos_done = 0

    for vdir in video_dirs:
        video_id = vdir.name
        meta_path = vdir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta:
            continue

        with open(vdir / "landmarks.pkl", "rb") as f:
            seq, fps = pickle.load(f)
        seq = np.asarray(seq, dtype=np.float32)
        fps = float(fps)

        # Pré-processa o vídeo inteiro ANTES de recortar (evita bias de init do Kalman).
        seq_pp = preprocess(seq)
        T_total = seq_pp.shape[0]

        clip_windows = []
        clip_meta = []
        for clip in meta:
            t_start, t_end = clip["t_start"], clip["t_end"]
            f_start = int(t_start * fps)
            f_end = int(t_end * fps)
            f_start = max(0, min(f_start, T_total - 1))
            f_end = max(f_start + 1, min(f_end, T_total))

            clip_seq = seq_pp[f_start:f_end]
            if clip_seq.shape[0] < 1:
                continue
            clip_resampled = resample(clip_seq, WINDOW)
            clip_windows.append(clip_resampled)
            clip_meta.append(clip)

        if not clip_windows:
            continue

        # Batch pela GPU.
        all_embeddings = []
        X = np.stack(clip_windows).astype(np.float32)  # (N, 200, 225)
        with torch.no_grad():
            for i in range(0, len(X), BATCH_SIZE):
                xb = torch.from_numpy(X[i:i + BATCH_SIZE]).to(device)
                emb = model.embed(xb).cpu().numpy()
                all_embeddings.append(emb)
        embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 384)

        sims = cosine_sim_batch(embeddings, proto_unit)  # (N, C)
        best_idx = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(sims)), best_idx]

        for clip, ci, sim in zip(clip_meta, best_idx, best_sim):
            status = "confiante" if sim >= threshold else "desconhecido"
            rows.append({
                "video_id": video_id,
                "seg_id": clip["seg_id"],
                "clip_path": clip["clip"],
                "t_start": clip["t_start"],
                "t_end": clip["t_end"],
                "predicted_class": classes[ci],
                "similarity": float(sim),
                "status": status,
            })

        n_clips_total += len(clip_meta)
        n_videos_done += 1
        if n_videos_done % 20 == 0:
            print(f"  ... {n_videos_done}/{len(video_dirs)} vídeos, {n_clips_total} clipes processados")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "wild_preannotation.csv"
    df = pd.DataFrame(rows, columns=[
        "video_id", "seg_id", "clip_path", "t_start", "t_end",
        "predicted_class", "similarity", "status",
    ])
    df.to_csv(out_path, index=False)

    n_conf = (df["status"] == "confiante").sum()
    n_unk = (df["status"] == "desconhecido").sum()
    print(f"\nCSV salvo em {out_path}")
    print(f"vídeos processados: {n_videos_done}")
    print(f"total de clipes: {len(df)}")
    print(f"confiante: {n_conf} ({100 * n_conf / len(df):.1f}%)")
    print(f"desconhecido: {n_unk} ({100 * n_unk / len(df):.1f}%)")


if __name__ == "__main__":
    main()
