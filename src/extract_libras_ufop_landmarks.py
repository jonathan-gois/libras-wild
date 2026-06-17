"""
extract_libras_ufop_landmarks.py — Extrai landmarks MediaPipe Holistic do dataset
libras_ufop (frames RGB já decodificados) para alimentar o pipeline LiBERT.

Diferente de wild_pipeline.extract_landmarks (que recebe um caminho de vídeo e usa
cv2.VideoCapture), aqui a entrada já são arrays numpy de frames RGB decodificados
(sem vídeo para abrir). A lógica de extração dos 225 floats por frame a partir do
resultado do MediaPipe Holistic é a mesma; processamos frame a frame.

Cada amostra é uma sequência fixa de 91 frames RGB 224x224 (uint8). Não há um fps
real associado (são imagens, não um vídeo decodificado com timestamps), então
usamos um valor fixo placeholder FPS_PLACEHOLDER = 30.0 para todas as amostras,
salvo junto do array de landmarks para manter compatibilidade com o formato
(landmarks, fps) usado no resto do projeto (ver wild_pipeline.py).

Formato de saída por frame: 225 floats =
    pose       (33 landmarks x,y,z) -> índices   0..98
    mão esq.   (21 landmarks x,y,z) -> índices  99..161
    mão dir.   (21 landmarks x,y,z) -> índices 162..224
Frames sem detecção de uma parte ficam com zeros nessa parte.

Entrada:
    data/libras_ufop/LIBRAS-UFOP-Split/{train,val,test}_images.npy  (N, 91, 224, 224, 3) uint8
    data/libras_ufop/LIBRAS-UFOP-Split/{train,val,test}_labels.npy  (N,) int

NOTA sobre ordem de canais: apesar da documentação do dataset sugerir RGB, a inspeção
visual mostrou que os arrays armazenam os frames em ordem BGR (cores de pele/objetos
ficavam azuladas ao salvar como RGB direto; invertendo os canais a imagem fica correta).
Por isso este script inverte os canais (BGR->RGB) antes de passar ao MediaPipe Holistic,
que espera entrada RGB. Sem essa correção a taxa de detecção de mãos cai praticamente a
zero (a pose ainda é detectada razoavelmente porque é mais robusta a erro de cor, mas as
mãos não).

Saída:
    data/libras_ufop_landmarks/<split>/<idx>.pkl   -> pickle.dump((landmarks (91,225) float32, fps))
    data/libras_ufop_landmarks/index.csv           -> colunas: split,idx,label,pkl_path

Uso:
    env/bin/python src/extract_libras_ufop_landmarks.py
    env/bin/python src/extract_libras_ufop_landmarks.py --splits val test --limit 20
    env/bin/python src/extract_libras_ufop_landmarks.py --workers 8
"""

import argparse
import csv
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "libras_ufop" / "LIBRAS-UFOP-Split"
OUT_DIR = ROOT / "data" / "libras_ufop_landmarks"

FPS_PLACEHOLDER = 30.0  # não há fps real (entrada = imagens, não vídeo decodificado)
SPLITS = ["train", "val", "test"]
LOG_EVERY = 25  # loga progresso a cada N amostras processadas


def extract_landmarks_from_frames(frames: np.ndarray) -> np.ndarray:
    """Roda MediaPipe Holistic em cada frame RGB de `frames` (T, H, W, 3) uint8.

    Retorna array (T, 225) float32 com o mesmo esquema de wild_pipeline.extract_landmarks,
    mas usando static_image_mode=True (cada frame é tratado de forma independente,
    sem continuidade de vídeo real entre os frames vindos de um decoder).
    """
    import mediapipe as mp

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows = np.zeros((frames.shape[0], 225), dtype=np.float32)
    detected_any = 0
    try:
        for t in range(frames.shape[0]):
            rgb = frames[t][:, :, ::-1]  # arrays do dataset estão em BGR -> converte p/ RGB
            res = holistic.process(rgb)

            row = rows[t]
            has_detection = False
            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks.landmark[:33]):
                    row[i * 3:i * 3 + 3] = [lm.x, lm.y, lm.z]
                has_detection = True
            if res.left_hand_landmarks:
                for i, lm in enumerate(res.left_hand_landmarks.landmark):
                    base = 33 * 3 + i * 3
                    row[base:base + 3] = [lm.x, lm.y, lm.z]
                has_detection = True
            if res.right_hand_landmarks:
                for i, lm in enumerate(res.right_hand_landmarks.landmark):
                    base = 54 * 3 + i * 3
                    row[base:base + 3] = [lm.x, lm.y, lm.z]
                has_detection = True
            if has_detection:
                detected_any += 1
    finally:
        holistic.close()

    return rows, detected_any


# Cache de mmap por processo worker (evita reabrir o .npy a cada amostra; cada worker
# abre seu próprio mmap, então o array de frames inteiro nunca trafega pelo IPC do Pool).
_MMAP_CACHE: dict[str, np.ndarray] = {}


def _get_mmap(split: str) -> np.ndarray:
    if split not in _MMAP_CACHE:
        _MMAP_CACHE[split] = np.load(DATA_DIR / f"{split}_images.npy", mmap_mode="r")
    return _MMAP_CACHE[split]


def process_one(args) -> dict:
    """Processa uma amostra: lê os frames do mmap local ao worker, extrai landmarks e salva .pkl."""
    split, idx, label, pkl_path = args
    pkl_path = Path(pkl_path)
    if pkl_path.exists():
        return {"split": split, "idx": idx, "label": label, "pkl_path": str(pkl_path),
                "status": "skipped", "detected_frames": None}

    t0 = time.time()
    images = _get_mmap(split)
    frames = np.array(images[idx])  # materializa só esta amostra (91,224,224,3) ~13.6MB
    landmarks, detected_any = extract_landmarks_from_frames(frames)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump((landmarks, FPS_PLACEHOLDER), f)

    return {
        "split": split, "idx": idx, "label": label, "pkl_path": str(pkl_path),
        "status": "ok", "detected_frames": detected_any, "n_frames": frames.shape[0],
        "elapsed": time.time() - t0,
    }


def iter_jobs(split: str, limit: int | None):
    """Itera (split, idx, label, pkl_path) sem carregar nenhum array de frames — cada
    worker lê sua própria amostra do mmap em process_one, mantendo o uso de memória
    do processo pai (e do IPC do Pool) mínimo independente do tamanho do dataset."""
    labels = np.load(DATA_DIR / f"{split}_labels.npy")
    n = labels.shape[0] if limit is None else min(limit, labels.shape[0])
    split_out_dir = OUT_DIR / split
    for idx in range(n):
        pkl_path = split_out_dir / f"{idx}.pkl"
        yield (split, idx, int(labels[idx]), str(pkl_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--limit", type=int, default=None, help="Limita nº de amostras por split (debug)")
    ap.add_argument("--workers", type=int, default=8, help="Processos paralelos (default 8)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "index.csv"
    write_header = not index_path.exists()

    index_f = open(index_path, "a", newline="")
    index_writer = csv.writer(index_f)
    if write_header:
        index_writer.writerow(["split", "idx", "label", "pkl_path"])
        index_f.flush()

    total_ok = total_skipped = total_detected_frames = total_frames = 0
    t_start = time.time()

    for split in args.splits:
        labels = np.load(DATA_DIR / f"{split}_labels.npy")
        n_total = labels.shape[0] if args.limit is None else min(args.limit, labels.shape[0])
        print(f"[{split}] iniciando... {n_total} amostras a processar (workers={args.workers})", flush=True)

        # iter_jobs é um gerador: nenhum array de frames é carregado no processo pai.
        # Cada worker lê sua própria fatia do mmap em process_one (ver _get_mmap).
        n_done = 0
        with Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(process_one, iter_jobs(split, args.limit), chunksize=1):
                n_done += 1
                index_writer.writerow([result["split"], result["idx"], result["label"], result["pkl_path"]])
                if result["status"] == "ok":
                    total_ok += 1
                    total_detected_frames += result["detected_frames"]
                    total_frames += result["n_frames"]
                else:
                    total_skipped += 1

                if n_done % LOG_EVERY == 0 or n_done == n_total:
                    elapsed = time.time() - t_start
                    rate = total_detected_frames / total_frames * 100 if total_frames else 0.0
                    print(
                        f"[{split}] {n_done}/{n_total} "
                        f"(ok={total_ok}, skip={total_skipped}, "
                        f"taxa_deteccao={rate:.1f}%, elapsed={elapsed:.0f}s)",
                        flush=True,
                    )
                index_f.flush()

    index_f.close()
    elapsed = time.time() - t_start
    rate = total_detected_frames / total_frames * 100 if total_frames else 0.0
    print(
        f"\n=== Concluído em {elapsed:.0f}s ===\n"
        f"Amostras processadas (ok): {total_ok}\n"
        f"Amostras puladas (já existiam): {total_skipped}\n"
        f"Taxa de detecção (frames com >=1 parte detectada): {rate:.1f}% "
        f"({total_detected_frames}/{total_frames} frames)\n"
        f"Índice: {index_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
