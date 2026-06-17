"""
extract_v_librasil_landmarks.py — Extrai landmarks MediaPipe Holistic do
dataset V-Librasil (UFPE) para alimentar o pipeline LiBERT (pré-treino/
calibração).

Reaproveita exatamente a lógica de `wild_pipeline.extract_landmarks`:
MediaPipe Holistic (model_complexity=1) → vetor de 225 floats por frame
    índices   0- 32 → pose       (33 landmarks × x,y,z)
    índices  33- 53 → mão esq.   (21 landmarks × x,y,z)
    índices  54- 74 → mão dir.   (21 landmarks × x,y,z)
Partes não detectadas no frame ficam zeradas.

Entrada:
    data/v_librasil/videos UFPE (V-LIBRASIL)/annotations.csv
        colunas usadas: video_id, video_name, class, user_id
    data/v_librasil/videos UFPE (V-LIBRASIL)/data/<video_name>.mp4
        (arquivos no disco são nomeados por `video_name`, não `video_id` —
        ver normalize_key() abaixo para o motivo de não dar simplesmente
        `data / row["video_name"]`)

Saída:
    data/v_librasil_landmarks/<video_id_sem_extensao>.pkl
        tupla (landmarks: np.ndarray (T, 225) float32, fps: float)
        — mesmo formato usado em dataset/wild/<id>/landmarks.pkl
    data/v_librasil_landmarks/index.csv
        video_id, class, user_id, pkl_path, status (ok/falha), n_frames, motivo

Resumível: antes de processar um vídeo, verifica se o .pkl de saída já
existe (e está no índice como "ok") — se sim, pula. Permite interromper
(Ctrl-C / kill) e retomar rodando o script de novo.

Uso:
    python3 src/extract_v_librasil_landmarks.py
    python3 src/extract_v_librasil_landmarks.py --limit 30          # amostra
    python3 src/extract_v_librasil_landmarks.py --log-every 20

    # processamento completo em background:
    nohup env/bin/python src/extract_v_librasil_landmarks.py \
        > /tmp/v_librasil_extract.log 2>&1 &
"""

import argparse
import csv
import pickle
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wild_pipeline import extract_landmarks  # noqa: E402  (reuso exato da lógica)

ANNOTATIONS_CSV = ROOT / "data" / "v_librasil" / "videos UFPE (V-LIBRASIL)" / "annotations.csv"
VIDEOS_DIR = ROOT / "data" / "v_librasil" / "videos UFPE (V-LIBRASIL)" / "data"
OUT_DIR = ROOT / "data" / "v_librasil_landmarks"
INDEX_CSV = OUT_DIR / "index.csv"

INDEX_FIELDS = ["video_id", "class", "user_id", "pkl_path", "status", "n_frames", "motivo"]


def normalize_key(name: str) -> str:
    """Normaliza nomes de arquivo para casar `video_name` (CSV) com o nome
    real em disco. O CSV original tem aspas/barra/interrogação em alguns
    nomes (ex.: 'Cérebro (de "ervilha")...mp4', 'Frente\\Á Frente...mp4',
    'O quê?...mp4') que foram removidos/corrompidos na sanitização dos
    arquivos em disco (alguns viraram caracteres de área de uso privado
    Unicode, ex. U+F022 no lugar de '"'). Removendo esses caracteres de
    ambos os lados, os nomes batem 1:1 (verificado: 4086/4086)."""
    name = re.sub(r'[\\?"]', "", name)
    name = "".join(ch for ch in name if not (0xE000 <= ord(ch) <= 0xF8FF))
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_annotations() -> list[dict]:
    with open(ANNOTATIONS_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def build_file_map() -> dict[str, Path]:
    """normalized_name -> caminho real do arquivo em disco."""
    return {normalize_key(p.name): p for p in VIDEOS_DIR.iterdir() if p.is_file()}


def load_existing_index() -> dict[str, dict]:
    """video_id -> row, para retomar sem reprocessar o que já está ok."""
    if not INDEX_CSV.exists():
        return {}
    with open(INDEX_CSV, encoding="utf-8", newline="") as f:
        return {row["video_id"]: row for row in csv.DictReader(f)}


def write_index(rows: list[dict], index_rows: dict[str, dict]):
    """Grava o índice consolidado (ordem do CSV original). Chamada também em
    checkpoints intermediários, para que uma interrupção no meio do
    processamento completo (4086 vídeos, várias horas) ainda deixe um
    index.csv utilizável refletindo o progresso até aquele ponto."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            vid = row["video_id"]
            if vid in index_rows:
                writer.writerow(index_rows[vid])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="processa só os N primeiros vídeos do CSV (para teste)")
    ap.add_argument("--log-every", type=int, default=20, help="loga progresso a cada N vídeos processados")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_annotations()
    if args.limit:
        rows = rows[: args.limit]
    total = len(rows)

    file_map = build_file_map()
    existing_index = load_existing_index()

    # Reescreve o índice do zero, mas reaproveita entradas "ok" já existentes
    # (assim o índice final fica completo/consistente mesmo após retomadas).
    index_rows: dict[str, dict] = dict(existing_index)

    t0 = time.time()
    n_done = 0       # já estava ok antes desta execução (pulado)
    n_ok = 0         # processado com sucesso nesta execução
    n_fail = 0       # falhou nesta execução

    for i, row in enumerate(rows, 1):
        video_id = row["video_id"]
        video_id_stem = Path(video_id).stem
        out_pkl = OUT_DIR / f"{video_id_stem}.pkl"

        prev = existing_index.get(video_id)
        if out_pkl.exists() and prev is not None and prev.get("status") == "ok":
            n_done += 1
        else:
            key = normalize_key(row["video_name"])
            video_path = file_map.get(key)

            status, n_frames, motivo = "falha", 0, ""
            if video_path is None:
                motivo = f"arquivo não encontrado em disco (video_name={row['video_name']!r})"
                print(f"[{i}/{total}] FALHA {video_id}: {motivo}", flush=True)
            else:
                try:
                    landmarks, fps = extract_landmarks(video_path)
                    if landmarks.size == 0 or landmarks.shape[0] == 0:
                        motivo = "0 frames lidos do vídeo"
                        print(f"[{i}/{total}] FALHA {video_id}: {motivo}", flush=True)
                    else:
                        with open(out_pkl, "wb") as f:
                            pickle.dump((landmarks, fps), f)
                        status, n_frames = "ok", landmarks.shape[0]
                except Exception as e:  # noqa: BLE001 — não pode derrubar o lote
                    motivo = f"{type(e).__name__}: {e}"
                    print(f"[{i}/{total}] FALHA {video_id}: {motivo}", flush=True)

            if status == "ok":
                n_ok += 1
            else:
                n_fail += 1
                out_pkl.unlink(missing_ok=True)

            index_rows[video_id] = {
                "video_id": video_id,
                "class": row["class"],
                "user_id": row["user_id"],
                "pkl_path": str(out_pkl.relative_to(ROOT)) if status == "ok" else "",
                "status": status,
                "n_frames": n_frames,
                "motivo": motivo,
            }

        if i % args.log_every == 0 or i == total:
            elapsed = time.time() - t0
            write_index(rows, index_rows)
            print(
                f"[progresso] {i}/{total} vídeos varridos "
                f"(pulados={n_done}, ok={n_ok}, falha={n_fail}) "
                f"— {elapsed/60:.1f} min decorridos",
                flush=True,
            )

    # Grava o índice consolidado final (ordem do CSV original)
    write_index(rows, index_rows)

    elapsed = time.time() - t0
    print(
        f"\n[fim] total={total} pulados(já ok)={n_done} ok_nesta_execucao={n_ok} "
        f"falha_nesta_execucao={n_fail} — {elapsed/60:.1f} min",
        flush=True,
    )
    print(f"Índice salvo em: {INDEX_CSV}", flush=True)


if __name__ == "__main__":
    main()
