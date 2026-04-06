"""
wild_pipeline.py — Gera base "Libras Wild" a partir de lista de vídeos YouTube.

Para cada vídeo:
  1. Download (yt-dlp, 360p)
  2. Extração de landmarks (MediaPipe Holistic)
  3. Segmentação por picos de velocidade (sign_segmenter)
  4. Filtro de sinais válidos (sign_spotter)
  5. Extração de clipes (ffmpeg)
  6. Registro em dataset/index.json

Uso:
    python3 src/wild_pipeline.py --urls videos.txt --out dataset/wild
    python3 src/wild_pipeline.py --url https://youtu.be/XYZ --out dataset/wild

Output:
    dataset/wild/
        index.json          ← metadados de todos os clipes
        <video_id>/
            <seg_id>.mp4    ← clipe cortado
"""

import argparse, json, pickle, subprocess, sys, os, re, time
import numpy as np
from pathlib import Path
from dataclasses import asdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import mediapipe as mp
import cv2
from sign_segmenter import segment_signs, SignSegment
from sign_spotter   import load_spotter, filter_segments

# ── Config padrão ─────────────────────────────────────────────────────────────
SPOTTER_THRESHOLD = 0.50   # P(sinal válido) mínimo
MIN_DUR  = 0.25            # s — alinhado com novo segmentador
MAX_DUR  = 1.8             # s — 1 sinal = max 1.8s
FPS_DEFAULT = 30.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def yt_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url.strip() if len(url.strip()) == 11 else None


def download_video(url: str, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = yt_id(url)
    out_path = out_dir / f"{vid}.mp4"
    # Força h264 para compatibilidade com OpenCV
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[vcodec^=avc][height<=480]+bestaudio/bestvideo[height<=480]+bestaudio/best[height<=480]",
        "--merge-output-format", "mp4",
        "--postprocessor-args", "ffmpeg:-vcodec libx264 -crf 23",
        "-o", str(out_path), "--no-playlist", "--quiet", url,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not out_path.exists():
        # Fallback: baixa qualquer formato e re-encoda
        print("  Fallback: baixando e re-encodando...", flush=True)
        tmp = out_dir / f"{vid}_raw.%(ext)s"
        r2 = subprocess.run([
            "yt-dlp", "-f", "best[height<=480]",
            "-o", str(tmp), "--no-playlist", "--quiet", url,
        ], capture_output=True)
        raw = next(out_dir.glob(f"{vid}_raw.*"), None)
        if raw:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(raw),
                "-vcodec", "libx264", "-crf", "23", "-acodec", "aac",
                "-loglevel", "error", str(out_path),
            ])
            raw.unlink(missing_ok=True)
    if out_path.exists():
        return out_path
    print(f"  [ERRO download] {result.stderr.decode()[:200]}", flush=True)
    return None


def extract_landmarks(video_path: Path) -> tuple[np.ndarray, float]:
    """Retorna (landmarks (T,225), fps)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        row = np.zeros(225, dtype=np.float32)
        # Pose (33 landmarks → índices 0-32, só x,y,z dos primeiros 33)
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark[:33]):
                row[i*3:i*3+3] = [lm.x, lm.y, lm.z]
        # Mão esquerda (landmarks 33-53)
        if res.left_hand_landmarks:
            for i, lm in enumerate(res.left_hand_landmarks.landmark):
                base = 33*3 + i*3
                row[base:base+3] = [lm.x, lm.y, lm.z]
        # Mão direita (landmarks 54-74)
        if res.right_hand_landmarks:
            for i, lm in enumerate(res.right_hand_landmarks.landmark):
                base = 54*3 + i*3
                row[base:base+3] = [lm.x, lm.y, lm.z]
        frames.append(row)

    cap.release()
    holistic.close()
    return np.array(frames, dtype=np.float32), fps


def check_signer_present(video_path: Path,
                          sample_every_s: float = 1.5,
                          min_hand_ratio: float = 0.20) -> tuple[bool, float]:
    """
    Pré-checagem rápida: há um sinalizador no vídeo?

    Amostra 1 frame a cada sample_every_s segundos e verifica se o MediaPipe
    detecta pelo menos uma mão. Retorna (ok, ratio_frames_com_mao).

    Critério: pelo menos min_hand_ratio dos frames amostrados com mão visível.
    Rejeita vídeos sem sinalizador (entrevistas sem LIBRAS, conteúdo off-topic).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(sample_every_s * fps))

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )

    sampled, with_hand = 0, 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            sampled += 1
            if res.multi_hand_landmarks:
                with_hand += 1
        frame_idx += 1

    cap.release()
    hands.close()

    ratio = with_hand / max(sampled, 1)
    return ratio >= min_hand_ratio, ratio


def cut_clip(video_path: Path, t_start: float, t_end: float,
             out_path: Path) -> bool:
    """Corta clipe com ffmpeg. Retorna True se ok."""
    dur = t_end - t_start
    cmd = [
        "ffmpeg", "-y", "-ss", f"{t_start:.3f}", "-i", str(video_path),
        "-t", f"{dur:.3f}", "-c:v", "libx264", "-crf", "23",
        "-c:a", "aac", "-loglevel", "error", str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0


# ── Pipeline por vídeo ────────────────────────────────────────────────────────

def process_video(url: str, out_root: Path, clf,
                  keep_video: bool = False) -> list[dict]:
    vid = yt_id(url)
    if not vid:
        print(f"[SKIP] URL inválida: {url}", flush=True)
        return []

    vid_dir  = out_root / vid
    vid_dir.mkdir(parents=True, exist_ok=True)
    meta_file = vid_dir / "meta.json"

    print(f"\n=== {vid} ===", flush=True)

    # 1. Download
    video_file = next(vid_dir.glob("*.mp4"), None)
    if video_file is None:
        print("  Baixando...", flush=True)
        video_file = download_video(url, vid_dir)
        if video_file is None:
            return []
    else:
        print(f"  Já baixado: {video_file.name}", flush=True)

    # 2. Pré-checagem: há sinalizador no vídeo?
    check_file = vid_dir / "signer_check.json"
    if check_file.exists():
        check_data = json.load(open(check_file))
        if not check_data["ok"]:
            print(f"  [SKIP] Sem sinalizador (ratio={check_data['ratio']:.2f})", flush=True)
            if not (vid_dir / "landmarks.pkl").exists():
                video_file.unlink(missing_ok=True)
            return []
    else:
        print("  Verificando presença de sinalizador...", flush=True)
        ok, ratio = check_signer_present(video_file)
        check_file.write_text(json.dumps({"ok": ok, "ratio": round(ratio, 3)}))
        print(f"  Sinalizador: {'✓' if ok else '✗'}  mãos em {ratio*100:.0f}% dos frames", flush=True)
        if not ok:
            video_file.unlink(missing_ok=True)
            return []

    # 3. Landmarks (cache)
    lm_path = vid_dir / "landmarks.pkl"
    if lm_path.exists():
        print("  Landmarks em cache.", flush=True)
        lm, fps = pickle.load(open(lm_path, "rb"))
    else:
        print("  Extraindo landmarks...", flush=True)
        lm, fps = extract_landmarks(video_file)
        if len(lm) == 0:
            print("  [SKIP] Nenhum frame extraído (formato incompatível?).", flush=True)
            return []
        pickle.dump((lm, fps), open(lm_path, "wb"))
        print(f"  {len(lm)} frames @ {fps:.1f}fps", flush=True)

    if len(lm) < 10:
        print("  [SKIP] Vídeo muito curto ou sem frames.", flush=True)
        return []

    # 3. Segmentação
    print("  Segmentando...", flush=True)
    segs = segment_signs(lm, fps, min_dur_s=MIN_DUR, max_dur_s=MAX_DUR)
    print(f"  {len(segs)} candidatos", flush=True)

    # 4. Filtro sign_spotter
    segs = filter_segments(clf, segs, lm, fps, threshold=SPOTTER_THRESHOLD)
    print(f"  {len(segs)} após filtro (threshold={SPOTTER_THRESHOLD})", flush=True)

    if not segs:
        return []

    # 5. Corta clipes
    clips_dir = vid_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    records = []

    for seg in segs:
        clip_name = f"{seg.seg_id}.mp4"
        clip_path = clips_dir / clip_name

        if not clip_path.exists():
            ok = cut_clip(video_file, seg.t_start, seg.t_end, clip_path)
        else:
            ok = True

        if not ok:
            continue

        rec = {
            "seg_id":        seg.seg_id,
            "video_id":      vid,
            "video_url":     f"https://youtu.be/{vid}",
            "t_start":       round(seg.t_start, 3),
            "t_end":         round(seg.t_end, 3),
            "duration":      round(seg.duration, 3),
            "spotter_score": round(getattr(seg, "spotter_score", 0.0), 3),
            "energy_peak":   round(seg.energy_peak, 3),
            "clip":          str(clip_path.relative_to(out_root)),
        }
        records.append(rec)

    meta_file.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"  {len(records)} clipes salvos em {clips_dir}", flush=True)

    if not keep_video:
        video_file.unlink(missing_ok=True)
        print("  Vídeo original removido.", flush=True)

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",  type=str, help="URL única do YouTube")
    ap.add_argument("--urls", type=str, help="Arquivo TXT com uma URL por linha")
    ap.add_argument("--out",  type=str, default="dataset/wild")
    ap.add_argument("--keep-video", action="store_true",
                    help="Não apaga o vídeo após processar")
    ap.add_argument("--threshold", type=float, default=SPOTTER_THRESHOLD)
    args = ap.parse_args()

    urls = []
    if args.url:
        urls.append(args.url)
    if args.urls:
        urls += [l.strip() for l in open(args.urls) if l.strip() and not l.startswith("#")]
    if not urls:
        ap.print_help(); sys.exit(1)

    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    index_path = out_root / "index.json"

    # Carrega índice existente
    index = json.load(open(index_path)) if index_path.exists() else []
    existing_ids = {r["seg_id"] for r in index}

    print(f"Carregando sign spotter...", flush=True)
    clf = load_spotter()

    for url in urls:
        records = process_video(url, out_root, clf,
                                keep_video=args.keep_video)
        for r in records:
            if r["seg_id"] not in existing_ids:
                index.append(r)
                existing_ids.add(r["seg_id"])

    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"\nTotal no índice: {len(index)} clipes → {index_path}")


if __name__ == "__main__":
    main()
