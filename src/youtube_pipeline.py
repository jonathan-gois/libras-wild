"""
Pipeline: YouTube URL → segmentação de sinais → classificação → alinhamento com legendas.

Etapas:
  1. Download do vídeo + legendas (yt-dlp)
  2. Extração de landmarks frame a frame (MediaPipe)
  3. Sign spotting: detecta onde cada sinal começa e termina
  4. Classificação de cada segmento (ExtraTrees-1000 ou modelo neural)
  5. Alinhamento temporal com legendas (SRT/VTT)
  6. Salva JSON com resultados + vídeo anotado opcional

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate

    # Classificar sinais de um vídeo com legendas
    python3 -u src/youtube_pipeline.py --url "https://youtu.be/..." --output results/youtube/

    # Modo de construção de dataset (coleta amostras "in the wild")
    python3 -u src/youtube_pipeline.py --url "..." --harvest-dataset

Saídas:
    results/youtube/<video_id>/
        segments.json          — todos os segmentos detectados + classificação
        timeline.json          — alinhamento com legendas
        frames/<seg_id>/       — frames dos segmentos (opcional)
        dataset_harvest.pkl    — amostras para expandir o dataset (modo harvest)
"""

import argparse, json, re, sys, os, subprocess, pickle, tempfile, time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ── Estruturas de dados ───────────────────────────────────────────────────────
@dataclass
class SignSegment:
    seg_id:      str
    t_start:     float   # segundos
    t_end:       float
    duration:    float
    frame_start: int
    frame_end:   int
    top5:        list    # [(class_name, confidence), ...]
    velocity_peak: float # energia do movimento (qualidade do segmento)

@dataclass
class SubtitleEntry:
    t_start: float
    t_end:   float
    text:    str

@dataclass
class AlignedResult:
    subtitle:  SubtitleEntry
    segments:  list         # [SignSegment] sobrepostos com a legenda
    best_match: str | None  # classe mais votada no intervalo


# ── 1. Download ───────────────────────────────────────────────────────────────
def download_video(url: str, out_dir: Path) -> tuple[Path, Path | None]:
    """
    Baixa vídeo e legendas com yt-dlp.
    Retorna (video_path, subtitle_path).
    Subtitles: preferência por pt/pt-BR automático ou manual.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "%(id)s"

    print(f"Baixando vídeo: {url}", flush=True)
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--write-auto-subs", "--write-subs",
        "--sub-langs", "pt,pt-BR,pt-br,en",
        "--sub-format", "vtt/srt",
        "--convert-subs", "srt",
        "--output", str(base),
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(out_dir))
    if result.returncode != 0:
        print(f"  WARN yt-dlp: {result.stderr[:200]}", flush=True)

    # Encontra arquivos baixados
    videos = sorted(out_dir.glob("*.mp4"))
    subs   = sorted(out_dir.glob("*.srt")) + sorted(out_dir.glob("*.vtt"))

    video_path = videos[0] if videos else None
    sub_path   = subs[0]   if subs   else None

    if video_path:
        print(f"  Vídeo: {video_path.name}", flush=True)
    if sub_path:
        print(f"  Legenda: {sub_path.name}", flush=True)
    else:
        print("  Sem legenda disponível.", flush=True)

    return video_path, sub_path


# ── 2. Extração de landmarks ──────────────────────────────────────────────────
def extract_landmarks_from_video(video_path: Path) -> tuple[np.ndarray, float]:
    """
    Roda MediaPipe Holistic em todos os frames.
    Retorna (landmarks, fps):
        landmarks: (T, 225) float32
        fps: float
    """
    import cv2
    import mediapipe as mp

    mp_hol = mp.solutions.holistic
    cap    = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    print(f"  Extraindo landmarks (fps={fps:.1f})...", flush=True)
    with mp_hol.Holistic(static_image_mode=False, model_complexity=1,
                          min_detection_confidence=0.4,
                          min_tracking_confidence=0.4) as hol:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            row = []
            for part, n in [(res.pose_landmarks, 33),
                            (res.left_hand_landmarks, 21),
                            (res.right_hand_landmarks, 21)]:
                if part:
                    for lm in part.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                else:
                    row.extend([0.0] * n * 3)
            frames.append(row)
            i += 1
            if i % 300 == 0:
                print(f"    {i} frames ({i/fps:.1f}s)...", flush=True)

    cap.release()
    lm = np.array(frames, dtype=np.float32)
    print(f"  {len(lm)} frames extraídos ({len(lm)/fps:.1f}s)", flush=True)
    return lm, fps


# ── 3. Sign Spotter ───────────────────────────────────────────────────────────
# Índices MediaPipe relevantes para detectar movimento de sinais
RWRIST = 16   # pulso direito (pose landmark)
LWRIST = 15   # pulso esquerdo
RNOSE  = 0    # nariz (referência de posição da cabeça)

def compute_signing_energy(landmarks: np.ndarray, fps: float,
                           smooth_win: float = 0.3) -> np.ndarray:
    """
    Calcula a "energia de sinais" por frame:
    velocidade dos punhos + mãos (norma L2 da 1ª diferença).
    Suavizado com janela gaussiana.
    """
    T = len(landmarks)
    xyz = landmarks.reshape(T, 75, 3)

    # Velocidade dos pulsos (pose 15=esq, 16=dir) + raiz das mãos (33=esq, 54=dir)
    key_nodes = [15, 16, 33, 54]
    pos = xyz[:, key_nodes, :]            # (T, 4, 3)
    vel = np.diff(pos, axis=0)            # (T-1, 4, 3)
    energy = np.linalg.norm(vel, axis=-1).mean(axis=-1)  # (T-1,)
    energy = np.concatenate([[0.0], energy])              # (T,)

    # Suavização gaussiana
    win = int(smooth_win * fps)
    if win > 1:
        from scipy.ndimage import gaussian_filter1d
        energy = gaussian_filter1d(energy, sigma=win / 4)

    return energy


def detect_sign_segments(landmarks: np.ndarray, fps: float,
                          min_dur: float = 0.3,
                          max_dur: float = 4.0,
                          min_gap: float = 0.2,
                          energy_threshold_pct: float = 30.0,
                          context_pad: float = 0.1) -> list[SignSegment]:
    """
    Detecta segmentos de sinais baseado em picos de energia dos pulsos.

    Estratégia:
      1. Computa energia de movimento frame a frame
      2. Threshold adaptativo (percentil da energia não-zero)
      3. Agrupa frames acima do threshold em segmentos contíguos
      4. Filtra por duração mínima/máxima
      5. Adiciona padding de contexto

    Retorna lista de SignSegment (sem classificação — só timing).
    """
    energy = compute_signing_energy(landmarks, fps)
    T = len(landmarks)

    # Threshold adaptativo: percentil dos frames com movimento relevante
    active = energy[energy > 0.001]
    if len(active) == 0:
        return []
    threshold = np.percentile(active, energy_threshold_pct)

    # Máscara binária: acima do threshold = possível sinal
    mask = energy > threshold

    # Agrupa frames contíguos → segmentos brutos
    segments_raw = []
    in_seg = False
    seg_start = 0
    for i in range(T):
        if mask[i] and not in_seg:
            seg_start = i
            in_seg    = True
        elif not mask[i] and in_seg:
            segments_raw.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        segments_raw.append((seg_start, T - 1))

    # Une segmentos próximos (gap < min_gap)
    min_gap_f = int(min_gap * fps)
    merged = []
    for s, e in segments_raw:
        if merged and s - merged[-1][1] < min_gap_f:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Filtra por duração e adiciona padding
    pad_f    = int(context_pad * fps)
    min_f    = int(min_dur * fps)
    max_f    = int(max_dur * fps)
    result   = []

    for idx, (s, e) in enumerate(merged):
        dur_f = e - s
        if dur_f < min_f:
            continue
        if dur_f > max_f:
            # Divide segmentos longos em janelas deslizantes
            step = int(max_f * 0.5)
            for sub_s in range(s, e - min_f, step):
                sub_e = min(sub_s + max_f, e)
                ps = max(0, sub_s - pad_f)
                pe = min(T - 1, sub_e + pad_f)
                peak_e = float(energy[ps:pe].max())
                seg = SignSegment(
                    seg_id=f"seg_{idx:04d}_{sub_s}",
                    t_start=ps / fps, t_end=pe / fps,
                    duration=(pe - ps) / fps,
                    frame_start=ps, frame_end=pe,
                    top5=[], velocity_peak=peak_e,
                )
                result.append(seg)
            continue

        ps = max(0, s - pad_f)
        pe = min(T - 1, e + pad_f)
        peak_e = float(energy[ps:pe].max())
        seg = SignSegment(
            seg_id=f"seg_{idx:04d}",
            t_start=ps / fps, t_end=pe / fps,
            duration=(pe - ps) / fps,
            frame_start=ps, frame_end=pe,
            top5=[], velocity_peak=peak_e,
        )
        result.append(seg)

    print(f"  {len(result)} segmentos detectados", flush=True)
    return result


# ── 4. Classificação ──────────────────────────────────────────────────────────
def load_classifier(results_dir: Path = Path("results")):
    """
    Carrega ExtraTrees-1000 já treinado (se disponível).
    Fallback: re-treina rapidamente com dados disponíveis.
    """
    model_path = results_dir / "extratrees_model.pkl"
    if model_path.exists():
        print("  Carregando ET-1000 salvo...", flush=True)
        with open(model_path, "rb") as f:
            return pickle.load(f)

    # Re-treina com dados disponíveis
    print("  Treinando ET-1000 para classificação...", flush=True)
    import pickle as pk
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.preprocessing import LabelEncoder

    pkl = Path("data/processed_data_full.pkl")
    if not pkl.exists():
        pkl = Path("data/processed_data.pkl")

    with open(pkl, "rb") as f:
        df = pk.load(f)

    from data_loader import normalize_sequences
    from features.gei import compute_gei
    from features.kinematics import kinematic_stats
    from features.fft_features import sliding_fft

    seqs = normalize_sequences([np.array(r, dtype=np.float32) for r in df["landmarks"]])
    le   = LabelEncoder()
    y    = le.fit_transform(df["class"])

    X = np.concatenate([
        np.stack([compute_gei(s) for s in seqs]),
        np.stack([kinematic_stats(s) for s in seqs]),
        np.stack([sliding_fft(s) for s in seqs]),
    ], axis=1)

    clf = ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, random_state=42)
    clf.fit(X, y)

    model = {"clf": clf, "le": le}
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ET-1000 treinado e salvo → {model_path}", flush=True)
    return model


def classify_segment(seg: SignSegment, landmarks: np.ndarray,
                     model: dict, top_k: int = 5) -> SignSegment:
    """Classifica um segmento e preenche seg.top5."""
    from data_loader import normalize_sequences
    from features.gei import compute_gei
    from features.kinematics import kinematic_stats
    from features.fft_features import sliding_fft

    chunk = landmarks[seg.frame_start:seg.frame_end + 1]
    if len(chunk) < 10:
        seg.top5 = []
        return seg

    seq = normalize_sequences([chunk])[0]

    x = np.concatenate([
        compute_gei(seq).reshape(1, -1),
        kinematic_stats(seq).reshape(1, -1),
        sliding_fft(seq).reshape(1, -1),
    ], axis=1)

    proba = model["clf"].predict_proba(x)[0]
    top_idx = np.argsort(proba)[::-1][:top_k]
    seg.top5 = [(model["le"].classes_[i], float(proba[i])) for i in top_idx]
    return seg


# ── 5. Parser de legendas SRT ─────────────────────────────────────────────────
def parse_srt(srt_path: Path) -> list[SubtitleEntry]:
    """Parseia arquivo SRT em lista de SubtitleEntry."""
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\n+", text.strip())
    entries = []

    def ts_to_sec(ts: str) -> float:
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        # Linha de timestamp: "00:00:10,500 --> 00:00:12,000"
        time_line = next((l for l in lines if "-->" in l), None)
        if not time_line:
            continue
        m = re.match(r"([\d:,\.]+)\s+-->\s+([\d:,\.]+)", time_line)
        if not m:
            continue
        t_start = ts_to_sec(m.group(1))
        t_end   = ts_to_sec(m.group(2))
        text_lines = [l for l in lines if "-->" not in l and not l.strip().isdigit()]
        clean = re.sub(r"<[^>]+>", "", " ".join(text_lines)).strip()
        if clean:
            entries.append(SubtitleEntry(t_start=t_start, t_end=t_end, text=clean))

    return entries


# ── 6. Alinhamento temporal ───────────────────────────────────────────────────
def align_with_subtitles(segments: list[SignSegment],
                          subtitles: list[SubtitleEntry],
                          overlap_threshold: float = 0.3) -> list[AlignedResult]:
    """
    Para cada entrada de legenda, encontra quais segmentos se sobrepõem
    temporalmente e vota na classe mais provável para aquela janela.

    overlap_threshold: fração mínima de sobreposição para considerar alinhado
    """
    results = []
    for sub in subtitles:
        overlapping = []
        for seg in segments:
            # Calcula sobreposição
            overlap_start = max(seg.t_start, sub.t_start)
            overlap_end   = min(seg.t_end,   sub.t_end)
            overlap_dur   = max(0.0, overlap_end - overlap_start)
            seg_dur       = seg.t_end - seg.t_start
            if seg_dur > 0 and overlap_dur / seg_dur >= overlap_threshold:
                overlapping.append(seg)

        # Vota: classe mais recorrente nos top-1 dos segmentos sobrepostos
        votes = {}
        for seg in overlapping:
            if seg.top5:
                winner, conf = seg.top5[0]
                votes[winner] = votes.get(winner, 0) + conf
        best = max(votes, key=votes.get) if votes else None

        results.append(AlignedResult(
            subtitle=sub, segments=overlapping, best_match=best
        ))

    return results


# ── 7. Modo harvest: coleta dataset "in the wild" ────────────────────────────
def harvest_dataset(segments: list[SignSegment], landmarks: np.ndarray,
                    aligned: list[AlignedResult], out_pkl: Path,
                    min_confidence: float = 0.6,
                    require_subtitle_match: bool = True):
    """
    Coleta amostras de alta confiança para expandir o dataset.

    Critérios de qualidade:
      - Confiança do top-1 ≥ min_confidence
      - (opcional) Top-1 bate com palavra da legenda
      - Duração razoável (0.3s a 3s)
    """
    # Palavras das legendas para validação cruzada
    subtitle_words = set()
    for ar in aligned:
        for w in ar.subtitle.text.lower().split():
            subtitle_words.add(w)

    harvested = []
    for seg in segments:
        if not seg.top5:
            continue
        label, conf = seg.top5[0]
        if conf < min_confidence:
            continue
        if require_subtitle_match and label not in subtitle_words:
            continue

        chunk = landmarks[seg.frame_start:seg.frame_end + 1]
        if len(chunk) < 10:
            continue

        harvested.append({
            "filename": f"wild_{seg.seg_id}",
            "class":    label,
            "landmarks": chunk.tolist(),
            "meta": {
                "t_start":    seg.t_start,
                "t_end":      seg.t_end,
                "confidence": conf,
                "top5":       seg.top5,
                "source":     "youtube_wild",
            }
        })

    print(f"  {len(harvested)} amostras coletadas para dataset", flush=True)

    if harvested and out_pkl:
        import pandas as pd
        df_new = pd.DataFrame(harvested)
        if out_pkl.exists():
            import pickle
            with open(out_pkl, "rb") as f:
                df_old = pickle.load(f)
            df_out = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_out = df_new
        with open(out_pkl, "wb") as f:
            import pickle
            pickle.dump(df_out, f)
        print(f"  Dataset salvo: {len(df_out)} amostras → {out_pkl}", flush=True)

    return harvested


# ── Pipeline principal ────────────────────────────────────────────────────────
def run_pipeline(url: str, out_base: Path = Path("results/youtube"),
                 harvest: bool = False,
                 min_confidence: float = 0.55):

    # Extrai video_id do URL
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    video_id = m.group(1) if m else "video"
    out_dir  = out_base / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Pipeline: {url}", flush=True)
    print(f"Output:   {out_dir}", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. Download
    video_path, sub_path = download_video(url, out_dir)
    if not video_path:
        print("ERRO: vídeo não baixado.", flush=True)
        return

    # 2. Landmarks
    t0 = time.time()
    landmarks, fps = extract_landmarks_from_video(video_path)
    print(f"  MediaPipe: {time.time()-t0:.1f}s", flush=True)

    # 3. Sign spotting
    segments = detect_sign_segments(landmarks, fps)

    # 4. Classificação
    model = load_classifier()
    print(f"  Classificando {len(segments)} segmentos...", flush=True)
    for i, seg in enumerate(segments):
        classify_segment(seg, landmarks, model)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(segments)} classificados", flush=True)

    # 5. Legendas
    subtitles = parse_srt(sub_path) if sub_path else []
    print(f"  {len(subtitles)} entradas de legenda", flush=True)

    # 6. Alinhamento
    aligned = align_with_subtitles(segments, subtitles) if subtitles else []

    # 7. Salva resultados
    segments_out = [asdict(s) for s in segments]
    with open(out_dir / "segments.json", "w") as f:
        json.dump(segments_out, f, ensure_ascii=False, indent=2)

    if aligned:
        timeline_out = []
        for ar in aligned:
            timeline_out.append({
                "subtitle":    asdict(ar.subtitle),
                "n_segments":  len(ar.segments),
                "best_match":  ar.best_match,
                "segments":    [s.seg_id for s in ar.segments],
            })
        with open(out_dir / "timeline.json", "w") as f:
            json.dump(timeline_out, f, ensure_ascii=False, indent=2)

    # 8. Harvest
    if harvest:
        harvest_dataset(
            segments, landmarks, aligned,
            out_pkl=Path("data/processed_data_wild.pkl"),
            min_confidence=min_confidence,
        )

    # Sumário
    print(f"\n{'─'*50}", flush=True)
    print(f"Segmentos detectados: {len(segments)}", flush=True)
    print(f"Legendas alinhadas:   {len(aligned)}", flush=True)
    if aligned:
        matched = sum(1 for ar in aligned if ar.best_match)
        print(f"Com sinal identificado: {matched}/{len(aligned)}", flush=True)

    # Preview dos primeiros 10
    print(f"\nPrimeiros segmentos:", flush=True)
    for seg in segments[:10]:
        if seg.top5:
            top = ", ".join(f"{c}({p:.0%})" for c, p in seg.top5[:3])
            print(f"  [{seg.t_start:.1f}s–{seg.t_end:.1f}s] {top}", flush=True)

    if aligned:
        print(f"\nAlinhamento com legenda:", flush=True)
        for ar in aligned[:10]:
            mark = f"→ {ar.best_match}" if ar.best_match else "→ ?"
            print(f"  [{ar.subtitle.t_start:.1f}s] \"{ar.subtitle.text[:40]}\" {mark}", flush=True)

    print(f"\nResultados em: {out_dir}", flush=True)
    return out_dir


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline YouTube → Libras")
    parser.add_argument("--url",      required=True, help="URL do YouTube")
    parser.add_argument("--output",   default="results/youtube", help="Pasta de saída")
    parser.add_argument("--harvest-dataset", action="store_true",
                        help="Coleta amostras para expandir o dataset")
    parser.add_argument("--min-confidence", type=float, default=0.55,
                        help="Confiança mínima para harvest (padrão: 0.55)")
    args = parser.parse_args()

    run_pipeline(
        url=args.url,
        out_base=Path(args.output),
        harvest=args.harvest_dataset,
        min_confidence=args.min_confidence,
    )
