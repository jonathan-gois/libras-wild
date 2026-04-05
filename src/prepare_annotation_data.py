"""
Prepara dados para a ferramenta de anotação web.

Gera:
  docs/data/segments.json   — metadados de cada segmento (timestamps, previsão, landmarks)
  docs/data/config.json     — config geral (classes, vídeo YouTube)

Filtra segmentos com duração entre 0.5s e 5s.
Inclui landmarks normalizados (amostrados a 16 frames) para exibição no canvas.
"""
import json, pickle, sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent.parent)

SEGS_FILE   = Path("results/youtube/-ZDkdbPqUZg/segments.json")
LM_PKL      = Path("results/youtube/-ZDkdbPqUZg/landmarks.pkl")
OUT_SEGS    = Path("docs/data/segments.json")
OUT_CFG     = Path("docs/data/config.json")
YOUTUBE_ID  = "-ZDkdbPqUZg"
MIN_DUR     = 0.5
MAX_DUR     = 5.0
N_FRAMES_UI = 8    # frames amostrados por segmento para exibição

CLASSES = [
    "acontecer","aluno","amarelo","america","aproveitar",
    "bala","banco","banheiro","barulho","cinco",
    "conhecer","espelho","esquina","filho","maca",
    "medo","ruim","sapo","vacina","vontade",
]

# Conexões MediaPipe para visualização (índices de landmark)
CONNECTIONS = [
    # Pose (torso + braços)
    [11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],
    # Mão esquerda (landmarks 33-53)
    [33,34],[34,35],[35,36],[36,37],
    [33,38],[38,39],[39,40],[40,41],
    [33,42],[42,43],[43,44],[44,45],
    [33,46],[46,47],[47,48],[48,49],
    [33,50],[50,51],[51,52],[52,53],
    # Mão direita (landmarks 54-74)
    [54,55],[55,56],[56,57],[57,58],
    [54,59],[59,60],[60,61],[61,62],
    [54,63],[63,64],[64,65],[65,66],
    [54,67],[67,68],[68,69],[69,70],
    [54,71],[71,72],[72,73],[73,74],
]

def sample_frames(arr, n):
    """Amostra uniformemente n frames de arr (T, 225)."""
    T = len(arr)
    if T == 0:
        return []
    idx = [int(i * T / n) for i in range(n)]
    return [arr[min(i, T-1)].tolist() for i in idx]

def normalize_lm(frame_lm):
    """Normaliza landmarks: centraliza no quadril, escala por ombro."""
    pts = np.array(frame_lm).reshape(75, 3)
    # Quadris: pose lm 23,24 → índices 23,24
    hip = (pts[23] + pts[24]) / 2
    pts -= hip
    # Ombros: pose lm 11,12
    shoulder_dist = np.linalg.norm(pts[11] - pts[12])
    if shoulder_dist > 1e-6:
        pts /= shoulder_dist
    return [[round(float(v), 3) for v in p] for p in pts[:, :2]]  # só x,y, 3 decimais

print("Carregando segmentos...", flush=True)
segs_raw = json.load(open(SEGS_FILE))
segs_good = [s for s in segs_raw if MIN_DUR <= s["duration"] <= MAX_DUR]
print(f"  {len(segs_raw)} total → {len(segs_good)} válidos ({MIN_DUR}-{MAX_DUR}s)", flush=True)

# Tenta carregar landmarks salvos
landmarks = None
if LM_PKL.exists():
    print("Carregando landmarks...", flush=True)
    with open(LM_PKL, "rb") as f:
        landmarks = pickle.load(f)  # (T, 225) numpy array
    print(f"  {len(landmarks)} frames", flush=True)

print("Montando JSON de segmentos...", flush=True)
out_segs = []
for s in segs_good:
    entry = {
        "id":           s["seg_id"],
        "t_start":      round(s["t_start"], 3),
        "t_end":        round(s["t_end"], 3),
        "duration":     round(s["duration"], 3),
        "frame_start":  s["frame_start"],
        "frame_end":    s["frame_end"],
        "pred_class":   s["top5"][0][0] if s["top5"] else "unknown",
        "pred_conf":    round(s["top5"][0][1], 3) if s["top5"] else 0,
        "top5":         [[c, round(p, 3)] for c, p in s["top5"][:5]],
    }
    # Adiciona landmarks normalizados se disponíveis
    if landmarks is not None:
        fs = max(0, s["frame_start"])
        fe = min(len(landmarks) - 1, s["frame_end"])
        seg_lm = landmarks[fs:fe+1]
        if len(seg_lm) > 0:
            sampled = sample_frames(seg_lm, N_FRAMES_UI)
            entry["keyframes"] = [normalize_lm(f) for f in sampled]
    out_segs.append(entry)

OUT_SEGS.write_text(json.dumps(out_segs, ensure_ascii=False, separators=(",", ":")))
print(f"Salvo: {OUT_SEGS}  ({OUT_SEGS.stat().st_size/1024:.0f} KB)", flush=True)

cfg = {
    "youtube_id":  YOUTUBE_ID,
    "classes":     CLASSES,
    "connections": CONNECTIONS,
    "n_segments":  len(out_segs),
    "version":     "1.0",
}
OUT_CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
print(f"Salvo: {OUT_CFG}", flush=True)
print("Pronto.", flush=True)

# Script pode ser re-executado após landmarks.pkl estar disponível
# para adicionar keyframes ao segments.json:
# python3 src/prepare_annotation_data.py
