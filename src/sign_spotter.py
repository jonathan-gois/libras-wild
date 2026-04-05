"""
sign_spotter.py — Detector binário: "este segmento contém um sinal Libras completo?"

Treina um classificador leve usando os dados MINDS (positivos) e negativos sintéticos
(cortes parciais, holds, transições).

Uso:
    # Treinar e salvar modelo:
    python3 src/sign_spotter.py --train
    # Avaliar um pkl de landmarks:
    python3 src/sign_spotter.py --eval results/youtube/-ZDkdbPqUZg/segments_v2.json
"""

import argparse, json, pickle, sys, os
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = ROOT / "src" / "models" / "sign_spotter.pkl"

# ── Extração de features ──────────────────────────────────────────────────────

def _velocity(seq: np.ndarray) -> np.ndarray:
    """seq: (T, 225) → velocidade (T, 75) por norma L2 de cada landmark."""
    xyz = seq.reshape(len(seq), 75, 3)
    diff = np.diff(xyz, axis=0)  # (T-1, 75, 3)
    diff = np.concatenate([diff[:1], diff], axis=0)  # (T, 75, 3)
    return np.linalg.norm(diff, axis=2)  # (T, 75)


def _normalize_by_shoulder(seq: np.ndarray) -> np.ndarray:
    """Normaliza coordenadas pela distância entre ombros (lm 11,12)."""
    xyz = seq.reshape(len(seq), 75, 3)
    shoulder = np.linalg.norm(xyz[:, 11] - xyz[:, 12], axis=1, keepdims=True)  # (T,1)
    shoulder = np.clip(shoulder, 1e-6, None)
    return (xyz / shoulder[:, :, None]).reshape(len(seq), 225)


def extract_features(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 225) float32
    Retorna vetor de features fixo (independente de T).
    """
    if len(seq) < 4:
        seq = np.tile(seq, (4 // len(seq) + 1, 1))[:4]

    seq = _normalize_by_shoulder(seq)
    vel = _velocity(seq)  # (T, 75)

    # Índices de interesse: pulsos (15,16), mãos esq (33-53), mãos dir (54-74)
    wrist_vel  = vel[:, [15, 16]]                  # (T, 2)
    hand_vel   = np.concatenate([vel[:, 33:54], vel[:, 54:75]], axis=1)  # (T, 42)
    total_vel  = vel.sum(axis=1)                   # (T,)  energia global

    feats = []
    for arr in [wrist_vel, hand_vel]:
        feats += [arr.mean(0), arr.max(0), arr.std(0)]

    # Curva de energia: pico, vale, razão
    e = total_vel
    feats.append([e.max(), e.mean(), e.std(),
                  e.max() / (e.mean() + 1e-9),       # proeminência do pico
                  np.percentile(e, 10),               # energia de hold
                  np.percentile(e, 90),               # energia de stroke
                  float(len(e)),                      # duração em frames
                  e[:len(e)//4].mean(),               # energia início (preparação)
                  e[-len(e)//4:].mean(),              # energia fim (retração)
                  e[len(e)//4: 3*len(e)//4].mean(),  # energia meio (stroke)
                  ])

    # Posição média das mãos (relativa ao quadril)
    xyz = seq.reshape(len(seq), 75, 3)
    hip = (xyz[:, 23] + xyz[:, 24]) / 2
    lhand = (xyz[:, 33:54, :2] - hip[:, None, :2]).mean(axis=1)  # (T, 2)
    rhand = (xyz[:, 54:75, :2] - hip[:, None, :2]).mean(axis=1)
    for arr in [lhand, rhand]:
        feats += [arr.mean(0), arr.std(0)]

    return np.concatenate([np.atleast_1d(f).ravel() for f in feats]).astype(np.float32)


# ── Geração de negativos sintéticos ──────────────────────────────────────────

def make_negatives(samples: list) -> list:
    """
    Gera negativos a partir dos positivos (sinais MINDS):
      1. Corte inicial (preparação sem stroke)
      2. Corte final (retração sem stroke)
      3. Hold artificialmente longo (frames repetidos)
      4. Segmento muito curto (<25% do original)
    """
    negs = []
    rng  = np.random.default_rng(42)

    for seq in samples:
        T = len(seq)
        if T < 8:
            continue

        # 1. Só preparação: primeiros 35%
        cut = max(4, int(T * 0.35))
        negs.append(seq[:cut])

        # 2. Só retração: últimos 35%
        negs.append(seq[-cut:])

        # 3. Hold: repete um frame do meio várias vezes
        mid = T // 2
        hold_len = int(T * rng.uniform(0.4, 0.8))
        negs.append(np.tile(seq[mid:mid+1], (hold_len, 1)))

        # 4. Corte aleatório muito curto
        short = max(3, int(T * rng.uniform(0.1, 0.22)))
        start = rng.integers(0, T - short)
        negs.append(seq[start:start + short])

    return negs


# ── Treino ────────────────────────────────────────────────────────────────────

def train(data_pkl: Path = ROOT / "data" / "processed_data.pkl"):
    print("Carregando MINDS...", flush=True)
    df = pickle.load(open(data_pkl, "rb"))
    positives = [np.array(r["landmarks"], dtype=np.float32) for _, r in df.iterrows()]
    print(f"  {len(positives)} sinais positivos", flush=True)

    negatives = make_negatives(positives)
    print(f"  {len(negatives)} negativos sintéticos gerados", flush=True)

    X = np.stack([extract_features(s) for s in positives + negatives])
    y = np.array([1] * len(positives) + [0] * len(negatives))

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("gbt", GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
    print(f"  F1 cross-val: {scores.mean():.3f} ± {scores.std():.3f}", flush=True)

    clf.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Modelo salvo: {MODEL_PATH}", flush=True)
    return clf


# ── Inferência ────────────────────────────────────────────────────────────────

def load_spotter():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}\nRode: python3 src/sign_spotter.py --train")
    return joblib.load(MODEL_PATH)


def score_segment(clf, landmarks: np.ndarray) -> float:
    """Retorna P(sinal válido) ∈ [0,1]."""
    feat = extract_features(landmarks).reshape(1, -1)
    return float(clf.predict_proba(feat)[0, 1])


def filter_segments(clf, segments: list, landmarks_full: np.ndarray,
                    fps: float, threshold: float = 0.55) -> list:
    """
    Filtra lista de SignSegment (ou dicts com frame_start/frame_end),
    mantendo apenas os com P(válido) >= threshold.
    """
    kept = []
    for seg in segments:
        fs = seg.frame_start if hasattr(seg, "frame_start") else seg["frame_start"]
        fe = seg.frame_end   if hasattr(seg, "frame_end")   else seg["frame_end"]
        chunk = landmarks_full[fs:fe + 1]
        if len(chunk) < 4:
            continue
        p = score_segment(clf, chunk)
        if hasattr(seg, "energy_peak"):
            seg.spotter_score = p
        else:
            seg["spotter_score"] = p
        if p >= threshold:
            kept.append(seg)
    return kept


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Treina e salva o modelo")
    ap.add_argument("--eval",  type=str, default=None,
                    help="JSON de segmentos + pkl de landmarks para avaliar")
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    if args.train:
        train()

    if args.eval:
        clf = load_spotter()
        seg_path = Path(args.eval)
        lm_path  = seg_path.parent / "landmarks.pkl"

        segs = json.load(open(seg_path))
        lm   = pickle.load(open(lm_path, "rb"))
        fps  = 30.0

        kept = []
        for s in segs:
            chunk = lm[s["frame_start"]: s["frame_end"] + 1]
            p = score_segment(clf, chunk)
            s["spotter_score"] = round(p, 3)
            if p >= args.threshold:
                kept.append(s)
            else:
                print(f"  DESCARTADO  {s['seg_id']}  p={p:.3f}  dur={s['duration']:.2f}s")

        print(f"\n{len(segs)} segmentos → {len(kept)} mantidos "
              f"(threshold={args.threshold})")
        out = seg_path.parent / "segments_filtered.json"
        out.write_text(json.dumps(kept, indent=2))
        print(f"Salvo: {out}")
