"""
train_wild.py — Treina classificador combinando MINDS (base) + anotações Wild.

Fluxo:
  1. Carrega MINDS (processed_data.pkl) — 800 amostras com rótulos
  2. Carrega anotações Wild do Supabase (ou de arquivo JSON local)
  3. Para cada anotação válida, extrai landmarks do pkl correspondente
  4. Combina e treina modelo v4 (Transformer) ou baseline
  5. Avalia com cross-validation + salva modelo

Uso:
    python3 src/train_wild.py                          # usa anotações do Supabase
    python3 src/train_wild.py --local annotations.json # usa arquivo local
    python3 src/train_wild.py --dry-run                # mostra estatísticas sem treinar
"""

import argparse, json, pickle, sys, os
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── Carrega anotações do Supabase ─────────────────────────────────────────────

def load_from_supabase() -> list[dict]:
    """Baixa todas as anotações da tabela 'annotations' no Supabase."""
    try:
        import httpx
        url = "https://rynxhuistciljkuqqcsh.supabase.co"
        key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ5bnhodWlzdGNpbGprdXFxY3NoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzUzNzI2NDAsImV4cCI6MjA5MDk0ODY0MH0.Dyu9D6mFg7ZGDgruPtWwcUBui1dABCIfKD6GLXWvciw"
        headers = {"apikey": key, "Authorization": f"Bearer {key}"}
        r = httpx.get(f"{url}/rest/v1/annotations?select=*&limit=10000",
                      headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[WARN] Supabase indisponível: {e}")
        return []


# ── Filtra anotações utilizáveis ──────────────────────────────────────────────

KNOWN_CLASSES = [
    "acontecer","aluno","amarelo","america","aproveitar",
    "bala","banco","banheiro","barulho","cinco",
    "conhecer","espelho","esquina","filho","maca",
    "medo","ruim","sapo","vacina","vontade",
]

def filter_annotations(anns: list[dict]) -> list[dict]:
    """
    Mantém anotações onde:
      - valid == 'yes'
      - label está nas classes conhecidas (ou outro_name mapeável)
      - confidence <= 2 (tem certeza ou razoável)
    """
    kept = []
    for a in anns:
        if a.get("valid") != "yes":
            continue
        label = a.get("label") or ""
        if label not in KNOWN_CLASSES:
            # Tenta outro_name
            outro = (a.get("outro_name") or "").lower().strip()
            if outro in KNOWN_CLASSES:
                a = dict(a, label=outro)
            else:
                continue
        if int(a.get("confidence") or 3) > 2:
            continue
        kept.append(a)
    return kept


# ── Extrai landmarks do segmento ──────────────────────────────────────────────

def load_landmarks_for_annotation(ann: dict, wild_root: Path) -> np.ndarray | None:
    """
    Tenta carregar landmarks para uma anotação Wild.
    Procura em: dataset/wild/<video_id>/landmarks.pkl
    """
    vid = ann.get("video_id", "")
    lm_path = wild_root / vid / "landmarks.pkl"
    if not lm_path.exists():
        return None
    lm_full, fps = pickle.load(open(lm_path, "rb"))
    t_start = ann.get("t_start", 0)
    t_end   = ann.get("t_end",   0)
    fs = int(t_start * fps)
    fe = int(t_end   * fps)
    chunk = lm_full[fs:fe+1]
    return chunk if len(chunk) >= 8 else None


# ── Pipeline de treinamento ───────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local",   type=str, help="JSON local com anotações")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model",   choices=["rf","svm","gbt"], default="gbt")
    args = ap.parse_args()

    # 1. MINDS
    print("Carregando MINDS...", flush=True)
    df = pickle.load(open(ROOT / "data" / "processed_data_full.pkl", "rb"))
    minds_X = [np.array(r["landmarks"], dtype=np.float32) for _, r in df.iterrows()]
    minds_y = list(df["class"])
    print(f"  MINDS: {len(minds_X)} amostras, {len(set(minds_y))} classes")

    # 2. Anotações Wild
    if args.local:
        anns_raw = json.load(open(args.local))
    else:
        print("Baixando anotações do Supabase...", flush=True)
        anns_raw = load_from_supabase()
    print(f"  Anotações brutas: {len(anns_raw)}")

    anns = filter_annotations(anns_raw)
    print(f"  Após filtro (valid+label+confiança): {len(anns)}")

    if args.dry_run:
        print("\nDistribuição de classes Wild:")
        for cls, cnt in sorted(Counter(a["label"] for a in anns).items()):
            print(f"  {cls:20s} {cnt}")
        return

    # 3. Extrai landmarks Wild
    wild_root = ROOT / "dataset" / "wild"
    wild_X, wild_y = [], []
    missing = 0
    for a in anns:
        lm = load_landmarks_for_annotation(a, wild_root)
        if lm is None:
            missing += 1
            continue
        wild_X.append(lm)
        wild_y.append(a["label"])
    print(f"  Wild landmarks: {len(wild_X)} ok, {missing} sem arquivo")

    # 4. Combina
    all_X = minds_X + wild_X
    all_y = minds_y + wild_y
    print(f"\nTotal combinado: {len(all_X)} amostras")

    # 5. Features (reutiliza pipeline existente)
    from features.gei import compute_gei_batch
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("Extraindo features GEI...", flush=True)
    # GEI usa sequência de T frames → vetor fixo de 576d
    # Precisamos adaptar pois as sequências têm tamanhos variados
    # Usa normalização por frame count via padding/sampling
    T_fixed = 64
    def pad_or_sample(lm, T):
        if len(lm) >= T:
            idx = np.linspace(0, len(lm)-1, T, dtype=int)
            return lm[idx]
        else:
            reps = int(np.ceil(T / len(lm)))
            return np.tile(lm, (reps, 1))[:T]

    X_fixed = np.stack([pad_or_sample(lm, T_fixed) for lm in all_X])  # (N, T, 225)
    y_arr   = np.array(all_y)

    # Achata para (N, T*225) ou usa feature extrator dedicado
    # Por simplicidade usa velocidade + posição como features fixas
    from sign_spotter import extract_features
    X_feat = np.stack([extract_features(lm) for lm in all_X])

    # 6. Treina
    if args.model == "gbt":
        from sklearn.ensemble import GradientBoostingClassifier
        clf = Pipeline([("sc", StandardScaler()),
                        ("clf", GradientBoostingClassifier(
                            n_estimators=400, max_depth=4,
                            learning_rate=0.05, subsample=0.8, random_state=42))])
    elif args.model == "svm":
        from sklearn.svm import SVC
        clf = Pipeline([("sc", StandardScaler()),
                        ("clf", SVC(kernel="rbf", C=10, probability=True))])
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = Pipeline([("sc", StandardScaler()),
                        ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_feat, y_arr, cv=cv, scoring="accuracy")
    print(f"\nAcurácia 5-fold: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

    clf.fit(X_feat, y_arr)
    out = ROOT / "src" / "models" / "wild_classifier.pkl"
    import joblib
    joblib.dump(clf, out)
    print(f"Modelo salvo: {out}")


if __name__ == "__main__":
    main()
