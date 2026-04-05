"""
Leave-One-Person-Out (LOPO) com ExtraTrees-1000.

Cada fold: treina nos N-1 sinalizadores, testa no sinalizador restante.
Mede generalização para NOVOS sinalizadores — protocolo mais realista.

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python3 -u src/eval_lopo.py 2>&1 | tee results/lopo.log
"""

import pickle, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "src")
from data_loader import normalize_sequences
from features.gei import compute_gei
from features.kinematics import kinematic_stats
from features.fft_features import sliding_fft

# ── config ────────────────────────────────────────────────────────────────────
PKL_PATH    = Path("data/processed_data_full.pkl")   # usa dataset completo se disponível
PKL_BASE    = Path("data/processed_data.pkl")
RESULTS_DIR = Path("results")
N_ESTIMATORS = 1000
SEED = 42

GEI_H, GEI_W = 64, 48
FFT_WINDOW, FFT_STEP, FFT_TOPK = 16, 8, 5


# ── extrai signer ID do nome do arquivo ─────────────────────────────────────
def signer_from_filename(fname: str) -> str | None:
    """
    Padrões possíveis:
      01acontecer_Sinalizador03_rec1.mp4  → "03"
      Sinalizador03_01acontecer_rec1.mp4  → "03"
    """
    m = re.search(r"[Ss]inalizador(\d{2})", fname)
    if m:
        return m.group(1)
    return None


# ── feature extraction ────────────────────────────────────────────────────────
def extract_features(sequences):
    print("  GEI...", flush=True)
    gei = np.stack([compute_gei(s, H=GEI_H, W=GEI_W) for s in sequences])
    print("  Kinematics...", flush=True)
    kin = np.stack([kinematic_stats(s) for s in sequences])
    print("  FFT...", flush=True)
    fft = np.stack([sliding_fft(s, window_size=FFT_WINDOW,
                                step=FFT_STEP, n_top_freqs=FFT_TOPK)
                    for s in sequences])
    return np.concatenate([gei, kin, fft], axis=1)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # carrega dataset
    pkl = PKL_PATH if PKL_PATH.exists() else PKL_BASE
    print(f"Carregando {pkl}...", flush=True)
    with open(pkl, "rb") as f:
        df = pickle.load(f)
    print(f"  {len(df)} amostras", flush=True)

    # extrai signer id
    df["signer"] = df["filename"].apply(signer_from_filename)
    missing_signer = df["signer"].isna().sum()
    if missing_signer > 0:
        print(f"  WARN: {missing_signer} amostras sem signer ID — serão ignoradas", flush=True)
        df = df[df["signer"].notna()].reset_index(drop=True)

    signers = sorted(df["signer"].unique())
    print(f"  Sinalizadores: {signers}", flush=True)

    # encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["class"])
    class_names = list(le.classes_)

    # extrai sequências normalizadas + features
    seqs_raw = [np.array(r, dtype=np.float32) for r in df["landmarks"]]
    seqs     = normalize_sequences(seqs_raw)
    labels   = df["label"].values

    print("\nExtraindo features para todo o dataset...", flush=True)
    X = extract_features(seqs)
    print(f"  X shape: {X.shape}", flush=True)

    # ── LOPO loop ─────────────────────────────────────────────────────────────
    fold_accs   = []
    all_preds   = []
    all_labels  = []
    all_signers = []

    for s in signers:
        test_mask  = df["signer"] == s
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], labels[train_mask]
        X_test,  y_test  = X[test_mask],  labels[test_mask]

        n_train = train_mask.sum()
        n_test  = test_mask.sum()
        print(f"\n── Signer {s} ──  treino={n_train}  teste={n_test}", flush=True)

        clf = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1,
                                   random_state=SEED)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = (preds == y_test).mean()
        fold_accs.append(acc)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_test.tolist())
        all_signers.extend([s] * n_test)

        print(f"  Acurácia Signer {s}: {acc*100:.2f}%", flush=True)

    # ── resultados globais ────────────────────────────────────────────────────
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n{'='*55}", flush=True)
    print(f"LOPO ExtraTrees-1000: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%", flush=True)

    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=4)
    print("\n", report, flush=True)

    # por signer
    print("\nAcurácia por sinalizador:", flush=True)
    for s, acc in zip(signers, fold_accs):
        print(f"  Signer {s}: {acc*100:.2f}%", flush=True)

    # salva
    results = {
        "protocol": "LOPO",
        "model": "ExtraTrees-1000",
        "dataset": str(pkl),
        "n_signers": len(signers),
        "signers": signers,
        "mean_acc": float(mean_acc),
        "std_acc": float(std_acc),
        "fold_accs": [float(a) for a in fold_accs],
    }
    out = RESULTS_DIR / "lopo_extratrees.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    rep_out = RESULTS_DIR / "lopo_extratrees_report.txt"
    with open(rep_out, "w") as f:
        f.write(f"LOPO ExtraTrees-1000\n")
        f.write(f"Dataset: {pkl}  |  Sinalizadores: {signers}\n")
        f.write(f"CV: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%\n\n")
        for s, acc in zip(signers, fold_accs):
            f.write(f"Signer {s}: {acc*100:.2f}%\n")
        f.write(f"\n{report}")

    print(f"\nSalvo em {out}", flush=True)


if __name__ == "__main__":
    main()
