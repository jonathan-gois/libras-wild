"""
Baselines com classificadores clássicos (sklearn).
Testa: KNN, SVM, RF, XGBoost, LightGBM usando features estáticas
(GEI + kinematic_stats + FFT).

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python3 -u src/baseline_sklearn.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import json, time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_dataset, normalize_sequences
from features.gei import batch_gei
from features.kinematics import batch_kinematic_stats
from features.fft_features import batch_fft_features

PKL_PATH  = "data/processed_data.pkl"
RESULTS_DIR = Path("results")
SEED = 42
N_FOLDS = 10


def extract_static_features(sequences):
    print("  GEI...", flush=True)
    gei  = batch_gei(sequences)                # (N, 3072)
    print("  Kinematics...", flush=True)
    kin  = batch_kinematic_stats(sequences)    # (N, 1800)
    print("  FFT...", flush=True)
    fft  = batch_fft_features(sequences)       # (N, 6300)
    X = np.concatenate([gei, kin, fft], axis=1)
    print(f"  Features shape: {X.shape}", flush=True)
    return X


def run_classifier(name, clf, X, y, n_folds=N_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    t0 = time.time()
    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    elapsed = time.time() - t0
    mean, std = scores.mean(), scores.std()
    print(f"  {name:30s}: {mean*100:.2f}% ± {std*100:.2f}%  ({elapsed:.1f}s)", flush=True)
    return {"name": name, "mean": mean, "std": std, "folds": scores.tolist()}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)

    print("=== Baseline Sklearn ===", flush=True)
    print("Carregando dataset...", flush=True)
    seqs, labels, classes = load_dataset(PKL_PATH)
    seqs = normalize_sequences(seqs)
    print(f"  {len(seqs)} amostras, {len(classes)} classes", flush=True)

    print("Extraindo features estáticas...", flush=True)
    X = extract_static_features(seqs)
    y = labels

    # Pipelines
    classifiers = {
        "KNN-k5":
            Pipeline([("scaler", StandardScaler()),
                      ("pca",    PCA(n_components=200, random_state=SEED)),
                      ("clf",    KNeighborsClassifier(n_neighbors=5, n_jobs=-1))]),

        "KNN-k3":
            Pipeline([("scaler", StandardScaler()),
                      ("pca",    PCA(n_components=200, random_state=SEED)),
                      ("clf",    KNeighborsClassifier(n_neighbors=3, n_jobs=-1))]),

        "SVM-RBF":
            Pipeline([("scaler", StandardScaler()),
                      ("pca",    PCA(n_components=200, random_state=SEED)),
                      ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                                    decision_function_shape="ovr"))]),

        "SVM-Linear":
            Pipeline([("scaler", StandardScaler()),
                      ("pca",    PCA(n_components=200, random_state=SEED)),
                      ("clf",    SVC(kernel="linear", C=1.0))]),

        "RF-200":
            Pipeline([("scaler", StandardScaler()),
                      ("clf",    RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                        random_state=SEED))]),

        "RF-500":
            Pipeline([("scaler", StandardScaler()),
                      ("clf",    RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                                        random_state=SEED))]),
    }

    # Tentar XGBoost
    try:
        from xgboost import XGBClassifier
        classifiers["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=200, random_state=SEED)),
            ("clf",    XGBClassifier(n_estimators=300, max_depth=6,
                                     learning_rate=0.1, use_label_encoder=False,
                                     eval_metric="mlogloss", n_jobs=-1,
                                     random_state=SEED)),
        ])
    except ImportError:
        pass

    # Tentar LightGBM
    try:
        import lightgbm as lgb
        classifiers["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=200, random_state=SEED)),
            ("clf",    lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                          learning_rate=0.05, n_jobs=-1,
                                          random_state=SEED, verbose=-1)),
        ])
    except ImportError:
        pass

    print(f"\nRodando {len(classifiers)} classificadores ({N_FOLDS}-fold CV):\n", flush=True)
    results = []
    for name, clf in classifiers.items():
        res = run_classifier(name, clf, X, y)
        results.append(res)

    results.sort(key=lambda r: -r["mean"])
    print(f"\n{'='*50}", flush=True)
    print("Ranking:", flush=True)
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']:30s}: {r['mean']*100:.2f}% ± {r['std']*100:.2f}%", flush=True)

    with open(RESULTS_DIR / "baseline_sklearn.json", "w") as f:
        json.dump({"results": results, "n_folds": N_FOLDS,
                   "n_samples": len(seqs), "n_classes": len(classes)}, f, indent=2)
    print(f"\nResultados salvos em {RESULTS_DIR}/baseline_sklearn.json", flush=True)


if __name__ == "__main__":
    main()
