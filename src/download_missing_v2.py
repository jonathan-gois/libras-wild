"""
Download dos sinalizadores ausentes (03, 04, 07, 09) usando
kaggle CLI com flag -f por arquivo.
"""
import subprocess, sys, os, re, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd

MISSING = ["03", "04", "07", "09"]
OUT_DIR = Path("data/videos_missing")
BASE_PKL = Path("data/processed_data.pkl")
FULL_PKL = Path("data/processed_data_full.pkl")
DATASET = "j0aopsantos/minds-libras"

CLASSES = {
    "01":"acontecer","02":"aluno","03":"amarelo","04":"america",
    "05":"aproveitar","06":"bala","07":"banco","08":"banheiro",
    "09":"barulho","10":"cinco","11":"conhecer","12":"espelho",
    "13":"esquina","14":"filho","15":"maca","16":"medo",
    "17":"ruim","18":"sapo","19":"vacina","20":"vontade",
}

CLASSES_CAPITALIZED = {k: v.capitalize() for k,v in CLASSES.items()}

def list_missing_files():
    """Lista arquivos dos sinalizadores ausentes via kaggle API."""
    result = subprocess.run(
        ["kaggle", "datasets", "files", DATASET, "--csv", "--page-size", "2000"],
        capture_output=True, text=True
    )
    files = []
    for line in result.stdout.strip().split("\n")[1:]:
        fname = line.split(",")[0].strip()
        if fname and any(f"Sinalizador{s}" in fname for s in MISSING):
            files.append(fname)
    return files

def download_file(fname):
    out = OUT_DIR / fname
    if out.exists() and out.stat().st_size > 500_000:
        return True
    result = subprocess.run(
        ["kaggle", "datasets", "download", DATASET, "-f", fname, "-p", str(OUT_DIR)],
        capture_output=True, text=True
    )
    # Renomear se veio com .zip
    zipped = OUT_DIR / (fname + ".zip")
    if zipped.exists():
        import zipfile
        with zipfile.ZipFile(zipped) as z:
            z.extractall(OUT_DIR)
        zipped.unlink()
    return out.exists() and out.stat().st_size > 500_000

def extract_with_mediapipe(video_path):
    try:
        import cv2, mediapipe as mp
        mp_hol = mp.solutions.holistic
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        with mp_hol.Holistic(static_image_mode=False, model_complexity=1,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as hol:
            while True:
                ret, frame = cap.read()
                if not ret: break
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
        cap.release()
        return np.array(frames, dtype=np.float32) if len(frames) >= 10 else None
    except Exception as e:
        print(f"  Erro {video_path.name}: {e}", flush=True)
        return None

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Listando arquivos ausentes no Kaggle...", flush=True)
    files = list_missing_files()
    print(f"  {len(files)} arquivos a baixar", flush=True)

    ok = 0
    for i, f in enumerate(files):
        if download_file(f):
            ok += 1
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(files)} processados ({ok} ok)...", flush=True)
    print(f"Download: {ok}/{len(files)} ok", flush=True)

    # Extrai landmarks
    print("\nExtraindo landmarks MediaPipe...", flush=True)
    new_rows = []
    videos = sorted(OUT_DIR.glob("*.mp4"))
    print(f"  {len(videos)} vídeos encontrados", flush=True)
    for i, vp in enumerate(videos):
        m = re.match(r"(\d{2})\w+Sinalizador", vp.name)
        if not m: continue
        class_name = CLASSES.get(m.group(1), "unknown")
        lm = extract_with_mediapipe(vp)
        if lm is not None:
            new_rows.append({"filename": vp.name, "class": class_name,
                              "landmarks": lm.tolist()})
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(videos)} extraídos...", flush=True)

    print(f"  {len(new_rows)} novos landmarks extraídos", flush=True)

    with open(BASE_PKL, "rb") as f:
        df_base = pickle.load(f)

    if new_rows:
        df_full = pd.concat([df_base, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_full = df_base

    with open(FULL_PKL, "wb") as f:
        pickle.dump(df_full, f)
    print(f"Dataset completo: {len(df_full)} amostras → {FULL_PKL}", flush=True)

if __name__ == "__main__":
    main()
