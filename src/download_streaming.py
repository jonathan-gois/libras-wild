"""
Baixa o zip do MINDS-Libras via streaming e extrai apenas os
sinalizadores ausentes (03, 04, 07, 09), sem ocupar 48 GB em disco.
"""
import sys, os, re, io, zipfile, requests, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

MISSING = {"03", "04", "07", "09"}
OUT_DIR  = Path("data/videos_missing")
BASE_PKL = Path("data/processed_data.pkl")
FULL_PKL = Path("data/processed_data_full.pkl")

CLASSES = {
    "01":"acontecer","02":"aluno","03":"amarelo","04":"america",
    "05":"aproveitar","06":"bala","07":"banco","08":"banheiro",
    "09":"barulho","10":"cinco","11":"conhecer","12":"espelho",
    "13":"esquina","14":"filho","15":"maca","16":"medo",
    "17":"ruim","18":"sapo","19":"vacina","20":"vontade",
}

DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/j0aopsantos/minds-libras"
)

def get_creds():
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        return json.load(f)

def extract_with_mediapipe(video_bytes: bytes, filename: str):
    import cv2, mediapipe as mp, tempfile
    mp_hol = mp.solutions.holistic
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        with mp_hol.Holistic(static_image_mode=False, model_complexity=1,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as hol:
            while True:
                ret, frame = cap.read()
                if not ret: break
                res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                row = []
                for part, n in [(res.pose_landmarks,33),
                                 (res.left_hand_landmarks,21),
                                 (res.right_hand_landmarks,21)]:
                    if part:
                        for lm in part.landmark: row.extend([lm.x, lm.y, lm.z])
                    else:
                        row.extend([0.0]*n*3)
                frames.append(row)
        cap.release()
        return np.array(frames, dtype=np.float32) if len(frames)>=10 else None
    except Exception as e:
        print(f"  Erro {filename}: {e}", flush=True)
        return None
    finally:
        os.unlink(tmp_path)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    creds = get_creds()
    sess  = requests.Session()
    sess.auth = (creds["username"], creds["key"])

    print("Iniciando download streaming do zip (~48 GB)...", flush=True)
    print("Extraindo apenas sinalizadores:", MISSING, flush=True)
    print("(Isso pode levar várias horas)", flush=True)

    r = sess.get(DOWNLOAD_URL, stream=True, timeout=60)
    if r.status_code != 200:
        print(f"Erro HTTP {r.status_code}", flush=True)
        sys.exit(1)

    # Stream para buffer circular — lemos chunk por chunk e alimentamos ZipFile
    buffer = io.BytesIO()
    total_bytes = 0
    chunk_size  = 1024 * 1024  # 1 MB
    new_rows    = []

    print("Processando zip em streaming...", flush=True)

    # Coleta o zip inteiro em memória gerenciada (usa tmpfile em disco)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False,
                                      dir="data/", prefix="minds_libras_") as tmp:
        tmp_path = tmp.name
        for chunk in r.iter_content(chunk_size):
            tmp.write(chunk)
            total_bytes += len(chunk)
            if total_bytes % (100 * 1024 * 1024) < chunk_size:
                print(f"  {total_bytes/1e9:.1f} GB baixados...", flush=True)

    print(f"Download completo: {total_bytes/1e9:.2f} GB", flush=True)
    print("Extraindo vídeos dos sinalizadores ausentes...", flush=True)

    with zipfile.ZipFile(tmp_path, "r") as zf:
        names = [n for n in zf.namelist()
                 if n.endswith(".mp4") and
                 any(f"Sinalizador{s}" in n for s in MISSING)]
        print(f"  {len(names)} vídeos a processar", flush=True)
        for i, name in enumerate(names):
            fname = Path(name).name
            m = re.match(r"(\d{2})", fname)
            if not m: continue
            class_name = CLASSES.get(m.group(1), "unknown")
            video_bytes = zf.read(name)
            lm = extract_with_mediapipe(video_bytes, fname)
            if lm is not None:
                new_rows.append({"filename": fname, "class": class_name,
                                  "landmarks": lm.tolist()})
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(names)} processados ({len(new_rows)} ok)",
                      flush=True)

    os.unlink(tmp_path)
    print(f"{len(new_rows)} novos landmarks extraídos", flush=True)

    with open(BASE_PKL, "rb") as f:
        df_base = pickle.load(f)
    if new_rows:
        df_full = pd.concat([df_base, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_full = df_base
    with open(FULL_PKL, "wb") as f:
        pickle.dump(df_full, f)
    print(f"Dataset salvo: {len(df_full)} amostras → {FULL_PKL}", flush=True)

if __name__ == "__main__":
    main()
