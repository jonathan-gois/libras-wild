"""
Baixa vídeos dos sinalizadores ausentes (03, 04, 07, 09) e extrai
landmarks MediaPipe, salvando um PKL unificado com todos os 1200 samples.

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python3 -u src/download_missing.py
"""

import os, sys, re, pickle, subprocess
import numpy as np
import pandas as pd
from pathlib import Path

MISSING_SIGNERS = ["03", "04", "07", "09"]
VIDEOS_DIR = Path("data/videos_missing")
OUTPUT_PKL = Path("data/processed_data_full.pkl")
LANDMARKS_PKL = Path("data/processed_data.pkl")

CLASSES = {
    "01": "Acontecer", "02": "Aluno",   "03": "Amarelo",  "04": "America",
    "05": "Aproveitar","06": "Bala",    "07": "Banco",    "08": "Banheiro",
    "09": "Barulho",   "10": "Cinco",   "11": "Conhecer", "12": "Espelho",
    "13": "Esquina",   "14": "Filho",   "15": "Maca",     "16": "Medo",
    "17": "Ruim",      "18": "Sapo",    "19": "Vacina",   "20": "Vontade",
}


def list_kaggle_files():
    """Retorna lista de todos os nomes de arquivo no dataset kaggle."""
    import subprocess
    result = subprocess.run(
        ["kaggle", "datasets", "files", "j0aopsantos/minds-libras",
         "--csv", "--page-size", "2000"],
        capture_output=True, text=True
    )
    files = []
    for line in result.stdout.strip().split("\n")[1:]:
        parts = line.split(",")
        if parts:
            files.append(parts[0].strip())
    return files


def download_missing_videos():
    """Baixa apenas os vídeos dos sinalizadores ausentes via kaggle API."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    print("Listando arquivos no Kaggle...", flush=True)
    all_files = list_kaggle_files()

    target_files = [
        f for f in all_files
        if any(f"Sinalizador{s}" in f for s in MISSING_SIGNERS)
    ]
    print(f"Arquivos a baixar: {len(target_files)}", flush=True)

    # Download individual por arquivo via requests + kaggle token
    import json, requests
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as fp:
        creds = json.load(fp)
    session = requests.Session()
    session.auth = (creds["username"], creds["key"])

    base_url = "https://www.kaggle.com/api/v1/datasets/download"
    ok, fail = 0, 0
    for fname in target_files:
        dest = VIDEOS_DIR / fname
        if dest.exists() and dest.stat().st_size > 1_000_000:
            ok += 1
            continue
        url = f"{base_url}/j0aopsantos/minds-libras?path={fname}"
        try:
            r = session.get(url, stream=True, timeout=60)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
                ok += 1
                if ok % 20 == 0:
                    print(f"  {ok}/{len(target_files)} baixados...", flush=True)
            else:
                fail += 1
                print(f"  ERRO {r.status_code}: {fname}", flush=True)
        except Exception as e:
            fail += 1
            print(f"  Falha {fname}: {e}", flush=True)

    print(f"Download: {ok} ok, {fail} falhas", flush=True)


def extract_landmarks_from_video(video_path: str) -> np.ndarray | None:
    """Extrai landmarks MediaPipe Holistic de um vídeo. Retorna (T, 225) ou None."""
    try:
        import cv2
        import mediapipe as mp

        mp_holistic = mp.solutions.holistic
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frames = []
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(frame_rgb)

                row = []
                # Pose (33 landmarks)
                if result.pose_landmarks:
                    for lm in result.pose_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                else:
                    row.extend([0.0] * 99)
                # Left hand (21)
                if result.left_hand_landmarks:
                    for lm in result.left_hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                else:
                    row.extend([0.0] * 63)
                # Right hand (21)
                if result.right_hand_landmarks:
                    for lm in result.right_hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                else:
                    row.extend([0.0] * 63)

                frames.append(row)

        cap.release()
        if len(frames) < 10:
            return None
        return np.array(frames, dtype=np.float32)

    except Exception as e:
        print(f"  Erro extração {video_path}: {e}", flush=True)
        return None


def build_full_dataset():
    """Combina o PKL existente com os novos vídeos extraídos."""
    print("Carregando dataset base (800 samples)...", flush=True)
    with open(LANDMARKS_PKL, "rb") as f:
        df_base = pickle.load(f)

    new_rows = []
    mp_available = False
    try:
        import mediapipe
        import cv2
        mp_available = True
    except ImportError:
        print("MediaPipe não instalado. Instale com: pip install mediapipe", flush=True)

    if mp_available:
        video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
        print(f"Extraindo landmarks de {len(video_files)} vídeos...", flush=True)

        for i, vf in enumerate(video_files):
            fname = vf.name
            # Determina classe pelo nome do arquivo
            m = re.match(r"(\d{2})(\w+?)Sinalizador", fname)
            if not m:
                continue
            class_num = m.group(1)
            class_name = CLASSES.get(class_num, "unknown").lower()

            landmarks = extract_landmarks_from_video(str(vf))
            if landmarks is not None:
                new_rows.append({
                    "filename": fname,
                    "class": class_name,
                    "landmarks": landmarks.tolist(),
                })
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(video_files)} processados...", flush=True)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_full = pd.concat([df_base, df_new], ignore_index=True)
        print(f"Dataset completo: {len(df_full)} amostras", flush=True)
    else:
        df_full = df_base
        print("Nenhum novo vídeo processado. Usando dataset base.", flush=True)

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(df_full, f)
    print(f"Salvo em {OUTPUT_PKL}", flush=True)
    return df_full


if __name__ == "__main__":
    print("=== Download e extração dos sinalizadores ausentes ===", flush=True)
    print(f"Sinalizadores ausentes: {MISSING_SIGNERS}", flush=True)

    # Instalar mediapipe se necessário
    try:
        import mediapipe
    except ImportError:
        print("Instalando mediapipe...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "mediapipe", "-q"])

    download_missing_videos()
    build_full_dataset()
    print("Concluído!", flush=True)
