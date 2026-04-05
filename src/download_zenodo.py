"""
Baixa sinalizadores 03, 04, 07, 09 do Zenodo (registro 4322984).
Cada zip tem ~2.5-3 GB. Extrai vídeos RGB, roda MediaPipe, salva pkl.

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python3 -u src/download_zenodo.py 2>&1 | tee results/zenodo_download.log
"""

import os, re, sys, zipfile, tempfile, pickle, json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── configuração ────────────────────────────────────────────────────────────
MISSING     = ["03", "04", "07", "09"]
ZENODO_BASE = "https://zenodo.org/api/records/4322984/files"
OUT_DIR     = Path("data/videos_zenodo")
BASE_PKL    = Path("data/processed_data.pkl")
FULL_PKL    = Path("data/processed_data_full.pkl")

CLASSES = {
    "01":"acontecer","02":"aluno","03":"amarelo","04":"america",
    "05":"aproveitar","06":"bala","07":"banco","08":"banheiro",
    "09":"barulho","10":"cinco","11":"conhecer","12":"espelho",
    "13":"esquina","14":"filho","15":"maca","16":"medo",
    "17":"ruim","18":"sapo","19":"vacina","20":"vontade",
}

# ── mediapipe ────────────────────────────────────────────────────────────────
def extract_with_mediapipe(video_path: Path):
    try:
        import cv2
        import mediapipe as mp
        mp_hol = mp.solutions.holistic
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        with mp_hol.Holistic(static_image_mode=False, model_complexity=1,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as hol:
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
        cap.release()
        if len(frames) >= 10:
            return np.array(frames, dtype=np.float32)
        return None
    except Exception as e:
        print(f"  Erro MediaPipe {video_path.name}: {e}", flush=True)
        return None


# ── identifica classe pelo nome do arquivo ────────────────────────────────────
def class_from_filename(fname: str, zip_path: str = "") -> str | None:
    """
    Extrai classe a partir do nome do arquivo ou do caminho no zip.

    Formatos do MINDS-Libras (Zenodo):
      3-01Acontecer_1RGB.mp4           → classe "01" → acontecer
      Sinalizador03/01Acontecer/...    → diretório "01Acontecer"

    Extrai o número de 2 dígitos imediatamente após o hífen no nome,
    ou do diretório pai no caminho do zip.
    """
    # Formato primário: <signer>-<NN><ClassName>_<rep>RGB.mp4
    m = re.match(r"^\d+-(\d{2})", fname)
    if m:
        return CLASSES.get(m.group(1))

    # Fallback: diretório no zip (Sinalizador03/01Acontecer/...)
    parts = zip_path.replace("\\", "/").split("/")
    for part in parts:
        m = re.match(r"^(\d{2})[A-Z]", part)
        if m:
            return CLASSES.get(m.group(1))

    # Fallback final: dois dígitos em qualquer posição
    m = re.search(r"[-_](\d{2})[A-Za-z]", fname)
    if m:
        return CLASSES.get(m.group(1))
    return None


# ── download de um zip de sinalizador ────────────────────────────────────────
def download_and_process(signer: str, out_dir: Path) -> list[dict]:
    url = f"{ZENODO_BASE}/Sinalizador{signer}.zip/content"
    print(f"\n[Sinalizador{signer}] Baixando de {url}", flush=True)

    # download para arquivo temporário
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False,
                                    dir="data/", prefix=f"s{signer}_") as tmp:
        tmp_path = Path(tmp.name)

    sess = requests.Session()
    r = sess.get(url, stream=True, timeout=120)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} — abortando Sinalizador{signer}", flush=True)
        return []

    total = 0
    chunk = 1024 * 1024
    with open(tmp_path, "wb") as f:
        for data in r.iter_content(chunk):
            f.write(data)
            total += len(data)
            if total % (200 * 1024 * 1024) < chunk:
                print(f"  {total/1e9:.2f} GB baixados...", flush=True)

    print(f"  Download completo: {total/1e9:.2f} GB → {tmp_path}", flush=True)

    # inspeciona estrutura do zip
    print("  Inspecionando estrutura do zip...", flush=True)
    rows = []
    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            all_names = zf.namelist()
            # filtra vídeos RGB (evita depth/IR)
            mp4s = [n for n in all_names
                    if n.lower().endswith(".mp4")
                    and "depth" not in n.lower()
                    and "ir" not in n.lower()]
            print(f"  {len(all_names)} entradas no zip, {len(mp4s)} vídeos RGB", flush=True)
            if mp4s:
                print(f"  Exemplos: {mp4s[:3]}", flush=True)

            for i, name in enumerate(mp4s):
                fname = Path(name).name
                cls = class_from_filename(fname, zip_path=name)
                if cls is None:
                    print(f"  WARN: não reconheci classe de '{fname}' ({name})", flush=True)
                    continue

                # extrai para pasta temporária
                vdir = out_dir / f"S{signer}"
                vdir.mkdir(parents=True, exist_ok=True)
                vpath = vdir / fname
                if not vpath.exists():
                    with zf.open(name) as src, open(vpath, "wb") as dst:
                        dst.write(src.read())

                lm = extract_with_mediapipe(vpath)
                if lm is not None:
                    rows.append({"filename": fname, "class": cls,
                                 "landmarks": lm.tolist()})
                else:
                    print(f"  SKIP (mediapipe falhou): {fname}", flush=True)

                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(mp4s)} — {len(rows)} ok", flush=True)

    except zipfile.BadZipFile as e:
        print(f"  Zip corrompido: {e}", flush=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"  [Sinalizador{signer}] {len(rows)} landmarks extraídos", flush=True)
    return rows


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(BASE_PKL, "rb") as f:
        df_base = pickle.load(f)
    print(f"Dataset base: {len(df_base)} amostras", flush=True)

    # Verifica quais sinalizadores já estão no pkl completo
    if FULL_PKL.exists():
        with open(FULL_PKL, "rb") as f:
            df_existing = pickle.load(f)
        print(f"pkl completo existente: {len(df_existing)} amostras", flush=True)
    else:
        df_existing = df_base

    all_new_rows = []
    for s in MISSING:
        rows = download_and_process(s, OUT_DIR)
        all_new_rows.extend(rows)
        print(f"Total novo até agora: {len(all_new_rows)}", flush=True)

        # salva checkpoint incremental
        if rows:
            df_check = pd.concat(
                [df_existing, pd.DataFrame(all_new_rows)], ignore_index=True
            )
            with open(FULL_PKL, "wb") as f:
                pickle.dump(df_check, f)
            print(f"  Checkpoint salvo: {len(df_check)} amostras", flush=True)

    if all_new_rows:
        df_full = pd.concat(
            [df_existing, pd.DataFrame(all_new_rows)], ignore_index=True
        )
    else:
        df_full = df_existing

    with open(FULL_PKL, "wb") as f:
        pickle.dump(df_full, f)

    print(f"\nDataset final: {len(df_full)} amostras → {FULL_PKL}", flush=True)
    classes_ok = df_full["class"].value_counts().sort_index()
    print(classes_ok, flush=True)


if __name__ == "__main__":
    main()
