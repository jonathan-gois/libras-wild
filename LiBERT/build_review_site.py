"""Monta o site estático de revisão (LiBERT/review_site/) a partir de
LiBERT/results/wild_sample_for_review.csv: copia os 120 clipes pra uma pasta
flat e gera o clips.json que o app.js consome.

Uso: env/bin/python LiBERT/build_review_site.py
"""

import shutil
import json
import pandas as pd
from pathlib import Path

from config import PROJECT_ROOT, LIBERT_DIR

WILD_DIR = PROJECT_ROOT / "dataset" / "wild"
SITE_DIR = LIBERT_DIR / "review_site"
CLIPS_OUT = SITE_DIR / "clips"
REVIEW_CSV = LIBERT_DIR / "results" / "wild_sample_for_review.csv"


def main():
    CLIPS_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REVIEW_CSV)

    records = []
    for _, row in df.iterrows():
        flat_name = f"{row['video_id']}__{row['seg_id']}.mp4"
        src = WILD_DIR / row["clip_path"]
        dst = CLIPS_OUT / flat_name
        if not dst.exists():
            shutil.copy(src, dst)
        records.append({
            "video_id": row["video_id"],
            "seg_id": row["seg_id"],
            "clip_file": flat_name,
            "t_start": float(row["t_start"]),
            "t_end": float(row["t_end"]),
            "predicted_class": row["predicted_class"],
            "similarity": float(row["similarity"]),
            "status": row["status"],
        })

    with open(SITE_DIR / "clips.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"{len(records)} clipes copiados para {CLIPS_OUT}")
    print(f"clips.json gerado em {SITE_DIR / 'clips.json'}")


if __name__ == "__main__":
    main()
