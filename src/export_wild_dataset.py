"""
Exporta anotações coletadas pela ferramenta web para um dataset de landmarks.

Recebe arquivos JSON exportados pelos anotadores e cruza com os landmarks
do vídeo original para gerar processed_data_wild.pkl (mesmo formato do
processed_data.pkl da base MINDS-Libras).

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python3 src/export_wild_dataset.py \\
        --annotations results/annotations_joao.json results/annotations_maria.json \\
        --landmarks   results/youtube/-ZDkdbPqUZg/landmarks.pkl \\
        --segments    results/youtube/-ZDkdbPqUZg/segments.json \\
        --output      data/processed_data_wild.pkl
"""
import argparse, json, pickle, sys
import numpy as np
import pandas as pd
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Exporta dataset wild anotado")
    p.add_argument("--annotations", nargs="+", required=True)
    p.add_argument("--landmarks",   required=True)
    p.add_argument("--segments",    required=True)
    p.add_argument("--output",      default="data/processed_data_wild.pkl")
    p.add_argument("--min-confidence", type=int, default=2,
                   help="Confiança máxima aceita (1=certeza, 2=razoável, 3=incerto)")
    p.add_argument("--valid-only",  action="store_true", default=True)
    return p.parse_args()

def load_annotations(files):
    """Carrega e mescla arquivos JSON de múltiplos anotadores.
    Em caso de conflito, usa anotação com maior confiança (menor número)."""
    ann_map = {}   # seg_id → melhor anotação
    total = 0
    for f in files:
        anns = json.load(open(f))
        total += len(anns)
        for a in anns:
            sid = a["seg_id"]
            if sid not in ann_map or a.get("confidence", 3) < ann_map[sid].get("confidence", 3):
                ann_map[sid] = a
    print(f"  {total} anotações carregadas, {len(ann_map)} segmentos únicos")
    return ann_map

def main():
    args = parse_args()

    print("Carregando anotações...", flush=True)
    ann_map = load_annotations(args.annotations)

    print("Carregando landmarks...", flush=True)
    with open(args.landmarks, "rb") as f:
        landmarks = pickle.load(f)   # (T, 225) numpy
    print(f"  {len(landmarks)} frames")

    print("Carregando segmentos...", flush=True)
    segments = {s["seg_id"]: s for s in json.load(open(args.segments))}

    print("Montando dataset...", flush=True)
    rows = []
    skipped_invalid = 0
    skipped_conf    = 0
    skipped_label   = 0

    for seg_id, ann in ann_map.items():
        # Filtros
        if args.valid_only and ann.get("valid") != "yes":
            skipped_invalid += 1
            continue
        if ann.get("confidence", 3) > args.min_confidence:
            skipped_conf += 1
            continue
        label = ann.get("label")
        if not label or label in ("", "desconhecido", None):
            skipped_label += 1
            continue

        seg = segments.get(seg_id)
        if not seg:
            continue

        fs = max(0, seg["frame_start"])
        fe = min(len(landmarks) - 1, seg["frame_end"])
        lm = landmarks[fs:fe+1]

        if len(lm) < 10:
            continue

        rows.append({
            "filename":  seg_id,
            "class":     label,
            "landmarks": lm.tolist(),
            "annotator": ann.get("annotator", "unknown"),
            "confidence":ann.get("confidence", 2),
            "handshape": ann.get("handshape"),
            "location":  ann.get("location"),
            "movement":  ann.get("movement"),
            "orientation":ann.get("orientation"),
            "facial":    ann.get("facial"),
            "source":    "wild",
        })

    print(f"\nDataset wild: {len(rows)} amostras")
    print(f"  Descartados: {skipped_invalid} inválidos | "
          f"{skipped_conf} baixa conf. | {skipped_label} sem rótulo")

    if not rows:
        print("ERRO: Nenhuma amostra válida.")
        return

    df = pd.DataFrame(rows)
    print("\nDistribuição por classe:")
    print(df["class"].value_counts().sort_index().to_string())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(df, f)
    print(f"\nSalvo: {out}  ({len(df)} amostras)")

    # Relatório por anotador
    print("\nAnotações por pessoa:")
    print(df["annotator"].value_counts().to_string())

if __name__ == "__main__":
    main()
