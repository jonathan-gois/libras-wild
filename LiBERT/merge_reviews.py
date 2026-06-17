"""Junta um ou mais CSVs exportados pelo site de revisão (LiBERT/review_site/)
num único resultado, agregando por clipe quando há revisões duplicadas
(mesmo clipe revisado por mais de um colaborador).

Uso: env/bin/python LiBERT/merge_reviews.py wild_review_*.csv
"""

import sys
import pandas as pd
from pathlib import Path

from config import LIBERT_DIR


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Uso: env/bin/python LiBERT/merge_reviews.py <csv1> <csv2> ...")
        sys.exit(1)

    dfs = [pd.read_csv(p) for p in paths]
    full = pd.concat(dfs, ignore_index=True)
    print(f"{len(full)} linhas de {len(paths)} arquivo(s), revisores: {sorted(full['reviewer'].unique())}")

    key = full.groupby(["video_id", "seg_id"])
    n_dup = (key.size() > 1).sum()
    if n_dup:
        print(f"{n_dup} clipes revisados por mais de um colaborador (mantidos todos, sem deduplicar)")

    out_path = LIBERT_DIR / "results" / "wild_review_merged.csv"
    full.to_csv(out_path, index=False)
    print(f"Salvo em {out_path}")

    # Resumo rápido de concordância (só onde há predicted_class e real_gloss preenchidos)
    confiante = full[full["status"] == "confiante"].copy()
    confiante = confiante[confiante["real_gloss"].notna() & (confiante["real_gloss"].str.strip() != "")]
    if len(confiante):
        acc = (confiante["match"] == "sim").mean()
        print(f"\nEntre clipes 'confiante' revisados com glosa informada: {acc:.1%} de concordância "
              f"({len(confiante)} clipes)")
        print("\nPor classe proposta:")
        print(confiante.groupby("predicted_class")["match"].apply(lambda s: (s == "sim").mean()).sort_values())


if __name__ == "__main__":
    main()
