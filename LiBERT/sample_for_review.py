"""Amostragem estratificada do CSV de pré-anotação para revisão manual.

Issue: jonathan-gois/libras-wild#4 (depende de #1, já implementada em
`preannotate_wild.py`, que gera `LiBERT/results/wild_preannotation.csv`).

O que faz:
    Lê `LiBERT/results/wild_preannotation.csv` (uma linha por clipe do dataset
    wild, com colunas video_id, seg_id, clip_path, t_start, t_end,
    predicted_class, similarity, status) e gera uma amostra de tamanho N para
    revisão manual humana, salva em `LiBERT/results/wild_sample_for_review.csv`.

Estratégia de amostragem (estratificada em dois níveis):
    1. Separa o CSV em dois grupos por `status`: "confiante" e "desconhecido".
       A amostra final tem ~metade de cada grupo (N//2 confiante, resto
       desconhecido), porque o objetivo é estimar separadamente a precisão
       dos clipes "confiantes" (taxa de acerto do matching) e a taxa de
       "desconhecidos" que na verdade tinham uma classe correta entre as 20
       conhecidas.
    2. Dentro do grupo "confiante", uma amostra puramente aleatória ficaria
       dominada pela classe mais frequente: em ~4.641 clipes confiantes,
       3.277 (70%) caem em "acontecer". Para dar visibilidade às outras
       classes propostas, a amostra de "confiante" é estratificada também por
       `predicted_class`: cada classe presente recebe uma cota mínima de
       exemplos (até o limite de exemplos que ela tem disponível), com um
       teto por classe para não deixar a classe dominante (ex. "acontecer")
       engolir a cota toda. Sobras de cota (classes pequenas que não têm
       exemplos suficientes para preencher seu teto) são redistribuídas via
       amostragem aleatória entre as classes que ainda têm clipes não
       amostrados, até completar o tamanho alvo do grupo confiante.
    3. Dentro do grupo "desconhecido" não há classe proposta confiável para
       estratificar (status existe justamente porque o matching não bateu
       com confiança), então a amostra é aleatória simples.

Seed fixa: usa SEED = 42 (mesma convenção de `LiBERT/config.py`), garantindo
que rodar o script de novo produz exatamente a mesma amostra.

Como rodar:
    /home/jonathangois/Documentos/Libras_2026/env/bin/python LiBERT/sample_for_review.py [--n N]

Saída:
    LiBERT/results/wild_sample_for_review.csv com colunas:
        video_id, seg_id, clip_path, t_start, t_end, predicted_class,
        similarity, status, revisor_correto
    - revisor_correto: coluna vazia para o revisor humano preencher (ex.
      "sim", "não", ou a glosa correta quando predicted_class estiver errado).

Como abrir um clipe amostrado:
    `clip_path` vem do CSV de entrada relativo a `dataset/wild/`, ex.
    "-aaTTl2RFFI/clips/seg_0005_339.mp4". O caminho completo a partir da raiz
    do repo é `dataset/wild/<clip_path>`, ex.:
        dataset/wild/-aaTTl2RFFI/clips/seg_0005_339.mp4

Não faz a revisão manual em si (não assiste vídeos, não preenche
revisor_correto) — só gera a planilha de amostra.
"""

import argparse

import numpy as np
import pandas as pd

from config import LIBERT_DIR, SEED

RESULTS_DIR = LIBERT_DIR / "results"
INPUT_CSV = RESULTS_DIR / "wild_preannotation.csv"
OUTPUT_CSV = RESULTS_DIR / "wild_sample_for_review.csv"

DEFAULT_N = 120
# Teto de exemplos por predicted_class dentro do grupo "confiante", para que
# a classe dominante ("acontecer", ~70% do grupo) não ocupe toda a cota.
MAX_PER_CLASS = 8


def stratified_sample_confiante(df_conf: pd.DataFrame, n_target: int, rng) -> pd.DataFrame:
    """Amostra `n_target` linhas de df_conf, estratificando por predicted_class.

    Usa alocação round-robin (1 exemplo por classe por rodada, até
    MAX_PER_CLASS) em vez de preencher uma classe inteira antes de passar
    para a próxima — isso garante que toda classe presente seja contemplada
    antes de qualquer classe (ex. a dominante "acontecer") atingir seu teto.
    Se a soma das cotas por classe não atingir n_target (poucas classes ou
    poucos clipes), completa com amostragem aleatória entre as linhas
    restantes ainda não escolhidas.
    """
    classes = list(df_conf["predicted_class"].unique())
    # Pools embaralhados por classe, para que o round-robin consuma em ordem
    # aleatória (mas reprodutível) dentro de cada classe.
    pools = {
        cls: df_conf[df_conf["predicted_class"] == cls]
        .sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1))
        .index.tolist()
        for cls in classes
    }
    taken_per_class = {cls: 0 for cls in classes}
    chosen_idx = []

    progressed = True
    while len(chosen_idx) < n_target and progressed:
        progressed = False
        for cls in classes:
            if len(chosen_idx) >= n_target:
                break
            if taken_per_class[cls] >= MAX_PER_CLASS:
                continue
            if not pools[cls]:
                continue
            chosen_idx.append(pools[cls].pop())
            taken_per_class[cls] += 1
            progressed = True

    # Completa a cota faltante (todas as classes bateram MAX_PER_CLASS ou
    # ficaram sem exemplos) com amostragem aleatória simples entre as linhas
    # ainda não escolhidas.
    remaining_needed = n_target - len(chosen_idx)
    if remaining_needed > 0:
        leftover = df_conf.drop(index=chosen_idx)
        take = min(remaining_needed, len(leftover))
        if take > 0:
            extra = leftover.sample(n=take, random_state=rng.integers(0, 2**32 - 1))
            chosen_idx.extend(extra.index.tolist())

    return df_conf.loc[chosen_idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                         help=f"Tamanho total da amostra (default: {DEFAULT_N})")
    args = parser.parse_args()
    n_total = args.n

    df = pd.read_csv(INPUT_CSV)
    print(f"CSV de entrada: {INPUT_CSV} ({len(df)} linhas)")

    rng = np.random.default_rng(SEED)

    df_conf = df[df["status"] == "confiante"].copy()
    df_unk = df[df["status"] == "desconhecido"].copy()
    print(f"confiante disponível: {len(df_conf)}, desconhecido disponível: {len(df_unk)}")

    n_conf_target = n_total // 2
    n_unk_target = n_total - n_conf_target

    n_conf_target = min(n_conf_target, len(df_conf))
    n_unk_target = min(n_unk_target, len(df_unk))

    sample_conf = stratified_sample_confiante(df_conf, n_conf_target, rng)
    sample_unk = df_unk.sample(n=n_unk_target, random_state=SEED)

    sample = pd.concat([sample_conf, sample_unk], ignore_index=True)
    sample["revisor_correto"] = ""

    cols = ["video_id", "seg_id", "clip_path", "t_start", "t_end",
            "predicted_class", "similarity", "status", "revisor_correto"]
    sample = sample[cols]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_CSV, index=False)

    print(f"\nAmostra salva em {OUTPUT_CSV}")
    print(f"total: {len(sample)} (confiante: {len(sample_conf)}, desconhecido: {len(sample_unk)})")
    print("\nDistribuição de predicted_class dentro do grupo confiante amostrado:")
    print(sample_conf["predicted_class"].value_counts())


if __name__ == "__main__":
    main()
