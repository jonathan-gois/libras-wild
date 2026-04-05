"""
Orquestrador: roda todos os experimentos em sequência e gera o relatório final.

Ordem:
  1. Baseline sklearn  (rápido, ~20 min)
  2. v1 BiLSTM sem aug (já rodando em paralelo)
  3. v2 BiLSTM + aug
  4. v3 Transformer + aug
  5. Gera RESULTADOS.md

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python3 -u src/run_all.py 2>&1 | tee results/run_all.log
"""

import sys, os, json, subprocess, time
from pathlib import Path

BASE = Path("/home/jonathangois/Documentos/Libras_2026")
RESULTS = BASE / "results"
PYTHON = str(BASE / "env/bin/python3")

def run_script(script_name, log_name):
    log_path = RESULTS / log_name
    print(f"\n{'='*60}", flush=True)
    print(f"▶ Rodando {script_name} → {log_name}", flush=True)
    print(f"  Iniciado: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(
            [PYTHON, "-u", str(BASE / "src" / script_name)],
            cwd=str(BASE),
            stdout=lf, stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else f"ERRO (rc={proc.returncode})"
    print(f"  {status} — {elapsed/60:.1f} min", flush=True)
    return ok, elapsed


def load_result(json_name):
    p = RESULTS / json_name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def generate_report():
    """Gera RESULTADOS.md com todos os experimentos."""
    lines = []

    lines += [
        "# Resultados — Reconhecimento de Libras (MINDS-Libras)",
        "",
        f"**Data:** {time.strftime('%Y-%m-%d %H:%M')}  ",
        "**Dataset:** MINDS-Libras (800 amostras, 20 classes, 10-fold CV estratificado)  ",
        "**Hardware:** CPU (sem GPU)  ",
        "",
        "---",
        "",
        "## Referências (Estado da Arte no MINDS-Libras)",
        "",
        "| Referência | Método | Acurácia |",
        "|-----------|--------|----------|",
        "| Rezende et al. [CNN 3D-stream] | RGB + Depth + Skeleton (CNN 3D) | 93.3% |",
        "| De Castro et al. | 3D CNN (RGB only, 10-fold) | 96.0% |",
        "| Rego et al. (2025) | FFT + Kinematics + BiLSTM | **92.0%** |",
        "| Passos et al. (2021) | GEI + SVM | 84.66% |",
        "| Arcanjo et al. | FastDTW | ~96% |",
        "",
        "---",
        "",
        "## Experimentos Realizados",
        "",
    ]

    # ── Baseline sklearn ──
    sk = load_result("baseline_sklearn.json")
    lines += [
        "### 1. Baselines Clássicos (sklearn)",
        "",
        "**Descrição:** Features estáticas (GEI + kinematic_stats + FFT → PCA(200)) "
        "com vários classificadores, 10-fold CV.",
        "",
        "**Features:**",
        "- GEI (64×48 px renderizado de skeleton landmarks) = 3072 dims",
        "- Kinematic stats (mean|std|max|min de velocidade e aceleração) = 1800 dims",
        "- FFT global + sliding window stats = 6300 dims",
        "- **Total: 11172 dims → PCA(200)**",
        "",
    ]
    if sk:
        lines += ["| Classificador | Acurácia | ±std |", "|--------------|----------|------|"]
        for r in sorted(sk["results"], key=lambda x: -x["mean"]):
            lines.append(f"| {r['name']} | {r['mean']*100:.2f}% | {r['std']*100:.2f}% |")
        best = max(sk["results"], key=lambda x: x["mean"])
        lines += [
            "",
            f"**Melhor baseline:** `{best['name']}` com **{best['mean']*100:.2f}% ± {best['std']*100:.2f}%**",
        ]
    else:
        lines += ["*(resultado pendente)*"]

    lines += ["", "---", ""]

    # ── v1 BiLSTM ──
    v1 = load_result("cv_results.json")
    lines += [
        "### 2. Modelo v1 — BiLSTM Fusion (sem augmentação)",
        "",
        "**Descrição:** Modelo de fusão multimodal com BiLSTM temporal.",
        "",
        "**Arquitetura:**",
        "```",
        "GEI (3072) → MLP(512→128) ─┐",
        "pos+vel+acc (200×675) → BiLSTM(256×2) → proj(256) ─┤→ FC(512→256→20)",
        "kinematic_stats+FFT (8100) → MLP(512→128) ──────────┘",
        "```",
        "**Total params:** ~9.6M  ",
        "**Scheduler:** CosineAnnealingLR | **Loss:** CrossEntropy + label_smoothing=0.1",
        "",
    ]
    if v1:
        folds_str = " | ".join(f"{a*100:.1f}%" for a in v1['fold_accs'])
        lines += [
            f"**Resultado:** {v1['mean_acc']*100:.2f}% ± {v1['std_acc']*100:.2f}%  ",
            f"Folds: {folds_str}",
        ]
    else:
        lines += ["*(resultado pendente)*"]

    lines += ["", "---", ""]

    # ── v2 BiLSTM + aug ──
    v2 = load_result("v2_bilstm_aug.json")
    lines += [
        "### 3. Modelo v2 — BiLSTM + Data Augmentation",
        "",
        "**Descrição:** Mesma arquitetura do v1 com augmentação no treino.",
        "",
        "**Augmentações:**",
        "- Mirror (espelhar eixo X, trocar mão esq/dir) → dobra o dataset",
        "- Time stretch (taxa 0.7×–1.3×, prob 50%)",
        "- Gaussian noise (σ=0.005, prob 70%)",
        "- Temporal crop (85%–100% da sequência, prob 50%)",
        "- Coordinate dropout (p=5% por landmark, prob 30%)",
        "- Mixup (α=0.2) no batch",
        "",
        "**Dataset treino:** 800 originais → ~2400 por fold (mirror + 1 extra)",
        "**Scheduler:** OneCycleLR | **Loss:** CrossEntropy + label_smoothing=0.15",
        "",
    ]
    if v2:
        folds_str = " | ".join(f"{a*100:.1f}%" for a in v2['fold_accs'])
        lines += [
            f"**Resultado:** {v2['mean_acc']*100:.2f}% ± {v2['std_acc']*100:.2f}%  ",
            f"Folds: {folds_str}",
        ]
    else:
        lines += ["*(resultado pendente)*"]

    lines += ["", "---", ""]

    # ── v3 Transformer ──
    v3 = load_result("v3_transformer_aug.json")
    lines += [
        "### 4. Modelo v3 — Transformer + Data Augmentation",
        "",
        "**Descrição:** Substitui BiLSTM por Transformer Encoder com token [CLS].",
        "",
        "**Arquitetura temporal:**",
        "```",
        "pos+vel+acc (200×675) → Linear(675→256) → Positional Encoding",
        "→ [CLS] + Transformer(4 layers, 8 heads, FFN=512, Pre-LN) → CLS token → proj(256)",
        "```",
        "**Melhorias sobre v2:**",
        "- Pre-LayerNorm (mais estável que Post-LN)",
        "- GELU activation em todos os MLPs",
        "- LayerNorm antes do classificador final",
        "- Atenção multi-cabeça capta dependências de longo alcance",
        "",
    ]
    if v3:
        folds_str = " | ".join(f"{a*100:.1f}%" for a in v3['fold_accs'])
        lines += [
            f"**Resultado:** {v3['mean_acc']*100:.2f}% ± {v3['std_acc']*100:.2f}%  ",
            f"Folds: {folds_str}",
        ]
    else:
        lines += ["*(resultado pendente)*"]

    lines += ["", "---", ""]

    # ── Sumário ──
    lines += [
        "## Sumário Comparativo",
        "",
        "| Modelo | Método | Acurácia | Δ vs Rego 2025 |",
        "|--------|--------|----------|---------------|",
        "| Passos et al. (2021) | GEI + SVM | 84.66% | -7.3pp |",
        "| Rego et al. (2025) | FFT+Kin+BiLSTM | 92.0% | baseline |",
        "| Rezende et al. | CNN 3D | 93.3% | +1.3pp |",
    ]
    for tag, label, data in [
        ("baseline", "Sklearn (melhor)", sk),
        ("v1", "BiLSTM fusion (sem aug)", v1),
        ("v2", "BiLSTM + aug + mixup", v2),
        ("v3", "Transformer + aug", v3),
    ]:
        if data:
            if tag == "baseline":
                best = max(data["results"], key=lambda x: x["mean"])
                acc, std = best["mean"]*100, best["std"]*100
            else:
                acc, std = data["mean_acc"]*100, data["std_acc"]*100
            delta = acc - 92.0
            sign = "+" if delta >= 0 else ""
            lines.append(f"| **{label}** | - | **{acc:.2f}% ± {std:.2f}%** | {sign}{delta:.1f}pp |")

    lines += [
        "",
        "---",
        "",
        "## Análise de Features",
        "",
        "### Contribuição de cada modalidade",
        "",
        "As três fontes de features têm papéis complementares:",
        "",
        "| Feature | Informação capturada | Dimensão |",
        "|---------|---------------------|----------|",
        "| GEI (Gait Energy Image) | Envelope espacial do movimento (silhueta média) | 3072 |",
        "| Kinematic stats | Distribuição de velocidades e acelerações | 1800 |",
        "| FFT sliding window | Padrões de frequência e ritmo do sinal | 6300 |",
        "| pos+vel+acc (temporal) | Trajetória completa frame-a-frame | 200×675 |",
        "",
        "### Decisões de design",
        "",
        "1. **GEI via skeleton**: Como não temos os frames de vídeo, "
        "o GEI é computado renderizando as conexões do esqueleto 2D em imagens 64×48 e "
        "calculando a média pixel-a-pixel. Captura o envelope espacial do movimento.",
        "",
        "2. **Normalização**: Centralização pelo centro dos quadris e escala "
        "pela distância ombro-a-ombro — torna as features invariantes a "
        "distância da câmera e posição do sinalizador.",
        "",
        "3. **Augmentação mirror**: Simula sinalizadores canhotos. "
        "Libras usa predominantemente a mão dominante; espelhar o sinal "
        "cria uma variante válida e aumenta diversidade.",
        "",
        "4. **Mixup**: Interpolação linear entre pares de amostras no espaço "
        "de features. Regulariza sem criar artefatos temporais.",
        "",
        "---",
        "",
        "## Próximos Passos",
        "",
        "- [ ] Download dos 4 sinalizadores ausentes (03, 04, 07, 09) → 1200 amostras totais",
        "- [ ] Ensemble v2 + v3 com soft-voting",
        "- [ ] Avaliação Leave-One-Person-Out (LOPO) para medir generalização real",
        "- [ ] Arquitetura seq2seq para reconhecimento contínuo (além de sinais isolados)",
        "",
        "---",
        "",
        "*Documento gerado automaticamente por `src/run_all.py`*",
    ]

    report_path = BASE / "RESULTADOS.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nRelatório salvo em {report_path}", flush=True)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)

    # Verifica se o v1 já está rodando
    import subprocess as sp
    v1_running = bool(sp.run(
        ["pgrep", "-f", "src/train.py"], capture_output=True
    ).stdout)

    print("=== PIPELINE COMPLETO ===", flush=True)
    print(f"v1 já rodando: {v1_running}", flush=True)

    # 1. Baseline sklearn
    run_script("baseline_sklearn.py", "baseline_sklearn.log")

    # 2. v2 BiLSTM + aug
    run_script("train_v2.py", "train_v2.log")

    # 3. v3 Transformer + aug
    run_script("train_v3.py", "train_v3.log")

    # 4. Relatório
    generate_report()
    print("\n=== PIPELINE CONCLUÍDO ===", flush=True)
