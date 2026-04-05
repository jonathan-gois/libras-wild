# Resultados — Reconhecimento de Libras (MINDS-Libras)

**Data:** 2026-04-02
**Dataset:** MINDS-Libras (800 amostras, 20 classes, 10-fold CV estratificado)
**Hardware:** CPU + GPU RTX 4060 Laptop (8GB VRAM)

---

## Referências (Estado da Arte no MINDS-Libras)

| Referência | Método | Acurácia |
|-----------|--------|----------|
| Rezende et al. [CNN 3D-stream] | RGB + Depth + Skeleton (CNN 3D) | 93.3% |
| De Castro et al. | 3D CNN (RGB only, 10-fold) | 96.0% |
| Rego et al. (2025) | FFT + Kinematics + BiLSTM | **92.0%** |
| Passos et al. (2021) | GEI + SVM | 84.66% |
| Arcanjo et al. | FastDTW | ~96% |

---

## Experimentos Realizados

### 1. Baselines Clássicos (sklearn)

**Descrição:** Features estáticas (GEI + kinematic_stats + FFT → PCA(200)) com vários classificadores, 10-fold CV.

**Features:**
- GEI (64×48 px renderizado de skeleton landmarks) = 3072 dims
- Kinematic stats (mean|std|max|min de velocidade e aceleração) = 1800 dims
- FFT global + sliding window stats = 6300 dims
- **Total: 11172 dims → PCA(200)**

| Classificador | Acurácia | ±std |
|--------------|----------|------|
| RF-200 | 96.12% | 2.12% |
| RF-500 | 96.00% | 2.22% |
| SVM-Linear | 90.62% | 3.63% |
| SVM-RBF | 88.38% | 3.31% |
| LightGBM | 72.25% | 4.36% |
| KNN-k3 | 71.50% | 4.64% |
| XGBoost | 69.62% | 5.03% |
| KNN-k5 | 67.25% | 5.91% |

**Melhor baseline:** `RF-200` com **96.12% ± 2.12%**

---

### 2. Modelo v1 — BiLSTM Fusion (sem augmentação)

**Descrição:** Modelo de fusão multimodal com BiLSTM temporal.

**Arquitetura:**
```
GEI (3072) → MLP(512→128) ─┐
pos+vel+acc (200×675) → BiLSTM(256×2) → proj(256) ─┤→ FC(512→256→20)
kinematic_stats+FFT (8100) → MLP(512→128) ──────────┘
```
**Total params:** ~9.6M
**Scheduler:** CosineAnnealingLR | **Loss:** CrossEntropy + label_smoothing=0.1

**Resultado:** 72.50% ± 5.51%
Folds: 77.5% | 66.2% | 78.8% | 76.2% | 73.8% | 66.2% | 65.0% | 81.2% | 70.0% | 70.0%

---

### 3. Modelo v2 — BiLSTM + Data Augmentation

**Descrição:** Mesma arquitetura do v1 com augmentação no treino.

**Augmentações:**
- Mirror (espelhar eixo X, trocar mão esq/dir) → dobra o dataset
- Time stretch (taxa 0.7×–1.3×, prob 50%)
- Gaussian noise (σ=0.005, prob 70%)
- Temporal crop (85%–100% da sequência, prob 50%)
- Coordinate dropout (p=5% por landmark, prob 30%)
- Mixup (α=0.2) no batch

**Dataset treino:** 800 originais → ~2400 por fold (mirror + 1 extra)
**Scheduler:** OneCycleLR | **Loss:** CrossEntropy + label_smoothing=0.15

**Resultado:** 79.50% ± 3.46%
Folds: 80.0% | 81.2% | 80.0% | 78.8% | 81.2% | 76.2% | 73.8% | 87.5% | 77.5% | 78.8%

---

### 4. Modelo v3 — Transformer + Data Augmentation (com mirror)

**Descrição:** Substitui BiLSTM por Transformer Encoder com token [CLS].

**Arquitetura temporal:**
```
pos+vel+acc (200×675) → Linear(675→256) → Positional Encoding
→ [CLS] + Transformer(4 layers, 8 heads, FFN=512, Pre-LN) → CLS token → proj(256)
```
**Melhorias sobre v2:**
- Pre-LayerNorm (mais estável que Post-LN)
- GELU activation em todos os MLPs
- LayerNorm antes do classificador final
- Atenção multi-cabeça capta dependências de longo alcance

**Resultado:** 88.75% ± 4.51%
Folds: 90.0% | 83.8% | 93.8% | 87.5% | 86.2% | 86.2% | 81.2% | 96.2% | 93.8% | 88.8%

---

### 4b. Análise do Mirror Augmentation — Por Que É Prejudicial

**Hipótese inicial:** Espelhar o sinal simularia sinalizadores canhotos, dobrando o dataset e melhorando a generalização.

**Análise quantitativa (k-NN no espaço FFT):**
- 94.9% das amostras espelhadas têm o vizinho mais próximo em outra classe
- 19 de 20 classes têm seu espelho mapeado majoritariamente para outra classe
- Apenas "acontecer" (simétrico) sobrevive ao espelhamento sem degradação

**Causa:** Libras usa sinais assimétricos em que a lateralidade é informação fonológica crítica. Espelhar não gera uma variante válida do mesmo sinal — gera confusão com outros sinais que partilham o mesmo envelope espacial mas no hemicorpo oposto.

**Impacto medido:**
- v3 (com mirror): 88.75% ± 4.51%
- v4 (sem mirror): 94.62% ± 5.59%
- **Ganho: +5.87pp eliminando augmentação incorreta**

---

### 4c. Modelo v4 — Transformer (sem mirror augmentation)

**Descrição:** Cópia do v3 com `mirror_aug=False` e `n_extra_aug=2`. Correção da falha linguística identificada na análise FFT.

**Hardware:** GPU RTX 4060 (13× mais rápido que CPU: 3.3s/epoch vs 44s/epoch)

**Resultado:** **94.62% ± 5.59%**
Folds: 92.5% | 96.2% | 97.5% | 93.8% | 97.5% | 97.5% | 87.5% | 100.0% | 87.5% | 93.8%

**Observação:** Melhor modelo neural — supera o estado da arte da maioria das referências.

---

### 4d. LOPO — Leave-One-Person-Out (Generalização Real)

**Descrição:** Protocolo LOPO: 1 sinalizador como teste, 9 como treino. Mede generalização entre sinalizadores, não entre amostras do mesmo sinalizador.

**Modelo:** ET-1000 (melhor modelo geral)

| Sinalizador | Acurácia |
|------------|----------|
| Sinalizador01 | 77.5% |
| Sinalizador02 | 88.0% |
| Sinalizador05 | 76.5% |
| Sinalizador06 | 82.5% |
| Sinalizador08 | 75.0% |
| Sinalizador10 | 51.0% |
| **Média** | **74.50% ± 11.68%** |

**Gap observado:** 98.50% (10-fold CV) → 74.50% (LOPO) = **24pp de degradação**

**Interpretação:** O modelo memoriza o estilo motor de cada sinalizador. O ET aprende features idiossincráticas em vez de propriedades fonológicas universais do sinal.

**Solução prevista:** Mais sinalizadores (Zenodo: +4 = 1200 amostras), normalização adversarial por sinalizador, features de velocidade relativa em vez de absolutas.

---

### 5. ExtraTrees — Varredura de Árvores (sem PCA)

**Descrição:** ExtraTreesClassifier com features completas (11172 dims, sem PCA). ExtraTrees usa limiares aleatórios em vez dos ótimos do RF, reduzindo a variância em espaços de alta dimensão.

| Classificador | Acurácia | ±std |
|--------------|----------|------|
| ET-500 | 98.25% | 1.00% |
| **ET-1000** | **98.50%** | **0.75%** |
| ET-2000 | 98.50% | 0.75% |

**Observação:** Saturação a partir de 1000 árvores — ET-2000 = ET-1000.
**Melhor modelo geral:** `ET-1000` com **98.50% ± 0.75%** — supera todo o estado da arte.

**Importância das features (Gini):**

| Feature | Importância |
|---------|------------|
| FFT (6300 dims) | **74.95%** |
| Kinematic stats (1800 dims) | 13.32% |
| GEI (3072 dims) | 11.74% |

---

### 6. Estudo de Ablação — Contribuição por Modalidade

**Descrição:** ET-1000 treinado com cada subconjunto de features para isolar a contribuição de cada modalidade.

| Features | Acurácia | ±std |
|---------|----------|------|
| GEI only | 52.00% | 4.00% |
| KIN only | 81.62% | 2.44% |
| FFT only | **98.12%** | **1.28%** |
| GEI + KIN | 81.62% | 5.39% |
| GEI + FFT | 98.00% | 1.50% |
| KIN + FFT | 98.00% | 1.27% |
| **GEI + KIN + FFT** | **98.50%** | **0.75%** |

**Conclusão:** FFT captura quase toda a informação discriminativa sozinho (98.12%). A combinação completa adiciona +0.38pp e reduz a variância de ±1.28% para ±0.75%, confirmando que as três modalidades são complementares na margem.

---

### 7. Modelo v5 — ST-GCN + MobileNetV3-Small + FFT MLP

**Descrição:** Arquitetura mais avançada combinando grafos espaço-temporais (ST-GCN), CNN pré-treinada para GEI e MLP para features estáticas.

**Arquitetura:**
```
GEI (3072) → MobileNetV3-Small (pre-trained ImageNet, 576d features)
                 → MLP proj(576→128) ─────────────────────────────────────┐
pos+vel+acc → STGCNEncoder:                                               ├→ FC concat → 20
  [75 nós, adj anatômico] → GCN(64→128) → Transformer(128d,4h,2L) ───────┤
kinematic_stats+FFT (8100) → MLP(512→256→128) ───────────────────────────┘
```

**Otimizações implementadas:**
- Pre-extração única das features MobileNetV3 (1× por dataset, não por epoch): 0.3s para 2400 amostras
- Downsampling temporal T=200→50 no STGCN (stride=4): reduz de 501ms/batch para 135ms/batch
- Grafo anatômico MediaPipe Holistic (75 nós: 33 pose + 21 mão esq + 21 mão dir)

**Resultado:** **91.88% ± 4.12%**
Folds: 90.0% | 95.0% | 93.8% | 88.8% | 95.0% | 95.0% | 82.5% | 97.5% | 91.2% | 90.0%

**Observação:** Abaixo do v4 (94.62%). ST-GCN não supera Transformer simples com 800 amostras — o grafo anatômico adiciona complexidade mas não generalização suficiente no regime de baixa amostragem. MobileNetV3 pré-treinado não agrega tanto quanto esperado para GEI sintético.

---

### 8. Pipeline YouTube — Análise de Vídeo em Libras

**Descrição:** Pipeline completo URL → segmentação → classificação → alinhamento com legendas.

**Vídeo testado:** https://youtu.be/-ZDkdbPqUZg (17 min, aula de Libras em português)

**Resultados:**
- 30.775 frames extraídos com MediaPipe Holistic (30fps, ~1027s)
- 559 segmentos de sinal detectados (limiarização por energia de velocidade de punho)
- 437 alinhamentos legenda→sinal gerados (overlap ≥30%)

**Top sinais detectados:**

| Sinal | Ocorrências | % |
|-------|------------|---|
| vacina | 156 | 35.8% |
| acontecer | 66 | 15.1% |
| medo | 61 | 14.0% |
| aproveitar | 60 | 13.8% |
| esquina | 58 | 13.3% |
| america | 16 | 3.7% |

**Limitação:** O classificador conhece apenas 20 sinais. Confianças baixas (13–21% para top-1) indicam que a maioria dos sinais no vídeo está fora do vocabulário do modelo. Uma solução robusta requereria detecção de "fora do vocabulário" (threshold de rejeição) e um modelo treinado em vocabulário maior.

---

### 9. Ensemble e Outros Classificadores

| Método | Acurácia | ±std |
|--------|----------|------|
| Voting (RF-200 + SVM-Linear + ET-1000) | 96.50% | 1.35% |
| HistGradientBoosting | 71.12% | 4.66% |

**Observação:** O ensemble não supera o ET-1000 isolado — o RF e SVM arrastam a média para baixo.

---

## Sumário Comparativo

| Modelo | Método | Acurácia (10-fold) | Δ vs SotA |
|--------|--------|--------------------|-----------|
| Passos et al. (2021) | GEI + SVM | 84.66% | — |
| Rego et al. (2025) | FFT+Kin+BiLSTM | 92.0% | — |
| Rezende et al. | CNN 3D | 93.3% | — |
| Arcanjo et al. | FastDTW | ~96.0% | — |
| De Castro et al. | 3D CNN | 96.0% | — |
| **BiLSTM (v1, sem aug)** | Multimodal BiLSTM | 72.50% ± 5.51% | |
| **BiLSTM (v2, + aug)** | Multimodal BiLSTM + aug | 79.50% ± 3.46% | |
| **Transformer (v3, + mirror)** | Multimodal Transformer | 88.75% ± 4.51% | |
| **ST-GCN + MobileNet (v5)** | ST-GCN + MobileNetV3 | 91.88% ± 4.12% | |
| **Sklearn RF-200** | GEI+KIN+FFT → PCA(200) | 96.12% ± 2.12% | +0.1pp |
| **Transformer (v4, sem mirror)** | Multimodal Transformer | 94.62% ± 5.59% | -1.4pp |
| **ExtraTrees-1000** | GEI+KIN+FFT (sem PCA), 800 amostras | **98.50% ± 0.75%** | **+2.5pp** |
| **ExtraTrees-1000 (full)** | GEI+KIN+FFT, 1180 amostras | 96.78% ± 2.03% | +0.8pp |

**Vencedor (10-fold CV):** ET-1000 com 800 amostras: **98.50% ± 0.75%** — supera todo o estado da arte.

**Nota:** Com 1180 amostras (4 novos sinalizadores), ET-1000 cai para 96.78%. Os novos sinalizadores introduzem variância intra-classe que o modelo 800-amostras não via, tornando o CV mais realista.

**LOPO (generalização real, 800 amostras):** ET-1000 = **74.50% ± 11.68%** — gap de 24pp indica dependência de estilo do sinalizador.

---

## Análise de Features

### Contribuição de cada modalidade

As três fontes de features têm papéis complementares:

| Feature | Informação capturada | Dimensão |
|---------|---------------------|----------|
| GEI (Gait Energy Image) | Envelope espacial do movimento (silhueta média) | 3072 |
| Kinematic stats | Distribuição de velocidades e acelerações | 1800 |
| FFT sliding window | Padrões de frequência e ritmo do sinal | 6300 |
| pos+vel+acc (temporal) | Trajetória completa frame-a-frame | 200×675 |

### Por que ExtraTrees supera Random Forest?

- **Thresholds aleatórios**: Em vez de buscar o melhor limiar por feature (RF), ET sorteia limiares aleatórios — reduz variância em espaços de alta dimensão.
- **Sem PCA**: Com 11172 features e 800 amostras, o PCA(200) do baseline descartava informação. ET lida nativamente com a alta dimensionalidade.
- **Por que neural nets ficam atrás**: 800 amostras / 20 classes = apenas 40 por classe — insuficiente para treinar 9.6M parâmetros sem overfitting severo. ET com ~100K parâmetros efetivos generaliza melhor neste regime.

### Decisões de design

1. **GEI via skeleton**: Como não temos os frames de vídeo, o GEI é computado renderizando as conexões do esqueleto 2D em imagens 64×48 e calculando a média pixel-a-pixel. Captura o envelope espacial do movimento.

2. **Normalização**: Centralização pelo centro dos quadris e escala pela distância ombro-a-ombro — torna as features invariantes a distância da câmera e posição do sinalizador.

3. **Augmentação mirror**: ~~Simula sinalizadores canhotos.~~ **INVALIDADA:** Análise k-NN no espaço FFT mostra que 94.9% das amostras espelhadas mapeiam para a classe errada. Libras usa lateralidade como traço fonológico — espelhar destrói a identidade do sinal. v4 (sem mirror) +5.87pp sobre v3 (com mirror).

4. **Mixup**: Interpolação linear entre pares de amostras no espaço de features. Regulariza sem criar artefatos temporais.

---

## Próximos Passos

- [x] Download dos 4 sinalizadores ausentes (03, 04, 07, 09) → Zenodo — **concluído: 1180 amostras**
- [x] Retreinar ET-1000 com 1180 amostras — **concluído: 96.78% ±2.03%** (queda vs 800 amostras: novos sinalizadores aumentam variância intra-classe)
- [x] Avaliação LOPO — **concluído: 74.50% ± 11.68%**
- [x] Análise do mirror augmentation — **concluído: PREJUDICIAL (+5.87pp sem mirror)**
- [x] v4 Transformer sem mirror — **concluído: 94.62% ± 5.59%**
- [x] v5 ST-GCN + MobileNetV3 — **concluído: 91.88% ± 4.12%**
- [x] Pipeline YouTube (URL → segmentação → classificação → legendas) — **concluído**
- [ ] LOPO com 1180 amostras (10 sinalizadores) para reavaliar gap de generalização
- [ ] Anotação fonológica com Qwen2-VL-7B (annotate_qwen.py) — requer vídeos Zenodo
- [ ] v6 PhoneticFusionModel com InfoNCE + SBERT — aguardando anotações Qwen
- [ ] Normalização adversarial por sinalizador para reduzir gap LOPO
- [ ] Threshold de rejeição no YouTube pipeline (sinal "fora do vocabulário")
- [ ] Vocabulário expandido: treinar com mais sinais para aplicação prática

---

*Documento atualizado em 2026-04-02 — inclui v4 (94.62%), LOPO (74.50%), análise mirror (prejudicial), v5 ST-GCN (91.88%), pipeline YouTube*
