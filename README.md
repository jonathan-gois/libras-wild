# Libras Wild — Anotação e Reconhecimento de Sinais

Ferramenta de anotação colaborativa para construir uma base de dados de Libras
capturada em vídeos reais ("in the wild").

## 🌐 Ferramenta de Anotação Online

**→ [Acessar ferramenta](https://SEU-USUARIO.github.io/libras-wild)**

494 segmentos de sinais detectados automaticamente aguardam validação.

---

## Como publicar no GitHub Pages (uma vez)

```bash
# 1. Crie o repositório no GitHub (público, nome: libras-wild)
#    https://github.com/new

# 2. No terminal da máquina local:
sudo apt install git        # se necessário
cd /home/jonathangois/Documentos/Libras_2026
git init
git add .
git commit -m "feat: ferramenta de anotacao Libras Wild"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/libras-wild.git
git push -u origin main

# 3. Ative GitHub Pages:
#    Repositório → Settings → Pages → Source: GitHub Actions
#    O workflow .github/workflows/pages.yml cuida do deploy automaticamente.

# 4. URL do site: https://SEU-USUARIO.github.io/libras-wild
```

Substitua `SEU-USUARIO` pelo seu nome de usuário do GitHub.

---

## Fluxo de Anotação

```
Anotador abre o site
    │
    ▼
Assiste ao clipe (YouTube embed, timestamps automáticos)
    │
    ▼
Valida: é sinal válido? Qual sinal? Parâmetros fonológicos?
    │
    ▼
Salva localmente (IndexedDB) → Exporta JSON
    │
    ▼
Envia o JSON via Issues ou PR para este repositório
    │
    ▼
export_wild_dataset.py mescla anotações → processed_data_wild.pkl
```

---

## Exportar dataset após coleta

```bash
source env/bin/activate
python3 src/export_wild_dataset.py \
    --annotations results/annotations_*.json \
    --landmarks   results/youtube/-ZDkdbPqUZg/landmarks.pkl \
    --segments    results/youtube/-ZDkdbPqUZg/segments.json \
    --output      data/processed_data_wild.pkl
```

---

## Adicionar novos vídeos à fila de anotação

```bash
# 1. Roda pipeline no novo vídeo
source env/bin/activate
python3 -c "
from src.youtube_pipeline import run_pipeline
run_pipeline('https://youtu.be/VIDEO_ID')
"

# 2. Atualiza docs/data/segments.json
python3 src/prepare_annotation_data.py

# 3. Commit e push → deploy automático
git add docs/data/
git commit -m 'data: adiciona segmentos de VIDEO_ID'
git push
```

---

## Resultados do Modelo (MINDS-Libras)

| Modelo | Acurácia (10-fold) |
|--------|-------------------|
| ExtraTrees-1000 | **98.50% ±0.75%** |
| Transformer v4 (sem mirror) | 94.62% ±5.59% |
| ST-GCN + MobileNetV3 v5 | 91.88% ±4.12% |
| LOPO (generalização real) | 74.50% ±11.68% |

Ver [RESULTADOS.md](RESULTADOS.md) para análise completa.
